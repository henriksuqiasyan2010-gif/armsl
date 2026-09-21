"""Инференс в реальном времени: поток кадров с камеры -> распознанное слово.

Конвейер на каждый кадр:
    frame_to_vector(hands) -> push(вектор кадра, СЫРОЙ)
                           -> кольцевой буфер на WINDOW_LENGTH кадров
    раз в PREDICT_STRIDE кадров:
        буфер -> normalize_window -> модель -> softmax
              -> порог уверенности -> голосование -> кулдаун -> слово

Нормализация живёт ЗДЕСЬ, а не у вызывающего кода, и иначе быть не может:
normalize_window считает origin и scale медианой по всему окну, у одного
кадра этих величин не существует. Поэтому буфер хранит сырые векторы —
ровно как data/raw хранит сырые окна, а нормализует их load_dataset.

Класс НЕ потокобезопасен: push() и result() должны вызываться из одного
цикла (у нас — цикл камеры в app.py). Общий буфер и история без блокировок.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src import config
from src.dataset import load_labels
from src.features import normalize_window


def _classes_fingerprint(class_list: list[str]) -> str:
    """sha256-отпечаток списка классов.

    ВНИМАНИЕ: рецепт обязан побитово совпадать с train._classes_fingerprint —
    там снимок пишется, здесь проверяется. Это осознанный дубль двух строк,
    а не забытый импорт: модуль инференса не должен зависеть от модуля
    обучения (иначе app.py потянул бы за собой весь обучающий код).
    Если меняешь рецепт здесь — поменяй и в src/train.py, иначе все ранее
    сохранённые снимки перестанут проходить проверку.
    """
    payload = json.dumps(class_list, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_classes_snapshot(snapshot: dict, current_classes: list[str]) -> None:
    """Сверить снимок классов, сохранённый вместе с моделью, с текущим labels.csv.

    Две независимые проверки:
    1. Снимок внутренне цел: пересчитанный sha256 от snapshot["classes"]
       совпадает с snapshot["classes_sha256"]. Ловит снимок, отредактированный
       руками (список поправили, хеш забыли) — без этой проверки поле с хешем
       было бы декорацией.
    2. Снимок совпадает с текущим data/labels.csv — и по составу, и по ПОРЯДКУ.
       Порядок критичен: индекс класса в этом списке = индекс выхода softmax
       (см. CLAUDE.md). Тот же набор меток в другом порядке — это модель,
       которая будет показывать не те слова.

    Ничего не возвращает, при расхождении бросает ValueError с описанием
    того, что именно разошлось.
    """
    for required_key in ("classes", "classes_sha256"):
        if required_key not in snapshot:
            raise ValueError(
                f"В снимке классов нет поля {required_key!r} — файл повреждён "
                "или сохранён другой версией train.py. Переобучи модель."
            )

    snapshot_classes = list(snapshot["classes"])

    expected_fingerprint = _classes_fingerprint(snapshot_classes)
    if snapshot["classes_sha256"] != expected_fingerprint:
        raise ValueError(
            "Снимок классов повреждён: sha256 не совпадает с самим списком "
            "классов внутри снимка (похоже, файл правили руками). "
            "Переобучи модель, чтобы снимок записался заново."
        )

    if snapshot_classes == current_classes:
        return

    added = [label for label in current_classes if label not in snapshot_classes]
    removed = [label for label in snapshot_classes if label not in current_classes]

    if not added and not removed:
        difference = (
            "тот же набор меток, но ДРУГОЙ ПОРЯДОК — индексы softmax съехали, "
            "модель показывала бы чужие слова"
        )
    else:
        parts = []
        if added:
            parts.append(f"в labels.csv добавлены: {added}")
        if removed:
            parts.append(f"из labels.csv удалены: {removed}")
        difference = "; ".join(parts)

    raise ValueError(
        "Модель обучена на другом списке классов, чем сейчас в data/labels.csv.\n"
        f"  в снимке ({len(snapshot_classes)}): {snapshot_classes}\n"
        f"  в labels.csv ({len(current_classes)}): {current_classes}\n"
        f"  разница: {difference}\n"
        "Работать с таким рассинхроном нельзя — переобучи модель: "
        "python -m src.train --model gru --val-groups ..."
    )


def _vote(history: Sequence[tuple[str | None, float]]) -> tuple[str, float] | None:
    """ЕДИНСТВЕННОЕ место в модуле, где рождается засчитанное слово.

    history -- последние инференсы (не кадры!) в виде (метка | None, уверенность).
               None означает "предсказание было неуверенным": такая запись
               занимает слот, но проголосовать не может.

    Возвращает (слово, средняя уверенность голосов-победителей) или None.
    Функция чистая и тотальная — никакого состояния, никакого времени,
    поэтому её можно исчерпывающе протестировать прямой подачей
    последовательностей. result() не умеет назначить слово никаким другим
    путём: засчитанное слово — это буквально возвращаемое значение _vote.

    Три условия, любое из которых даёт None:
    - история ещё не заполнена (в самом начале работы 3 из 3 согласных
      предсказаний словом не станут);
    - среди записей нет ни одной уверенной;
    - у лидера меньше SMOOTHING_MIN_VOTES голосов.
    """
    if len(history) < config.SMOOTHING_WINDOW:
        return None

    votes_by_label = Counter(label for label, _ in history if label is not None)
    if not votes_by_label:
        return None

    label, votes = votes_by_label.most_common(1)[0]
    if votes < config.SMOOTHING_MIN_VOTES:
        return None

    winning_confidences = [conf for lab, conf in history if lab == label]
    # Средняя уверенность именно голосов-победителей: честнее, чем взять
    # последнее значение.
    return label, float(np.mean(winning_confidences))


class RealtimePredictor:
    """Распознавание жестов в реальном времени поверх обученной GRU-модели.

    Использование в цикле камеры:
        predictor = RealtimePredictor()
        ...
        predictor.push(frame_to_vector(hands))
        word, confidence, top_k = predictor.result()
        if word is not None:
            # слово засчитано ровно один раз — добавляем его в предложение
    """

    def __init__(
        self,
        model=None,
        *,
        model_path: Path | None = None,
        snapshot_path: Path | None = None,
        labels_csv_path: Path | None = None,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        """Загрузить модель и сверить её список классов с data/labels.csv.

        model       -- уже готовая модель. None (обычный случай) -> грузим
                       с диска из model_path. Передают её в тестах: так весь
                       набор тестов обходится без TensorFlow и без обучения
                       настоящей сети. Модель должна быть вызываемой как
                       model(batch, training=False) и возвращать (1, n_classes)
                       вероятностей.
        model_path, snapshot_path, labels_csv_path -- переопределения путей
                       (по умолчанию из config, резолвятся ВНУТРИ функции,
                       а не как значения параметров по умолчанию — иначе
                       monkeypatch config.* в тестах не сработал бы).
        time_source -- источник времени для кулдауна. По умолчанию
                       time.monotonic: он не зависит от перевода системных
                       часов. Тесты подсовывают фейковые часы, чтобы не спать
                       по секунде на каждую проверку кулдауна.

        Сверка классов выполняется ВСЕГДА, в том числе когда модель передана
        готовой: рассинхрон модели и labels.csv — это именно то, что здесь
        обязано падать, а не работать молча.
        """
        if model_path is None:
            model_path = config.GRU_MODEL_PATH
        if snapshot_path is None:
            snapshot_path = config.GRU_CLASSES_SNAPSHOT_PATH

        # Опечатка в конфиге вида "7 из 5" (голосование не сработает никогда)
        # или "0 из 5" (сработает всегда) должна падать сразу, а не
        # превращаться в загадочное поведение на живой камере.
        if not 1 <= config.SMOOTHING_MIN_VOTES <= config.SMOOTHING_WINDOW:
            raise ValueError(
                f"Неверная настройка сглаживания: SMOOTHING_MIN_VOTES="
                f"{config.SMOOTHING_MIN_VOTES} при SMOOTHING_WINDOW="
                f"{config.SMOOTHING_WINDOW}. Должно быть 1 <= MIN_VOTES <= WINDOW."
            )
        if config.PREDICT_STRIDE < 1:
            raise ValueError(f"PREDICT_STRIDE должен быть >= 1, получено {config.PREDICT_STRIDE}")

        self._classes = list(load_labels(labels_csv_path))

        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Не найден снимок классов: {snapshot_path}\n"
                "Он сохраняется вместе с моделью при обучении. Обучи модель: "
                "python -m src.train --model gru --val-groups ..."
            )
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        verify_classes_snapshot(snapshot, self._classes)

        if model is None:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Не найден файл модели: {model_path}\n"
                    "Обучи модель: python -m src.train --model gru --val-groups ..."
                )
            from tensorflow import keras  # лениво: импорт TF стоит секунды

            model = keras.models.load_model(model_path)
        self._model = model

        # Длина окна и длина истории заданы самими контейнерами, а не
        # арифметикой: выйти за них невозможно.
        self._frames: deque[np.ndarray] = deque(maxlen=config.WINDOW_LENGTH)
        self._history: deque[tuple[str | None, float]] = deque(maxlen=config.SMOOTHING_WINDOW)

        # Стартуем "почти дозревшим", чтобы первый же полный буфер дал
        # инференс сразу, без лишнего ожидания PREDICT_STRIDE кадров.
        self._frames_since_inference = config.PREDICT_STRIDE - 1

        self._top_k: list[tuple[str, float]] = []
        self._pending_word: tuple[str, float] | None = None
        self._last_accepted_at: float | None = None
        self._time = time_source

    def push(self, vector: np.ndarray) -> None:
        """Добавить вектор очередного кадра (СЫРОЙ, из frame_to_vector).

        Вектор должен быть именно сырым, без normalize_window: нормализация
        применяется здесь же, но к целому окну, когда набираются все
        WINDOW_LENGTH кадров (см. docstring модуля).

        Сам инференс происходит не на каждый вызов, а раз в PREDICT_STRIDE
        кадров и только при полном буфере.
        """
        vector = np.asarray(vector, dtype=np.float32)
        expected_shape = (config.FEATURE_VECTOR_SIZE,)
        if vector.shape != expected_shape:
            raise ValueError(
                f"push() ждёт вектор одного кадра формы {expected_shape}, "
                f"получено {vector.shape}."
            )

        self._frames.append(vector)

        if len(self._frames) < config.WINDOW_LENGTH:
            return

        self._frames_since_inference += 1
        if self._frames_since_inference >= config.PREDICT_STRIDE:
            self._frames_since_inference = 0
            self._run_inference()

    def result(self) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Отдать засчитанное слово (если оно есть) и текущие гипотезы модели.

        Возвращает (слово | None, уверенность, top-K гипотез):
        - слово       -- засчитанное голосованием и прошедшее кулдаун, иначе None;
        - уверенность -- средняя уверенность голосов-победителей, 0.0 если слова нет;
        - top-K       -- [(метка, вероятность), ...] из ПОСЛЕДНЕГО инференса,
                         отсортированные по убыванию. Отдаются всегда: и когда
                         ничего не прошло порог, и во время кулдауна — чтобы GUI
                         мог показывать "как думает модель", а не замирать.
                         Пустой список, пока не было ни одного инференса.

        ВАЖНО: result() НЕ ИДЕМПОТЕНТНА. Слово отдаётся ровно один раз —
        повторный вызов result() без нового push() между ними вернёт None
        даже для того же самого кадра. Это легко перепутать при отладке:
        если в отладчике вызвать result() дважды подряд, второй вызов
        покажет None, и будет казаться, что слово потерялось.
        Так сделано потому, что основной потребитель (app.py) добавляет
        слова в предложение: отдавай модуль одно и то же слово 30 раз в
        секунду, GUI пришлось бы самому убирать дубли — а эта логика и есть
        смысл существования данного класса. Держать слово на экране GUI
        умеет своим собственным состоянием.
        """
        if self._pending_word is None:
            return None, 0.0, list(self._top_k)

        label, confidence = self._pending_word
        self._pending_word = None  # съедаем: слово отдаётся один раз
        return label, confidence, list(self._top_k)

    def _run_inference(self) -> None:
        """Прогнать текущее окно через модель и обновить историю/слово."""
        window = np.stack(self._frames)  # (WINDOW_LENGTH, FEATURE_VECTOR_SIZE), сырое
        normalized = normalize_window(window)
        batch = normalized[np.newaxis, ...]  # (1, WINDOW_LENGTH, FEATURE_VECTOR_SIZE)

        # Вызываем модель напрямую, а не через .predict(): для одного окна
        # в реальном времени predict() тратит заметно больше времени на
        # свою обвязку (батчинг, колбэки), чем на сам расчёт.
        # Последний слой модели — softmax, вероятности уже нормированы,
        # второй раз применять softmax не нужно.
        probabilities = np.asarray(self._model(batch, training=False), dtype=np.float64)[0]

        if probabilities.shape[0] != len(self._classes):
            raise RuntimeError(
                f"Модель вернула {probabilities.shape[0]} вероятностей, а классов "
                f"{len(self._classes)}. Модель и labels.csv не соответствуют друг другу."
            )

        order = np.argsort(probabilities)[::-1][: config.TOP_K_PREDICTIONS]
        self._top_k = [(self._classes[i], float(probabilities[i])) for i in order]

        best_index = int(np.argmax(probabilities))
        best_confidence = float(probabilities[best_index])
        if best_confidence >= config.CONFIDENCE_THRESHOLD:
            self._history.append((self._classes[best_index], best_confidence))
        else:
            # Неуверенное предсказание занимает слот, но голосовать не может:
            # дёрганый участок не должен становиться словом только потому,
            # что среди шума несколько раз мелькнула одна метка.
            self._history.append((None, best_confidence))

        voted = _vote(self._history)
        if voted is None:
            return

        now = self._time()
        if (
            self._last_accepted_at is not None
            and now - self._last_accepted_at < config.COOLDOWN_SECONDS
        ):
            # Кулдаун: слово не засчитывается, _pending_word остаётся пустым,
            # значит result() вернёт None — но top-K он всё равно отдаст.
            return

        self._pending_word = voted
        self._last_accepted_at = now
