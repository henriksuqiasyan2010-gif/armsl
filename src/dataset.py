"""Загрузка датасета из data/, сборка выборок и разбиение на train/val/test.

Этот файл отвечает только за запись и чтение сэмплов (record_sample,
load_dataset) и справочник меток (load_labels). Разбиение по группам
person+session (split_by_group) и аугментация (augment) — следующий слой
поверх (X, y, groups), сюда пока не входят.
"""

from __future__ import annotations

import csv
import warnings
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np

from src import config
from src.features import frame_to_vector, normalize_window
from src.landmarks import HandResult


def load_labels(labels_csv_path: Path | None = None) -> dict[str, dict]:
    """Прочитать справочник меток жестов из data/labels.csv.

    labels_csv_path -- путь к csv; по умолчанию config.LABELS_CSV_PATH.
                        Параметр в первую очередь для тестов (подсунуть свой
                        временный файл), в обычной работе не передаётся.

    Возвращает dict: label_id -> {"armenian": ..., "pronunciation": ...,
    "meaning": ...}. label_id — ASCII-транслитерация, именно она используется
    в коде и путях к файлам (data/raw/<label_id>/...); остальные поля —
    только справочные, для человека.

    ПОРЯДОК ВАЖЕН: data/labels.csv — единственный источник истины по классам,
    и порядок строк в нём = порядок классов = индекс выхода softmax модели.
    Возвращаемый dict сохраняет порядок строк файла (dict в Python помнит
    порядок вставки), поэтому list(load_labels()) даёт список классов в том
    самом порядке, в котором их должна выдавать модель.

    Строки можно только дописывать в конец: если переставить или удалить
    строку, индексы поедут, и обученная модель начнёт показывать не те слова.
    Дубликат label_id по той же причине — ошибка (ValueError): второй
    такой же ключ молча затёр бы первый и сдвинул все следующие индексы.
    """
    if labels_csv_path is None:
        labels_csv_path = config.LABELS_CSV_PATH

    if not labels_csv_path.exists():
        raise FileNotFoundError(
            f"Не найден файл со списком меток: {labels_csv_path}\n"
            "Без него record_sample/record_session.py не могут проверить, "
            "что метка не опечатка."
        )

    labels: dict[str, dict] = {}
    with open(labels_csv_path, encoding="utf-8", newline="") as csv_file:
        for row_number, row in enumerate(csv.DictReader(csv_file), start=2):
            label_id = row["label_id"]
            if label_id in labels:
                raise ValueError(
                    f"{labels_csv_path}, строка {row_number}: label_id {label_id!r} "
                    "встречается дважды. Дубликат сдвинул бы индексы классов, "
                    "и модель показывала бы не те слова."
                )
            labels[label_id] = {
                "armenian": row["armenian"],
                "pronunciation": row["pronunciation"],
                "meaning": row["meaning"],
            }
    return labels


def record_sample(
    label: str,
    window: Sequence[list[HandResult]],
    person: str,
    session: str,
    *,
    lighting: str = "",
    background: str = "",
    distance_m: float | None = None,
    notes: str = "",
) -> None:
    """Обработать окно кадров и сохранить готовый сэмпл на диск.

    label   -- label_id из data/labels.csv.
    window  -- ровно config.WINDOW_LENGTH элементов, каждый — то, что
               возвращает HandTracker.detect() за один кадр (list[HandResult],
               0/1/2 найденные руки). Это СЫРЫЕ кадры, до frame_to_vector.
    person, session -- кто и в какой сессии записал сэмпл. Обязательны, а не
               опциональны: без них нет ни имени файла по контракту
               ({person}_{session}_{idx}.npy), ни строки в meta.csv, ни
               группы для train/val-разбиения (см. CLAUDE.md — разбиение
               по группам "критично для честности результатов").
    lighting, background, distance_m, notes -- необязательные поля meta.csv,
               для ручной пометки условий записи.

    НА ДИСК СОХРАНЯЮТСЯ СЫРЫЕ ВЕКТОРЫ — результат frame_to_vector без
    normalize_window. Нормализация применяется позже, при чтении в
    load_dataset(). Причина: нормализация — это часть препроцессинга, а не
    самих данных. Если сохранять уже нормализованное:
    - нельзя сравнить "с нормализацией / без" — сырых чисел уже не осталось;
    - нельзя поменять сам алгоритм нормализации, не перезаписав весь
      накопленный датасет заново (а перезаписать его нельзя — жесты
      переснимать пришлось бы руками);
    - нельзя честно проверить окно короче WINDOW_LENGTH: у обрезанного
      окна origin/scale должны считаться по нему самому, а в уже
      нормализованном файле они посчитаны по полному окну.
    Сырые данные переживают смену препроцессинга, нормализованные — нет.

    ФОРМАТ ХРАНЕНИЯ — один .npy на сэмпл + общий data/meta.csv (см. контракт
    данных в CLAUDE.md; это не выбор этой функции, а уже принятое решение
    проекта, здесь только реализация). Почему это удобно:
    - Новый сэмпл = один новый файл + одна новая строка csv. Старые файлы
      при этом вообще не открываются на запись — в отличие от одного общего
      архива с append, где неудачная запись (например, программа упала
      посреди сохранения) могла бы повредить весь накопленный датасет.
    - Один битый файл (плохая запись, обрыв по питанию) — это потеря одного
      сэмпла, а не всего архива целиком.
    - Каждый сэмпл можно открыть/прослушать/удалить вручную по отдельности —
      это важно именно сейчас, когда сэмплов десятки-сотни и качество
      записи ещё нужно проверять глазами, а не автоматическими метриками.

    Исключения (не сохраняет сэмпл, если):
    - label не найден в data/labels.csv -- ValueError. Так неизвестная
      метка (например, опечатка) никогда не создаст свою собственную папку
      в data/raw/ незаметно для автора.
    - len(window) != config.WINDOW_LENGTH -- ValueError. Контракт "окно
      всегда ровно WINDOW_LENGTH кадров" не должен молча нарушаться —
      кадр короче или длиннее просто не считается валидным сэмплом.
    - ни на одном кадре окна не найдено ни одной руки -- ValueError. Это
      проверка на уровне API, а не только в CLI: если позже, например,
      augment.py случайно сгенерирует пустое окно, это тоже будет пойман
      здесь, а не тихо просочится в датасет как мусорный сэмпл.
    """
    labels = load_labels()
    if label not in labels:
        available = ", ".join(sorted(labels))
        raise ValueError(f"Неизвестная метка {label!r}. Доступные метки: {available}")

    if len(window) != config.WINDOW_LENGTH:
        raise ValueError(
            f"Окно должно содержать ровно {config.WINDOW_LENGTH} кадров, "
            f"получено {len(window)}."
        )

    raw = np.stack([frame_to_vector(frame_hands) for frame_hands in window]).astype(np.float32)

    # Последние 2 числа каждого кадра — флаги видимости левой/правой руки
    # (контракт данных, см. CLAUDE.md). Если они все нулевые — рук не было
    # вообще ни на одном кадре, весь сэмпл — брак записи.
    if not raw[:, -2:].any():
        raise ValueError(
            "В этом окне ни на одном кадре не найдено ни одной руки "
            "(человек не попал в кадр?). Сэмпл не сохранён."
        )

    label_dir = config.RAW_DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    idx = _next_sample_index(label_dir, person, session)
    file_path = label_dir / f"{person}_{session}_{idx}.npy"
    # Сохраняем именно raw, без normalize_window — см. объяснение в докстринге.
    np.save(file_path, raw)

    _append_meta_row(
        {
            "file": file_path.relative_to(config.DATA_DIR).as_posix(),
            "label": label,
            "person": person,
            "session": session,
            "date": date.today().isoformat(),
            "lighting": lighting,
            "background": background,
            "distance_m": "" if distance_m is None else distance_m,
            "notes": notes,
        }
    )


def _next_sample_index(label_dir: Path, person: str, session: str) -> int:
    """Следующий свободный idx для файла {person}_{session}_{idx}.npy.

    Сканируем уже существующие файлы с таким же префиксом person_session_
    и берём максимальный найденный idx + 1 (с нуля, если файлов ещё нет).
    Префикс сравнивается целиком (а не через split("_")), поэтому person
    или session с подчёркиванием внутри не путают разбор.
    """
    prefix = f"{person}_{session}_"
    max_idx = -1
    for path in label_dir.glob(f"{prefix}*.npy"):
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def _append_meta_row(row: dict) -> None:
    """Дописать одну строку в data/meta.csv, создав файл с заголовком,
    если его ещё нет."""
    meta_csv_path = config.META_CSV_PATH
    meta_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_is_new = not meta_csv_path.exists() or meta_csv_path.stat().st_size == 0

    with open(meta_csv_path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=config.META_CSV_COLUMNS)
        if file_is_new:
            writer.writeheader()
        writer.writerow(row)


def load_dataset(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загрузить весь накопленный датасет из data/meta.csv.

    data_dir -- папка, где лежат meta.csv и подпапка raw/ (по умолчанию
                config.DATA_DIR). Параметр — для тестов, чтобы не трогать
                настоящие data/ проекта.

    Источник истины — meta.csv, а не скан папок data/raw/: для каждой
    строки пытаемся прочитать соответствующий .npy файл. Битая или
    отсутствующая строка (файла нет, .npy не читается или неправильной
    формы) пропускается с warnings.warn — она не должна ронять загрузку
    всего датасета из-за одного плохого сэмпла.

    НОРМАЛИЗАЦИЯ ПРИМЕНЯЕТСЯ ЗДЕСЬ, ПРИ ЧТЕНИИ. На диске лежат сырые
    векторы (см. record_sample), normalize_window вызывается к каждому
    прочитанному сэмплу. Так препроцессинг можно менять, не переписывая
    сам датасет.

    Возвращает (X, y, groups):
      X      -- (N, WINDOW_LENGTH, FEATURE_VECTOR_SIZE), float32,
                УЖЕ нормализованные окна
      y      -- (N,) строковые метки
      groups -- (N,) строка "{person}__{session}" на сэмпл — нужна для
                будущего group-based split (чтобы один и тот же человек
                не попал одновременно в train и val).

    На пустом датасете (meta.csv нет или в нём нет строк) возвращает
    пустые массивы правильной формы, а не исключение.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    meta_csv_path = data_dir / "meta.csv"

    features_list: list[np.ndarray] = []
    labels_list: list[str] = []
    groups_list: list[str] = []

    if meta_csv_path.exists():
        with open(meta_csv_path, encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            # Строки csv нумеруем с 2 (строка 1 — заголовок) — так предупреждение
            # можно найти в файле глазами.
            for row_number, row in enumerate(reader, start=2):
                sample_path = data_dir / row["file"]
                try:
                    sample = np.load(sample_path)
                except (FileNotFoundError, OSError, ValueError, EOFError) as error:
                    warnings.warn(
                        f"meta.csv, строка {row_number}: не удалось прочитать "
                        f"{sample_path} ({error}). Сэмпл пропущен."
                    )
                    continue

                expected_shape = (config.WINDOW_LENGTH, config.FEATURE_VECTOR_SIZE)
                if sample.shape != expected_shape:
                    warnings.warn(
                        f"meta.csv, строка {row_number}: {sample_path} имеет форму "
                        f"{sample.shape}, ожидалось {expected_shape}. Сэмпл пропущен."
                    )
                    continue

                # На диске сырые векторы — нормализуем при чтении.
                features_list.append(normalize_window(sample.astype(np.float32)))
                labels_list.append(row["label"])
                groups_list.append(f"{row['person']}__{row['session']}")

    if features_list:
        X = np.stack(features_list)
    else:
        X = np.zeros((0, config.WINDOW_LENGTH, config.FEATURE_VECTOR_SIZE), dtype=np.float32)

    y = np.array(labels_list, dtype=str)
    groups = np.array(groups_list, dtype=str)
    return X, y, groups
