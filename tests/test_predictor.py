"""Тесты инференса и предсказаний (src/predictor.py).

Всё на синтетике: модель замокана, часы замоканы, камеры нет и TensorFlow
не нужен. Тестируется логика предиктора (буфер, шаг инференса, порог,
голосование, кулдаун), а не качество самой GRU.
"""

import hashlib
import json

import numpy as np
import pytest

from src import config
from src.predictor import RealtimePredictor, _vote, verify_classes_snapshot

CLASSES = ["barev", "jur", "lav"]


# --- фейки -------------------------------------------------------------------


class FakeModel:
    """Модель-заглушка: отдаёт заранее заданные вероятности и считает вызовы.

    probabilities_sequence -- список векторов вероятностей, по одному на
    инференс. Когда список кончается, повторяется последний элемент —
    так тесту не нужно расписывать вероятности на все инференсы подряд.
    """

    def __init__(self, probabilities_sequence: list[list[float]]):
        self._sequence = [np.array(p, dtype=np.float32) for p in probabilities_sequence]
        self.calls = 0

    def __call__(self, batch, training=False):
        index = min(self.calls, len(self._sequence) - 1)
        self.calls += 1
        return np.array([self._sequence[index]], dtype=np.float32)


class FakeClock:
    """Управляемые часы вместо time.monotonic — чтобы не спать в тестах."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _confident(label: str) -> list[float]:
    """Вероятности с уверенным перевесом в сторону label (выше порога)."""
    probabilities = [0.02] * len(CLASSES)
    probabilities[CLASSES.index(label)] = 0.96
    return probabilities


def _unconfident(label: str) -> list[float]:
    """Вероятности, где label лидирует, но НЕ дотягивает до порога."""
    probabilities = [(1.0 - 0.4) / (len(CLASSES) - 1)] * len(CLASSES)
    probabilities[CLASSES.index(label)] = 0.4
    return probabilities


def _write_snapshot(path, classes: list[str], *, break_fingerprint: bool = False) -> None:
    """Снимок классов в том же формате, что пишет train.save_model_with_labels_snapshot.

    sha256 считается здесь независимо от кода предиктора — так тест
    заодно закрепляет сам рецепт отпечатка, а не просто повторяет реализацию.
    """
    payload = json.dumps(classes, ensure_ascii=False).encode("utf-8")
    fingerprint = hashlib.sha256(payload).hexdigest()
    if break_fingerprint:
        fingerprint = "0" * 64

    path.write_text(
        json.dumps(
            {
                "classes": classes,
                "classes_sha256": fingerprint,
                "saved_at": "2026-01-01T00:00:00+00:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_labels_csv(path, classes: list[str]) -> None:
    rows = "".join(f"{label},Բարև,{label},перевод\n" for label in classes)
    path.write_text("label_id,armenian,pronunciation,meaning\n" + rows, encoding="utf-8")


@pytest.fixture
def predictor_files(tmp_path, monkeypatch):
    """labels.csv + снимок классов во временной папке, пути подменены в config."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    labels_csv = data_dir / "labels.csv"
    _write_labels_csv(labels_csv, CLASSES)

    snapshot_path = models_dir / "gesture_gru.classes.json"
    _write_snapshot(snapshot_path, CLASSES)

    monkeypatch.setattr(config, "LABELS_CSV_PATH", labels_csv)
    monkeypatch.setattr(config, "GRU_CLASSES_SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(config, "GRU_MODEL_PATH", models_dir / "gesture_gru.keras")

    return {"labels_csv": labels_csv, "snapshot_path": snapshot_path, "models_dir": models_dir}


def _frame_vector(offset: float = 0.0) -> np.ndarray:
    """Сырой вектор одного кадра: правая рука найдена, левой нет."""
    vector = np.zeros(config.FEATURE_VECTOR_SIZE, dtype=np.float32)
    right_hand = np.linspace(0.1, 0.9, config.HAND_VECTOR_SIZE, dtype=np.float32) + offset
    vector[config.HAND_VECTOR_SIZE : 2 * config.HAND_VECTOR_SIZE] = right_hand
    vector[-1] = 1.0  # флаг видимости правой руки
    return vector


def _push_frames(predictor: RealtimePredictor, count: int) -> None:
    for i in range(count):
        predictor.push(_frame_vector(offset=0.001 * i))


def _frames_for_inferences(count: int) -> int:
    """Сколько кадров нужно, чтобы произошло ровно count инференсов."""
    return config.WINDOW_LENGTH + (count - 1) * config.PREDICT_STRIDE


# --- _vote: чистая функция, сердце сглаживания ------------------------------


def test_vote_accepts_four_of_five():
    history = [("barev", 0.9), ("barev", 0.9), ("jur", 0.8), ("barev", 0.9), ("barev", 0.9)]
    voted = _vote(history)
    assert voted is not None
    assert voted[0] == "barev"
    assert voted[1] == pytest.approx(0.9)


def test_vote_rejects_three_of_five():
    history = [("barev", 0.9), ("barev", 0.9), ("jur", 0.8), ("jur", 0.8), ("barev", 0.9)]
    assert _vote(history) is None


def test_vote_rejects_incomplete_history():
    """4 из 4 в самом начале работы словом стать не должны."""
    history = [("barev", 0.9)] * (config.SMOOTHING_WINDOW - 1)
    assert _vote(history) is None


def test_vote_ignores_unconfident_entries():
    """None занимает слот, но проголосовать не может."""
    history = [("barev", 0.9), (None, 0.4), (None, 0.4), ("barev", 0.9), ("barev", 0.9)]
    assert _vote(history) is None


# --- сверка снимка классов ---------------------------------------------------


def test_verify_snapshot_passes_on_exact_match():
    snapshot = {
        "classes": CLASSES,
        "classes_sha256": hashlib.sha256(
            json.dumps(CLASSES, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    verify_classes_snapshot(snapshot, CLASSES)  # не должно бросать


def test_predictor_rejects_label_added_after_training(predictor_files):
    """В labels.csv дописали метку после сохранения снимка — работать нельзя."""
    _write_labels_csv(predictor_files["labels_csv"], CLASSES + ["tun"])

    with pytest.raises(ValueError, match="tun"):
        RealtimePredictor(model=FakeModel([_confident("barev")]))


def test_predictor_rejects_reordered_labels(predictor_files):
    """Тот же набор меток, но другой порядок — индексы softmax съехали."""
    _write_labels_csv(predictor_files["labels_csv"], ["jur", "barev", "lav"])

    with pytest.raises(ValueError, match="ПОРЯДОК"):
        RealtimePredictor(model=FakeModel([_confident("barev")]))


def test_predictor_rejects_hand_edited_snapshot(predictor_files):
    """Список в снимке поправили руками, хеш не пересчитали."""
    _write_snapshot(predictor_files["snapshot_path"], CLASSES, break_fingerprint=True)

    with pytest.raises(ValueError, match="sha256"):
        RealtimePredictor(model=FakeModel([_confident("barev")]))


def test_predictor_requires_snapshot_file(predictor_files):
    predictor_files["snapshot_path"].unlink()

    with pytest.raises(FileNotFoundError):
        RealtimePredictor(model=FakeModel([_confident("barev")]))


# --- буфер и шаг инференса ---------------------------------------------------


def test_result_is_none_until_buffer_is_full(predictor_files):
    model = FakeModel([_confident("barev")])
    predictor = RealtimePredictor(model=model, time_source=FakeClock())

    _push_frames(predictor, config.WINDOW_LENGTH - 1)

    assert predictor.result() == (None, 0.0, [])
    assert model.calls == 0


def test_first_inference_happens_when_buffer_fills(predictor_files):
    model = FakeModel([_confident("barev")])
    predictor = RealtimePredictor(model=model, time_source=FakeClock())

    _push_frames(predictor, config.WINDOW_LENGTH)

    assert model.calls == 1


def test_inference_runs_once_per_stride(predictor_files):
    model = FakeModel([_confident("barev")])
    predictor = RealtimePredictor(model=model, time_source=FakeClock())

    total_frames = config.WINDOW_LENGTH + 4 * config.PREDICT_STRIDE
    _push_frames(predictor, total_frames)

    expected_calls = 1 + (total_frames - config.WINDOW_LENGTH) // config.PREDICT_STRIDE
    assert model.calls == expected_calls == 5


def test_push_rejects_wrong_shape(predictor_files):
    predictor = RealtimePredictor(model=FakeModel([_confident("barev")]), time_source=FakeClock())

    with pytest.raises(ValueError):
        predictor.push(np.zeros(config.FEATURE_VECTOR_SIZE + 1, dtype=np.float32))


# --- порог уверенности и голосование сквозь push/result ---------------------


def test_word_counted_after_four_confident_of_five(predictor_files):
    model = FakeModel(
        [_confident("barev"), _confident("barev"), _confident("jur"),
         _confident("barev"), _confident("barev")]
    )
    predictor = RealtimePredictor(model=model, time_source=FakeClock())

    _push_frames(predictor, _frames_for_inferences(5))
    word, confidence, top_k = predictor.result()

    assert word == "barev"
    assert confidence == pytest.approx(0.96, abs=1e-3)
    assert top_k[0][0] == "barev"


def test_word_not_counted_on_three_of_five(predictor_files):
    model = FakeModel(
        [_confident("barev"), _confident("barev"), _confident("jur"),
         _confident("jur"), _confident("barev")]
    )
    predictor = RealtimePredictor(model=model, time_source=FakeClock())

    _push_frames(predictor, _frames_for_inferences(5))
    word, _, _ = predictor.result()

    assert word is None


def test_low_confidence_not_counted_but_present_in_top_k(predictor_files):
    model = FakeModel([_unconfident("barev")])
    predictor = RealtimePredictor(model=model, time_source=FakeClock())

    _push_frames(predictor, _frames_for_inferences(config.SMOOTHING_WINDOW))
    word, confidence, top_k = predictor.result()

    assert word is None
    assert confidence == 0.0
    # Слово не засчитано, но видно, что модель о нём думает.
    assert top_k[0][0] == "barev"
    assert top_k[0][1] == pytest.approx(0.4)
    assert len(top_k) == config.TOP_K_PREDICTIONS


# --- кулдаун ------------------------------------------------------------------


def test_cooldown_blocks_repeat_until_it_expires(predictor_files):
    clock = FakeClock()
    model = FakeModel([_confident("barev")])  # всегда уверенно "barev"
    predictor = RealtimePredictor(model=model, time_source=clock)

    _push_frames(predictor, _frames_for_inferences(config.SMOOTHING_WINDOW))
    first_word, _, _ = predictor.result()
    assert first_word == "barev"

    # Время стоит: сколько ни жестикулируй, слово повторно не засчитается.
    _push_frames(predictor, config.PREDICT_STRIDE * config.SMOOTHING_WINDOW)
    assert predictor.result()[0] is None

    # Кулдаун истёк — то же самое слово снова можно засчитать.
    clock.advance(config.COOLDOWN_SECONDS + 0.01)
    _push_frames(predictor, config.PREDICT_STRIDE)
    assert predictor.result()[0] == "barev"


# --- неидемпотентность result() ----------------------------------------------


def test_result_returns_word_only_once(predictor_files):
    model = FakeModel([_confident("barev")])
    predictor = RealtimePredictor(model=model, time_source=FakeClock())

    _push_frames(predictor, _frames_for_inferences(config.SMOOTHING_WINDOW))

    assert predictor.result()[0] == "barev"
    # Повторный вызов без нового push() — слова уже нет, но top-K остаётся.
    word, confidence, top_k = predictor.result()
    assert word is None
    assert confidence == 0.0
    assert top_k[0][0] == "barev"
