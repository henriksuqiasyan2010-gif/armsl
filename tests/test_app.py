"""Тесты сборки конвейера в приложении (src/app.py).

Тестируется ТОЛЬКО то, что имеет смысл тестировать без железа и без
цикла событий Tk: FramePipeline (детекция -> признаки -> предсказание),
фильтр класса покоя и чистые функции отрисовки слова.

Намеренно НЕ тестируются: создание виджетов Tkinter, планировщик after(),
реальная камера, показ видео, освобождение ресурсов при закрытии окна.
Юнит-тест на них проверял бы собственный мок или требовал живое железо
и дисплей — эта часть проверяется запуском приложения руками.
"""

import numpy as np
import pytest

from src import config
from src.app import (
    COLOR_BACKGROUND,
    COLOR_WORD,
    WORD_DISPLAY_SECONDS,
    WORD_FADE_SECONDS,
    FramePipeline,
    blend_colors,
    filter_rest_label,
    word_display_state,
)
from src.landmarks import HandResult


# --- фейки -------------------------------------------------------------------


class FakeTracker:
    """Отдаёт заранее заданный набор рук и считает вызовы detect()."""

    def __init__(self, hands_per_frame: list[list[HandResult]]):
        self._hands_per_frame = hands_per_frame
        self.calls = 0

    def detect(self, frame_bgr, timestamp_ms):
        index = min(self.calls, len(self._hands_per_frame) - 1)
        self.calls += 1
        return self._hands_per_frame[index]


class FakePredictor:
    """Записывает всё, что в него запушили, и отдаёт заданный результат."""

    def __init__(self, result=(None, 0.0, [])):
        self.pushed: list[np.ndarray] = []
        self._result = result

    def push(self, vector):
        self.pushed.append(np.array(vector, copy=True))

    def result(self):
        return self._result


def _hand(handedness: str = "Right") -> HandResult:
    landmarks = np.array(
        [[0.5 + 0.01 * i, 0.5 + 0.02 * i, 0.0] for i in range(config.NUM_LANDMARKS)],
        dtype=np.float32,
    )
    return HandResult(handedness=handedness, landmarks=landmarks, handedness_score=0.99)


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# --- FramePipeline: счётчик рук ---------------------------------------------


@pytest.mark.parametrize(
    "hands, expected_count",
    [
        ([], 0),
        ([_hand("Right")], 1),
        ([_hand("Left"), _hand("Right")], 2),
    ],
)
def test_pipeline_reports_hands_count(hands, expected_count):
    pipeline = FramePipeline(FakeTracker([hands]), FakePredictor())

    result = pipeline.process(_frame(), timestamp_ms=0)

    assert len(result.hands) == expected_count


# --- FramePipeline: push ровно раз на кадр, с вектором нужной формы ---------


def test_pipeline_pushes_one_vector_per_frame():
    predictor = FakePredictor()
    tracker = FakeTracker([[_hand("Right")]])
    pipeline = FramePipeline(tracker, predictor)

    for timestamp_ms in (0, 33, 66):
        pipeline.process(_frame(), timestamp_ms)

    assert tracker.calls == 3
    assert len(predictor.pushed) == 3
    for vector in predictor.pushed:
        assert vector.shape == (config.FEATURE_VECTOR_SIZE,)


def test_pipeline_pushes_vector_built_from_tracker_output():
    """В предиктор уходит именно то, что нашёл трекер: правая рука есть,
    левой нет -> её половина нулевая, флаги расставлены."""
    predictor = FakePredictor()
    pipeline = FramePipeline(FakeTracker([[_hand("Right")]]), predictor)

    pipeline.process(_frame(), timestamp_ms=0)

    vector = predictor.pushed[0]
    assert np.all(vector[: config.HAND_VECTOR_SIZE] == 0.0)  # левой руки нет
    assert vector[-2] == 0.0  # флаг левой
    assert vector[-1] == 1.0  # флаг правой


# --- FramePipeline: работа без модели ----------------------------------------


def test_pipeline_without_predictor_does_not_crash():
    """Модель ещё не обучена — видео и руки работают, распознавания нет."""
    tracker = FakeTracker([[_hand("Right"), _hand("Left")]])
    pipeline = FramePipeline(tracker, predictor=None)

    result = pipeline.process(_frame(), timestamp_ms=0)

    assert len(result.hands) == 2
    assert result.word is None
    assert result.confidence == 0.0
    assert result.top_k == []
    assert tracker.calls == 1


# --- фильтр класса покоя ------------------------------------------------------


def test_filter_rest_label_is_noop_when_rest_not_configured(monkeypatch):
    monkeypatch.setattr(config, "REST_LABEL_ID", None)
    assert filter_rest_label("barev") == "barev"
    assert filter_rest_label(None) is None


def test_filter_rest_label_hides_rest_word(monkeypatch):
    monkeypatch.setattr(config, "REST_LABEL_ID", "rest")
    assert filter_rest_label("rest") is None
    assert filter_rest_label("barev") == "barev"


def test_pipeline_hides_rest_word_but_keeps_it_in_top_k(monkeypatch):
    monkeypatch.setattr(config, "REST_LABEL_ID", "rest")
    top_k = [("rest", 0.91), ("barev", 0.05), ("jur", 0.02)]
    predictor = FakePredictor(result=("rest", 0.91, top_k))
    pipeline = FramePipeline(FakeTracker([[_hand()]]), predictor)

    result = pipeline.process(_frame(), timestamp_ms=0)

    assert result.word is None  # пользователю "покой" не показываем
    assert result.confidence == 0.0
    assert result.top_k == top_k  # а в гипотезах он остаётся как есть


def test_pipeline_passes_normal_word_through(monkeypatch):
    monkeypatch.setattr(config, "REST_LABEL_ID", "rest")
    predictor = FakePredictor(result=("barev", 0.93, [("barev", 0.93)]))
    pipeline = FramePipeline(FakeTracker([[_hand()]]), predictor)

    result = pipeline.process(_frame(), timestamp_ms=0)

    assert result.word == "barev"
    assert result.confidence == pytest.approx(0.93)


# --- показ слова: удержание и угасание ---------------------------------------


def test_word_display_state_without_word_is_empty():
    text, color = word_display_state(None, None, now=100.0)
    assert text == ""
    assert color == COLOR_BACKGROUND


def test_word_display_state_shows_fresh_word_at_full_brightness():
    text, color = word_display_state("barev", word_shown_at=100.0, now=100.1)
    assert text == "barev"
    assert color == COLOR_WORD


def test_word_display_state_fades_before_disappearing():
    shown_at = 100.0
    # Середина фазы угасания: слово ещё видно, но цвет уже не исходный.
    middle_of_fade = shown_at + WORD_DISPLAY_SECONDS - WORD_FADE_SECONDS / 2
    text, color = word_display_state("barev", shown_at, now=middle_of_fade)

    assert text == "barev"
    assert color not in (COLOR_WORD, COLOR_BACKGROUND)


def test_word_display_state_disappears_after_display_seconds():
    shown_at = 100.0
    text, color = word_display_state("barev", shown_at, now=shown_at + WORD_DISPLAY_SECONDS + 0.01)

    assert text == ""
    assert color == COLOR_BACKGROUND


def test_blend_colors_endpoints_and_middle():
    assert blend_colors("#ffffff", "#000000", 1.0) == "#ffffff"
    assert blend_colors("#ffffff", "#000000", 0.0) == "#000000"
    assert blend_colors("#ffffff", "#000000", 0.5) == "#808080"
    # Значения вне диапазона не должны ломать формат цвета.
    assert blend_colors("#ffffff", "#000000", 5.0) == "#ffffff"
    assert blend_colors("#ffffff", "#000000", -5.0) == "#000000"
