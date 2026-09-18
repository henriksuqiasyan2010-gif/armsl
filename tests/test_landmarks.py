"""Тесты извлечения ключевых точек (src/landmarks.py)."""

import numpy as np
import pytest

from src import config
from src.landmarks import HandTracker

# Файл модели hand_landmarker.task не хранится в git (см. .gitignore) и
# скачивается вручную. Без него тесты запустить нельзя — честно помечаем их
# как пропущенные, а не как упавшие: иначе отсутствие файла будет выглядеть
# как настоящая ошибка в коде.
pytestmark = pytest.mark.skipif(
    not config.HAND_LANDMARKER_MODEL_PATH.exists(),
    reason=(
        f"Не найден файл модели {config.HAND_LANDMARKER_MODEL_PATH}. "
        "Скачай hand_landmarker.task и положи его в папку models/."
    ),
)

# Размер тестового кадра — произвольный, важно лишь, что кадр валидный
# и трёхканальный (как у OpenCV).
FRAME_HEIGHT = 480
FRAME_WIDTH = 640


def make_black_frame() -> np.ndarray:
    """Пустой чёрный кадр в формате OpenCV (BGR, uint8). Рук на нём заведомо нет."""
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def test_black_frame_returns_empty_list():
    """Если рук в кадре нет, detect() возвращает пустой список, а не падает."""
    with HandTracker() as tracker:
        hands = tracker.detect(make_black_frame(), timestamp_ms=0)

    assert hands == []


def test_several_frames_in_a_row():
    """Несколько кадров подряд с растущим timestamp обрабатываются без ошибок.

    Режим VIDEO требует строго возрастающего времени — проверяем, что
    обычный сценарий «кадр за кадром» работает.
    """
    frame = make_black_frame()

    with HandTracker() as tracker:
        for timestamp_ms in (0, 33, 66, 99):
            hands = tracker.detect(frame, timestamp_ms)
            assert hands == []
