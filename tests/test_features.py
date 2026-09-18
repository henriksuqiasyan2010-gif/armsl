"""Тесты построения и нормализации признаков (src/features.py)."""

import numpy as np

from src import config
from src.features import frame_to_vector, normalize_window
from src.landmarks import HandResult

# Индексы срезов вектора кадра — те же, что в features.py, но заведены
# здесь заново намеренно: тест должен проверять контракт (числа 0, 63,
# 126, 127 из CLAUDE.md), а не подстраиваться под внутренние имена модуля.
_LEFT_SLICE = slice(0, 63)
_RIGHT_SLICE = slice(63, 126)
_LEFT_FLAG = 126
_RIGHT_FLAG = 127


def _base_hand_landmarks(wrist=(0.5, 0.5, 0.0)) -> np.ndarray:
    """Синтетическая рука фиксированной формы: 21 точка со смещением от
    запястья по детерминированному, ненулевому паттерну — чтобы "размер
    ладони" (расстояние запястье -> точка 9) был заведомо не нулём.
    """
    wrist_arr = np.array(wrist, dtype=np.float32)
    offsets = np.array(
        [[0.01 * i, 0.02 * i, 0.0] for i in range(config.NUM_LANDMARKS)],
        dtype=np.float32,
    )
    return wrist_arr + offsets  # offsets[0] == 0, значит landmark[0] == wrist


def _make_hand_result(handedness: str, landmarks: np.ndarray, score: float = 0.99) -> HandResult:
    return HandResult(handedness=handedness, landmarks=landmarks.astype(np.float32), handedness_score=score)


def _static_window(landmarks: np.ndarray, handedness: str, num_frames: int) -> np.ndarray:
    """Окно из num_frames одинаковых кадров с одной и той же рукой (без движения)."""
    frame_vector = frame_to_vector([_make_hand_result(handedness, landmarks)])
    return np.tile(frame_vector, (num_frames, 1))


# --- frame_to_vector ---------------------------------------------------------


def test_frame_to_vector_order_in_list_does_not_matter():
    """Раскладка идёт по метке handedness, а не по позиции в списке hands."""
    left = _make_hand_result("Left", _base_hand_landmarks(wrist=(0.3, 0.4, 0.0)))
    right = _make_hand_result("Right", _base_hand_landmarks(wrist=(0.7, 0.4, 0.0)))

    vector_normal_order = frame_to_vector([left, right])
    vector_reversed_order = frame_to_vector([right, left])

    assert np.array_equal(vector_normal_order, vector_reversed_order)


def test_frame_to_vector_missing_hand_is_zero_with_correct_flags():
    """Если в кадре только одна рука, вторая — точный ноль, флаги расставлены верно."""
    right_landmarks = _base_hand_landmarks(wrist=(0.7, 0.4, 0.0))
    right = _make_hand_result("Right", right_landmarks)

    vector = frame_to_vector([right])

    assert np.all(vector[_LEFT_SLICE] == 0.0)
    assert vector[_LEFT_FLAG] == 0.0
    assert np.allclose(vector[_RIGHT_SLICE], right_landmarks.reshape(-1))
    assert vector[_RIGHT_FLAG] == 1.0


# --- normalize_window: инвариантность к масштабу (расстоянию до камеры) -----


def test_normalize_window_is_scale_invariant():
    """Один и тот же жест, снятый "близко" и "далеко" (координаты умножены
    на константу), должен после нормализации давать близкие векторы.
    """
    close_landmarks = _base_hand_landmarks(wrist=(0.5, 0.5, 0.0))
    far_landmarks = close_landmarks * 0.4  # имитация уменьшения при удалении от камеры

    window_close = _static_window(close_landmarks, "Right", num_frames=10)
    window_far = _static_window(far_landmarks, "Right", num_frames=10)

    normalized_close = normalize_window(window_close)
    normalized_far = normalize_window(window_far)

    assert np.allclose(normalized_close, normalized_far, atol=1e-4)


# --- normalize_window: сохранение траектории движения -----------------------


def test_normalize_window_preserves_motion_trajectory():
    """Рука, едущая по прямой в течение окна, не должна "схлопнуться" в точку:
    положение запястья в первом и последнем кадре обязано заметно различаться.
    """
    num_frames = 20
    vectors = []
    for t in range(num_frames):
        x = 0.1 + (0.9 - 0.1) * t / (num_frames - 1)
        landmarks = _base_hand_landmarks(wrist=(x, 0.5, 0.0))
        vectors.append(frame_to_vector([_make_hand_result("Right", landmarks)]))
    window = np.stack(vectors, axis=0)

    normalized = normalize_window(window)

    # Правая рука лежит в _RIGHT_SLICE, первые 3 числа блока — координаты
    # запястья (x, y, z), нам нужна x.
    wrist_x_first_frame = normalized[0, _RIGHT_SLICE.start]
    wrist_x_last_frame = normalized[-1, _RIGHT_SLICE.start]

    # Порог не "лишь бы не ноль", а заметно больше нуля: движение в исходных
    # координатах было 0.8, размер ладони в нашей синтетической руке ~0.2,
    # значит в нормализованном пространстве разница должна быть порядка 4.
    assert abs(wrist_x_last_frame - wrist_x_first_frame) > 1.0


# --- normalize_window: устойчивость к отсутствию рук и вырожденным данным ---


def test_normalize_window_no_hands_at_all_stays_zero():
    """Если во всём окне ни одной руки не найдено, результат — те же нули,
    без NaN и без деления на ноль."""
    empty_window = np.zeros((8, config.FEATURE_VECTOR_SIZE), dtype=np.float32)

    result = normalize_window(empty_window)

    assert np.isfinite(result).all()
    assert np.array_equal(result, empty_window)


def test_normalize_window_degenerate_hand_no_nan_or_inf():
    """Рука с нулевым "размером ладони" (все точки слиплись в одну) не должна
    приводить к делению на ноль — есть epsilon-защита масштаба."""
    degenerate_landmarks = np.tile(np.array([0.5, 0.5, 0.0], dtype=np.float32), (config.NUM_LANDMARKS, 1))
    window = _static_window(degenerate_landmarks, "Left", num_frames=5)

    result = normalize_window(window)

    assert np.isfinite(result).all()


def test_normalize_window_random_inputs_never_produce_nan_or_inf():
    """Фуззинг-проверка: на случайных (в т.ч. "мусорных") входах не должно
    быть ни одного NaN/Inf в результате."""
    rng = np.random.default_rng(seed=42)

    for _ in range(20):
        num_frames = 15
        window = rng.normal(scale=5.0, size=(num_frames, config.FEATURE_VECTOR_SIZE)).astype(np.float32)
        # Флаги видимости по контракту — 0.0 или 1.0, а не произвольное число.
        window[:, config.FEATURE_VECTOR_SIZE - 2 :] = rng.integers(0, 2, size=(num_frames, 2)).astype(np.float32)

        result = normalize_window(window)

        assert np.isfinite(result).all()
