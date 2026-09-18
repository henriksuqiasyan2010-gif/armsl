"""Преобразование сырых ключевых точек в нормализованные признаки для модели.

Два шага конвейера:
    HandTracker.detect() -> list[HandResult]      (см. src/landmarks.py)
                          -> frame_to_vector()     -> вектор кадра (128,)
    60 таких векторов     -> normalize_window()    -> обученный вид (60,128)
"""

from __future__ import annotations

import numpy as np

from src import config
from src.landmarks import HandResult

# Срезы вектора кадра: первые 63 числа — левая рука, следующие 63 — правая,
# последние 2 — флаги видимости (см. контракт данных в CLAUDE.md).
_LEFT_SLICE = slice(0, config.HAND_VECTOR_SIZE)
_RIGHT_SLICE = slice(config.HAND_VECTOR_SIZE, 2 * config.HAND_VECTOR_SIZE)
_LEFT_FLAG_INDEX = 2 * config.HAND_VECTOR_SIZE
_RIGHT_FLAG_INDEX = 2 * config.HAND_VECTOR_SIZE + 1


def frame_to_vector(hands: list[HandResult]) -> np.ndarray:
    """Собрать вектор признаков одного кадра из списка найденных рук.

    hands -- результат HandTracker.detect() для одного кадра: 0, 1 или 2
             найденные руки.

    Возвращает np.float32 массив формы (128,): 63 числа левой руки,
    63 числа правой, 2 флага видимости (1.0 — рука найдена, 0.0 — нет).
    Раскладка идёт по метке hand.handedness, а не по позиции в списке —
    значит порядок элементов в hands результата не меняет.

    Метки MediaPipe используются как есть, без инверсии: проверено вживую
    (demo_tracking.py, кадр без зеркалирования) — когда физически поднимаешь
    правую руку, MediaPipe пишет "Right" (confidence ~0.98-0.99), левую —
    "Left" с такой же уверенностью. То есть метка руки совпадает
    с реальностью.
    """
    vector = np.zeros(config.FEATURE_VECTOR_SIZE, dtype=np.float32)

    for hand in hands:
        # (21, 3) -> (63,): порядок точек внутри руки как их отдаёт MediaPipe,
        # он нас не касается, лишь бы был одинаковым всегда (так и есть).
        flat_landmarks = hand.landmarks.reshape(-1).astype(np.float32)

        if hand.handedness == "Left":
            vector[_LEFT_SLICE] = flat_landmarks
            vector[_LEFT_FLAG_INDEX] = 1.0
        elif hand.handedness == "Right":
            vector[_RIGHT_SLICE] = flat_landmarks
            vector[_RIGHT_FLAG_INDEX] = 1.0
        else:
            # Такого не должно быть: MediaPipe в HandLandmarker отдаёт только
            # "Left"/"Right". Если появилось что-то другое — это сигнал о
            # поломке контракта где-то выше (например, кто-то подменил
            # HandResult руками), а не рабочая ситуация вроде "рук нет".
            # Падаем явно, а не тихо теряем руку.
            raise ValueError(
                f"Неизвестная метка handedness: {hand.handedness!r} "
                '(ожидалось "Left" или "Right")'
            )

    return vector


def normalize_window(window: np.ndarray) -> np.ndarray:
    """Убрать из окна положение человека в кадре и расстояние до камеры,
    сохранив при этом движение рук во времени и их взаимное расположение.

    window -- массив (T, 128): T кадров подряд, каждый в формате frame_to_vector().
              T обычно равен config.WINDOW_LENGTH (60), но функция не привязана
              к конкретной длине.

    Возвращает массив той же формы (T, 128).

    ПОЧЕМУ СДВИГ И МАСШТАБ СЧИТАЮТСЯ ОДИН РАЗ НА ВСЁ ОКНО, А НЕ НА КАЖДЫЙ
    КАДР ОТДЕЛЬНО:
    Если пересчитывать "где сейчас запястье" в каждом кадре и туда же его
    сдвигать, запястье во всех 60 кадрах окажется в одной и той же точке —
    а вместе с ним исчезнет и вся траектория движения руки. Для динамических
    жестов это фатально: многие жесты различаются именно движением, а не
    статичной формой кисти. Поэтому сдвиг (origin) и масштаб (scale) находим
    ОДИН РАЗ для всего окна и применяем эту же пару чисел ко всем кадрам —
    это как один раз применить пан и зум к целому видео в монтажной
    программе, а не к каждому кадру заново: расстояния МЕЖДУ кадрами
    (то есть само движение) не меняются, меняется только общее положение
    и масштаб всего клипа целиком.

    ПОЧЕМУ МЕДИАНА ПО ОКНУ, А НЕ СРЕДНЕЕ И НЕ ПЕРВЫЙ ВАЛИДНЫЙ КАДР:
    - Первый валидный кадр — самый ненадёжный вариант: если именно на нём
      случился глитч детекции (рука дрогнула, палец загнулся неудачно),
      весь дальнейший расчёт всего окна строится на одном плохом числе,
      и ничем не защищён.
    - Среднее по окну надёжнее, но одно выпадающее значение (например,
      один кадр со смазом при быстром движении, где размер ладони
      посчитался неправильно) утягивает средний результат за собой.
    - Медиана по кадрам устойчива к паре "плохих" кадров: если из 60 кадров
      55 нормальные и 5 с шумом, медиана их просто игнорирует, а среднее
      размазалось бы по всем 60.

    ЧТО ПРОИСХОДИТ С ОТСУТСТВУЮЩИМИ РУКАМИ:
    - Кадры, где рука не найдена, не участвуют в расчёте origin/scale для
      этой руки (там нечего усреднять).
    - На выходе такая рука остаётся ТОЧНЫМ нулём, независимо от вычисленных
      origin/scale — это решает флаг видимости, а не проверка "координата
      похожа на ноль" (иначе после вычитания origin у нулевой руки
      появились бы фиктивные ненулевые числа).
    - Если во всём окне рук не было вообще ни на одном кадре — считать
      медиану не из чего. В этом случае origin = 0, scale = 1.0: результат
      всё равно весь нулевой (раз рук не было), но без деления на ноль.

    ОБЩИЙ ORIGIN/SCALE НА ОБЕ РУКИ (а не по одному на каждую):
    Обе руки в кадре сдвигаются и масштабируются ОДИНАКОВО. Если бы каждая
    рука нормализовалась сама по себе (относительно своего же запястья
    и своего же размера ладони), обе руки после этого всегда оказывались
    бы в одной точке начала координат — и взаимное расположение рук
    (критично для двуручных жестов) было бы потеряно. Общий сдвиг+масштаб
    для пары рук — это перенос и растяжение всей фигуры целиком, поэтому
    расстояние и угол между руками не искажаются.
    """
    window = np.asarray(window, dtype=np.float32)
    num_frames = window.shape[0]
    result = window.copy()

    left = window[:, _LEFT_SLICE].reshape(num_frames, config.NUM_LANDMARKS, 3)
    right = window[:, _RIGHT_SLICE].reshape(num_frames, config.NUM_LANDMARKS, 3)
    left_present = window[:, _LEFT_FLAG_INDEX] > 0.5
    right_present = window[:, _RIGHT_FLAG_INDEX] > 0.5

    wrist_idx = config.WRIST_LANDMARK_INDEX
    palm_idx = config.PALM_SIZE_LANDMARK_INDEX

    # Шаг А: по каждому кадру, где найдена хоть одна рука, — среднее
    # запястий и средний "размер ладони" присутствующих в этом кадре рук.
    # Это промежуточные, ещё "по кадрам" значения — сама нормализация
    # по ним пока не делается.
    frame_origins: list[np.ndarray] = []
    frame_scales: list[float] = []

    for t in range(num_frames):
        wrists = []
        palm_sizes = []
        if left_present[t]:
            wrists.append(left[t, wrist_idx])
            palm_sizes.append(float(np.linalg.norm(left[t, palm_idx] - left[t, wrist_idx])))
        if right_present[t]:
            wrists.append(right[t, wrist_idx])
            palm_sizes.append(float(np.linalg.norm(right[t, palm_idx] - right[t, wrist_idx])))

        if wrists:
            frame_origins.append(np.mean(wrists, axis=0))
            frame_scales.append(float(np.mean(palm_sizes)))

    # Шаг Б: свернуть промежуточные значения по кадрам в одно origin
    # и один scale на всё окно — медианой (см. объяснение выше).
    if frame_origins:
        window_origin = np.median(np.stack(frame_origins, axis=0), axis=0)
        window_scale = float(np.median(frame_scales))
    else:
        # Во всём окне рук не найдено вообще — нормализовать нечего,
        # но и падать здесь не нужно: результат и так весь нулевой.
        window_origin = np.zeros(3, dtype=np.float32)
        window_scale = 1.0

    # Защита от деления на почти-ноль (вырожденный, "слипшийся" кадр).
    window_scale = max(window_scale, config.NORMALIZATION_EPSILON)

    # Шаг В: применяем ОДНУ и ту же пару (window_origin, window_scale)
    # ко всем кадрам сразу — вот тут и сохраняется траектория движения.
    left_normalized = (left - window_origin) / window_scale
    right_normalized = (right - window_origin) / window_scale

    # Отсутствующие руки — точный ноль, независимо от того, что насчиталось
    # выше (там всё равно были нули-заглушки, но после вычитания origin
    # они перестали бы быть нулями, если бы мы их не зануляли явно).
    left_normalized[~left_present] = 0.0
    right_normalized[~right_present] = 0.0

    result[:, _LEFT_SLICE] = left_normalized.reshape(num_frames, config.HAND_VECTOR_SIZE)
    result[:, _RIGHT_SLICE] = right_normalized.reshape(num_frames, config.HAND_VECTOR_SIZE)
    # Флаги видимости не нормализуются — они уже скопированы как есть
    # через result = window.copy() и не тронуты выше.

    return result
