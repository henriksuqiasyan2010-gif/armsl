"""Извлечение ключевых точек рук из кадра через MediaPipe Tasks API.

Обёртка вокруг HandLandmarker: принимает кадр от OpenCV (BGR), отдаёт
список найденных рук с координатами 21 точки и меткой Left/Right.
Используется и при записи датасета (recorder.py), и в реальном времени (app.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src import config


@dataclass
class HandResult:
    """Одна найденная в кадре рука.

    handedness       -- метка руки: "Left" или "Right" (как её назвал MediaPipe).
    landmarks        -- массив формы (21, 3): по строке на точку, координаты (x, y, z).
                        x, y — нормализованные координаты в кадре (0..1),
                        z — относительная «глубина» точки относительно запястья.
    handedness_score -- насколько модель уверена в метке Left/Right (0..1).
                        Полезно для отладки: если рука повёрнута ребром,
                        уверенность падает и метка может «прыгать».
    """

    handedness: str
    landmarks: np.ndarray
    handedness_score: float


class HandTracker:
    """Детектор рук поверх MediaPipe HandLandmarker в режиме видео.

    Режим VIDEO (а не IMAGE) выбран потому, что MediaPipe в нём использует
    результат предыдущего кадра для трекинга: это и быстрее на CPU, и точки
    меньше «дрожат» между соседними кадрами. Плата за это — каждый следующий
    вызов detect() обязан идти со строго большим timestamp_ms.
    """

    def __init__(self) -> None:
        model_path = config.HAND_LANDMARKER_MODEL_PATH

        # Отсутствие файла модели — это ошибка настройки проекта, а не
        # «рабочая ситуация». Падаем сразу и с понятным текстом, а не
        # молча отдаём пустые результаты на каждом кадре.
        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найден файл модели MediaPipe: {model_path}\n"
                "Скачай hand_landmarker.task со страницы MediaPipe Hand Landmarker "
                "и положи его в папку models/. В git этот файл не хранится."
            )

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[HandResult]:
        """Найти руки на кадре.

        frame_bgr    -- кадр как его отдаёт OpenCV: numpy-массив (высота, ширина, 3)
                        с порядком каналов BGR.
        timestamp_ms -- время кадра в миллисекундах. Должно строго расти от вызова
                        к вызову, иначе MediaPipe в режиме VIDEO бросит исключение.

        Возвращает список найденных рук (0, 1 или 2 элемента). Если рук в кадре
        нет — возвращается пустой список: это нормальная ситуация, а не ошибка.
        """
        # OpenCV хранит кадр в порядке каналов BGR, а MediaPipe ждёт RGB.
        # Конвертируем здесь, внутри detect(), а не у вызывающего кода —
        # тогда «кадр от камеры» остаётся единственным форматом, который знают
        # recorder.py, app.py и демо-скрипты.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        hands: List[HandResult] = []
        # result.hand_landmarks и result.handedness — два списка одинаковой длины:
        # i-я рука в одном соответствует i-й руке в другом.
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            points = np.array(
                [[point.x, point.y, point.z] for point in landmarks],
                dtype=np.float32,
            )
            # handedness[i] — список вариантов по убыванию уверенности,
            # нулевой элемент — самый уверенный ("Left" или "Right").
            top_label = handedness[0]
            hands.append(
                HandResult(
                    handedness=top_label.category_name,
                    landmarks=points,
                    handedness_score=float(top_label.score),
                )
            )
        return hands

    def close(self) -> None:
        """Освободить ресурсы MediaPipe. Вызывать при завершении работы."""
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Закрываем в любом случае — даже если внутри with произошла ошибка.
        self.close()
