"""Демо-скрипт: живая проверка HandTracker на веб-камере.

Запуск из корня проекта:

    python demo_tracking.py            # камера по умолчанию
    python demo_tracking.py --camera 1 # если камер несколько

Выход — клавиша 'q'. Скрипт лежит вне src/ намеренно: это инструмент
для глазами-проверки детекции, а не часть продукта.
"""

from __future__ import annotations

import argparse
import time

import cv2

from src.landmarks import HandResult, HandTracker

# Скелет кисти: пары индексов точек, которые соединяем линиями.
# Это топология модели MediaPipe Hand (какая точка с какой связана),
# прописана здесь явно, чтобы не тянуть mediapipe.solutions.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),           # большой палец
    (0, 5), (5, 6), (6, 7), (7, 8),           # указательный
    (5, 9), (9, 10), (10, 11), (11, 12),      # средний
    (9, 13), (13, 14), (14, 15), (15, 16),    # безымянный
    (13, 17), (17, 18), (18, 19), (19, 20),   # мизинец
    (0, 17),                                  # основание ладони
)

# Параметры отрисовки (только для этого демо, в продукт не идут).
COLOR_LEFT = (255, 160, 0)    # BGR: синевато-голубой для левой руки
COLOR_RIGHT = (0, 200, 255)   # BGR: жёлто-оранжевый для правой
COLOR_FPS = (0, 255, 0)
POINT_RADIUS = 3
LINE_THICKNESS = 2
# Сглаживание FPS: доля нового замера в бегущем среднем. Без сглаживания
# число прыгает так, что его невозможно прочитать.
FPS_SMOOTHING = 0.9


def draw_hand(frame_bgr, hand: HandResult) -> None:
    """Нарисовать на кадре одну руку: точки, связи между ними и подпись L/R."""
    height, width = frame_bgr.shape[:2]

    # Координаты точек нормализованы (0..1) — переводим в пиксели кадра.
    points_px = [
        (int(x * width), int(y * height)) for x, y, _z in hand.landmarks
    ]

    color = COLOR_LEFT if hand.handedness == "Left" else COLOR_RIGHT

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame_bgr, points_px[start_idx], points_px[end_idx], color, LINE_THICKNESS)

    for point in points_px:
        cv2.circle(frame_bgr, point, POINT_RADIUS, color, -1)

    # Подпись рядом с запястьем (точка 0) — первая буква метки: L или R.
    wrist_x, wrist_y = points_px[0]
    label = hand.handedness[0]
    cv2.putText(
        frame_bgr,
        f"{label} {hand.handedness_score:.2f}",
        (wrist_x - 20, wrist_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо детекции рук через HandTracker")
    parser.add_argument("--camera", type=int, default=0, help="индекс камеры (по умолчанию 0)")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру с индексом {args.camera}")

    tracker = HandTracker()

    start_time = time.perf_counter()
    last_timestamp_ms = -1
    last_frame_time = start_time
    fps = 0.0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                print("Кадр не получен, камера отключилась?")
                break

            # Режим VIDEO требует строго возрастающий timestamp. Два кадра
            # могут прийти в одну и ту же миллисекунду — поэтому берём
            # максимум из «реального» времени и «предыдущий + 1».
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            timestamp_ms = max(elapsed_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            # Кадр отдаём в detect() как есть, в BGR: конвертацию в RGB
            # делает сам HandTracker.
            hands = tracker.detect(frame_bgr, timestamp_ms)

            for hand in hands:
                draw_hand(frame_bgr, hand)

            # FPS считаем по времени между кадрами и сглаживаем бегущим средним.
            now = time.perf_counter()
            frame_seconds = now - last_frame_time
            last_frame_time = now
            if frame_seconds > 0:
                current_fps = 1.0 / frame_seconds
                fps = current_fps if fps == 0.0 else FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * current_fps

            cv2.putText(
                frame_bgr,
                f"FPS: {fps:.1f}  hands: {len(hands)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_FPS,
                2,
            )

            # Кадр показываем НЕ зеркаля: важно видеть ровно то, что видит
            # модель, включая то, какую руку она назвала Left, а какую Right.
            cv2.imshow("ArmSL - demo tracking (q - выход)", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
