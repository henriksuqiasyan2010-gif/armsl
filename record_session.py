"""CLI-скрипт записи датасета: камера -> окно из WINDOW_LENGTH кадров -> record_sample.

Запуск из корня проекта:

    python record_session.py --label barev --person henrik --session s1

Управление: SPACE — начать запись одного сэмпла (после короткого отсчёта
на экране), Q/ESC — выход. Можно записать сколько угодно сэмплов подряд,
не перезапуская скрипт (idx на диске увеличивается автоматически).
"""

from __future__ import annotations

import argparse
import time

import cv2

from src import config
from src.dataset import load_labels, record_sample
from src.landmarks import HandTracker

# Константы отображения — только для этого скрипта, в config.py не идут
# (не часть контракта данных, чисто UI), по аналогии с demo_tracking.py.
COUNTDOWN_SECONDS = 3
COLOR_IDLE = (0, 200, 255)
COLOR_RECORDING = (0, 0, 255)
COLOR_OK = (0, 255, 0)
COLOR_WARN = (0, 0, 255)

STATE_IDLE = "idle"
STATE_COUNTDOWN = "countdown"
STATE_RECORDING = "recording"


def _parse_args() -> argparse.Namespace:
    # Список меток читаем заранее, чтобы --label можно было провалидировать
    # средствами самого argparse (choices) — при опечатке argparse сам
    # напечатает "invalid choice" и полный список доступных label_id.
    try:
        labels = load_labels()
    except FileNotFoundError as error:
        raise SystemExit(str(error))

    parser = argparse.ArgumentParser(description="Запись сэмплов жестов с веб-камеры")
    parser.add_argument("--label", required=True, choices=sorted(labels), help="label_id из data/labels.csv")
    parser.add_argument("--person", required=True, help="идентификатор человека, который записывает жест")
    parser.add_argument("--session", required=True, help="идентификатор сессии записи")
    parser.add_argument("--camera", type=int, default=0, help="индекс камеры (по умолчанию 0)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру с индексом {args.camera}")

    tracker = HandTracker()

    start_time = time.perf_counter()
    last_timestamp_ms = -1
    state = STATE_IDLE
    countdown_started_at = 0.0
    frames_hands: list = []
    saved_in_this_run = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                print("Кадр не получен, камера отключилась?")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 == Esc
                break

            if state == STATE_IDLE:
                if key == ord(" "):
                    state = STATE_COUNTDOWN
                    countdown_started_at = time.perf_counter()

                cv2.putText(
                    frame_bgr,
                    f"label={args.label}  person={args.person}  session={args.session}  saved={saved_in_this_run}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_IDLE,
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    "SPACE - записать сэмпл, Q/ESC - выход",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_IDLE,
                    2,
                )

            elif state == STATE_COUNTDOWN:
                remaining = COUNTDOWN_SECONDS - (time.perf_counter() - countdown_started_at)
                if remaining <= 0:
                    state = STATE_RECORDING
                    frames_hands = []
                else:
                    cv2.putText(
                        frame_bgr,
                        f"{remaining:.0f}",
                        (frame_bgr.shape[1] // 2 - 20, frame_bgr.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        COLOR_RECORDING,
                        4,
                    )

            elif state == STATE_RECORDING:
                # Timestamp считаем от общего старта скрипта, монотонно
                # растущий — HandLandmarker в режиме VIDEO требует именно этого.
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                timestamp_ms = max(elapsed_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                hands = tracker.detect(frame_bgr, timestamp_ms)
                frames_hands.append(hands)

                cv2.putText(
                    frame_bgr,
                    f"REC {len(frames_hands)}/{config.WINDOW_LENGTH}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COLOR_RECORDING,
                    2,
                )

                if len(frames_hands) >= config.WINDOW_LENGTH:
                    if not any(len(hands_in_frame) > 0 for hands_in_frame in frames_hands):
                        # Рука не найдена ни на одном из WINDOW_LENGTH кадров —
                        # явно не зовём record_sample, а не ловим её ValueError:
                        # так быстрее и понятнее, что происходит именно здесь.
                        print("ПРЕДУПРЕЖДЕНИЕ: рука не найдена ни на одном кадре — сэмпл не сохранён.")
                    else:
                        try:
                            record_sample(args.label, frames_hands, args.person, args.session)
                        except ValueError as error:
                            print(f"ПРЕДУПРЕЖДЕНИЕ: сэмпл не сохранён: {error}")
                        else:
                            saved_in_this_run += 1
                            print(f"Сэмпл сохранён (по счёту в этом запуске: {saved_in_this_run}).")

                    state = STATE_IDLE

            cv2.imshow("ArmSL - запись датасета (q - выход)", frame_bgr)
    finally:
        capture.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
