"""Приложение реального времени: камера -> landmarks -> признаки -> слово на экране.

Запуск из корня проекта:

    python -m src.app            # камера по умолчанию
    python -m src.app --camera 1

Устройство модуля (три слоя, снизу вверх):
    FramePipeline -- сборка конвейера на один кадр, без Tk и без камеры;
                     это то, что покрыто тестами;
    CameraWorker  -- отдельный ПОТОК: чтение кадров, детекция, инференс;
    AppWindow     -- Tkinter: виджеты, root.after(), кнопки.

Почему тяжёлая работа в отдельном потоке: cap.read() блокирует до прихода
кадра (~33 мс), плюс MediaPipe, плюс инференс. Делай это в главном потоке —
и цикл событий Tk встанет на всё это время: окно перестанет перерисовываться,
кнопки перестанут нажиматься.

Почему predictor.result() вызывается В ВОРКЕРЕ, а не в UI: RealtimePredictor
не потокобезопасен, push() и result() мутируют общее состояние (история,
отложенное слово, top-K) и обязаны идти из одного цикла. Развести их по
потокам — получить плавающий баг, который не ловится в отладчике. Поэтому
воркер публикует уже ГОТОВЫЙ результат, а UI-поток предиктор не трогает.
"""

from __future__ import annotations

import argparse
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from src import config
from src.features import frame_to_vector
from src.landmarks import HandResult, HandTracker
from src.predictor import RealtimePredictor

# --- Константы оформления ---------------------------------------------------
# Они намеренно живут здесь, а не в config.py: по правилу из CLAUDE.md в
# config идут контрактные и модельные величины, а оформление остаётся там,
# где используется.

DEFAULT_CAMERA_INDEX = 0
UI_REFRESH_MS = 33  # ~30 обновлений экрана в секунду
WORKER_JOIN_TIMEOUT_SECONDS = 2.0

# Сколько секунд распознанное слово держится на экране и сколько из них
# уходит на плавное угасание (последняя секунда).
WORD_DISPLAY_SECONDS = 3.0
WORD_FADE_SECONDS = 1.0

FPS_SMOOTHING = 0.9  # то же сглаживание, что в demo_tracking.py

COLOR_BACKGROUND = "#1e1e1e"
COLOR_TEXT = "#e0e0e0"
COLOR_SECONDARY = "#9e9e9e"
COLOR_WORD = "#4dd0e1"
COLOR_LEFT_HAND = (255, 160, 0)   # BGR, как в demo_tracking.py
COLOR_RIGHT_HAND = (0, 200, 255)

FONT_WORD = ("Segoe UI", 72, "bold")   # крупно: слово должно читаться с 3 метров
FONT_CONFIDENCE = ("Segoe UI", 20)
FONT_TOP_K = ("Consolas", 16)
FONT_INFO = ("Segoe UI", 14)
FONT_STATUS = ("Segoe UI", 11)

POINT_RADIUS = 3
LINE_THICKNESS = 2

# Скелет кисти: пары индексов точек. Дубль таблицы из demo_tracking.py —
# осознанный: demo_tracking.py по правилам проекта CLI-скрипт, а не
# библиотека, импортировать из него нельзя.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


# --- Тестируемое ядро: конвейер на один кадр --------------------------------


def filter_rest_label(word: str | None) -> str | None:
    """Убрать из показа "класс покоя" (человек не жестикулирует).

    В predictor.py никакой логики для класса покоя нет и не должно быть:
    он там побеждает голосование наравне со всеми. Прятать его от
    пользователя — задача интерфейса, то есть этого модуля.

    config.REST_LABEL_ID читается при каждом вызове, а не кэшируется:
    сейчас там None (метки покоя в labels.csv ещё нет), и фильтр —
    пустая операция, но код и тест на него уже существуют.

    top-K это НЕ фильтрует: там покой показывается как есть, чтобы было
    видно, что модель вообще думает.
    """
    if word is None:
        return None
    if config.REST_LABEL_ID is not None and word == config.REST_LABEL_ID:
        return None
    return word


class PipelineResult(NamedTuple):
    """Что конвейер выдаёт по одному кадру."""

    hands: list[HandResult]        # для отрисовки скелета
    word: str | None               # УЖЕ отфильтрованное слово (или None)
    confidence: float
    top_k: list[tuple[str, float]]


class FramePipeline:
    """Сборка готовых частей на один кадр: детекция -> признаки -> предсказание.

    Ни Tk, ни камеры, ни потоков — поэтому именно этот класс покрыт тестами.
    Трекер и предиктор внедряются снаружи: в тестах это фейки, в бою —
    настоящие HandTracker и RealtimePredictor.

    predictor=None -- допустимое, штатное состояние: модель ещё не обучена.
    Тогда работает всё остальное (видео, скелет рук, счётчик рук, FPS),
    а распознавания просто нет. Это нужно уже сейчас: датасет ещё не снят,
    а приложением во время съёмки пользоваться хочется.
    """

    def __init__(self, tracker, predictor=None) -> None:
        self._tracker = tracker
        self._predictor = predictor

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PipelineResult:
        hands = self._tracker.detect(frame_bgr, timestamp_ms)

        if self._predictor is None:
            return PipelineResult(hands=hands, word=None, confidence=0.0, top_k=[])

        # Вектор кадра СЫРОЙ, без normalize_window: нормализацию делает сам
        # предиктор, когда наберётся полное окно (см. его докстринг).
        self._predictor.push(frame_to_vector(hands))
        word, confidence, top_k = self._predictor.result()

        shown_word = filter_rest_label(word)
        if shown_word is None:
            confidence = 0.0

        return PipelineResult(hands=hands, word=shown_word, confidence=confidence, top_k=top_k)


def draw_hands(frame_bgr: np.ndarray, hands: list[HandResult]) -> None:
    """Нарисовать на кадре скелеты найденных рук (на месте, in-place)."""
    height, width = frame_bgr.shape[:2]

    for hand in hands:
        points = [(int(x * width), int(y * height)) for x, y, _z in hand.landmarks]
        color = COLOR_LEFT_HAND if hand.handedness == "Left" else COLOR_RIGHT_HAND

        for start_index, end_index in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[start_index], points[end_index], color, LINE_THICKNESS)
        for point in points:
            cv2.circle(frame_bgr, point, POINT_RADIUS, color, -1)


# --- Обмен данными между потоками -------------------------------------------


@dataclass
class UiSnapshot:
    """Снимок состояния для отрисовки. Только простые данные, никаких виджетов."""

    frame_bgr: np.ndarray | None = None
    hands_count: int = 0
    fps: float = 0.0
    top_k: list[tuple[str, float]] = field(default_factory=list)
    word: str | None = None
    confidence: float = 0.0
    word_shown_at: float | None = None  # time.monotonic() последнего слова
    status: str = "Камера не запущена"
    running: bool = False


class SharedState:
    """Один снимок последнего состояния под замком — НЕ очередь.

    Для видео нужен самый свежий кадр: очередь при тормозящем интерфейсе
    копила бы отставание и показывала прошлое. Здесь воркер просто
    перезаписывает последнее состояние, а UI читает его копию.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = UiSnapshot()

    def publish(self, **changes) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, **changes)

    def read(self) -> UiSnapshot:
        with self._lock:
            return replace(self._snapshot)


# --- Рабочий поток ------------------------------------------------------------


class CameraWorker(threading.Thread):
    """Поток камеры: чтение кадров, детекция, инференс, публикация снимка.

    Виджеты Tk отсюда не трогаются вообще никогда — только SharedState.
    """

    def __init__(self, state: SharedState, camera_index: int = DEFAULT_CAMERA_INDEX) -> None:
        super().__init__(daemon=True)
        self._state = state
        self._camera_index = camera_index
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        capture = None
        tracker = None
        try:
            self._state.publish(status="Открываю камеру…")
            capture = cv2.VideoCapture(self._camera_index)
            if not capture.isOpened():
                self._state.publish(
                    status=f"Камера с индексом {self._camera_index} недоступна", running=False
                )
                return

            tracker = HandTracker()
            pipeline = FramePipeline(tracker, self._load_predictor())
            self._loop(capture, pipeline)
        except Exception as error:  # поток не должен умирать молча
            self._state.publish(status=f"Ошибка: {error}", running=False)
        finally:
            if capture is not None:
                capture.release()
            if tracker is not None:
                tracker.close()
            self._state.publish(running=False)

    def _load_predictor(self) -> RealtimePredictor | None:
        """Загрузить модель, если она есть. Её отсутствие — не авария.

        Загрузка идёт здесь, в воркере, а не в UI-потоке: импорт TensorFlow
        и load_model занимают секунды, и окно на это время замерло бы.
        """
        self._state.publish(status="Загружаю модель…")
        try:
            predictor = RealtimePredictor()
        except (FileNotFoundError, ValueError) as error:
            first_line = str(error).splitlines()[0]
            self._state.publish(status=f"Распознавание выключено: {first_line}")
            return None
        self._state.publish(status="Модель загружена, распознавание включено")
        return predictor

    def _loop(self, capture, pipeline: FramePipeline) -> None:
        start_time = time.perf_counter()
        last_timestamp_ms = -1
        last_frame_time = start_time
        fps = 0.0
        self._state.publish(running=True)

        while not self._stop_event.is_set():
            ok, frame_bgr = capture.read()
            if not ok:
                self._state.publish(
                    status="Кадр не получен — камера отключилась?", running=False
                )
                return

            # HandLandmarker в режиме VIDEO требует строго растущий timestamp.
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            timestamp_ms = max(elapsed_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            result = pipeline.process(frame_bgr, timestamp_ms)
            draw_hands(frame_bgr, result.hands)

            now = time.perf_counter()
            frame_seconds = now - last_frame_time
            last_frame_time = now
            if frame_seconds > 0:
                current_fps = 1.0 / frame_seconds
                fps = current_fps if fps == 0.0 else (
                    FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * current_fps
                )

            changes = {
                "frame_bgr": frame_bgr,
                "hands_count": len(result.hands),
                "fps": fps,
                "top_k": result.top_k,
            }
            if result.word is not None:
                # Защёлкиваем слово: result() отдаёт его ровно один раз, и
                # UI, опрашивающий состояние ~30 раз в секунду, иначе просто
                # не поймал бы его.
                changes["word"] = result.word
                changes["confidence"] = result.confidence
                changes["word_shown_at"] = time.monotonic()
            self._state.publish(**changes)


# --- Отрисовка ----------------------------------------------------------------


def blend_colors(foreground: str, background: str, alpha: float) -> str:
    """Смешать два цвета "#rrggbb": alpha=1 -> foreground, alpha=0 -> background."""
    alpha = max(0.0, min(1.0, alpha))
    fg = tuple(int(foreground[i : i + 2], 16) for i in (1, 3, 5))
    bg = tuple(int(background[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(int(round(f * alpha + b * (1 - alpha))) for f, b in zip(fg, bg))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def word_display_state(
    word: str | None, word_shown_at: float | None, now: float
) -> tuple[str, str]:
    """Что показывать в поле слова прямо сейчас: (текст, цвет).

    Слово держится WORD_DISPLAY_SECONDS секунд, последнюю WORD_FADE_SECONDS
    из них плавно угасает в фон, потом исчезает совсем — чтобы на демо не
    висело слово, распознанное минуту назад.
    """
    if word is None or word_shown_at is None:
        return "", COLOR_BACKGROUND

    elapsed = now - word_shown_at
    if elapsed >= WORD_DISPLAY_SECONDS:
        return "", COLOR_BACKGROUND

    fade_start = WORD_DISPLAY_SECONDS - WORD_FADE_SECONDS
    alpha = 1.0 if elapsed <= fade_start else 1.0 - (elapsed - fade_start) / WORD_FADE_SECONDS
    return word, blend_colors(COLOR_WORD, COLOR_BACKGROUND, alpha)


class AppWindow:
    """Окно Tkinter. Ничего не считает — только показывает снимок состояния."""

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX) -> None:
        self._camera_index = camera_index
        self._state = SharedState()
        self._worker: CameraWorker | None = None
        self._after_id: str | None = None
        self._photo: ImageTk.PhotoImage | None = None  # ссылку надо держать, иначе Tk выбросит картинку

        self._root = tk.Tk()
        self._root.title("ArmSL — распознавание армянского жестового языка")
        self._root.configure(bg=COLOR_BACKGROUND)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        content = tk.Frame(self._root, bg=COLOR_BACKGROUND)
        content.pack(fill="both", expand=True, padx=12, pady=12)

        self._video_label = tk.Label(content, bg=COLOR_BACKGROUND, text="Видео", fg=COLOR_SECONDARY)
        self._video_label.grid(row=0, column=0, sticky="nw")

        right = tk.Frame(content, bg=COLOR_BACKGROUND)
        right.grid(row=0, column=1, sticky="nw", padx=(20, 0))

        self._word_label = tk.Label(
            right, text="", font=FONT_WORD, bg=COLOR_BACKGROUND, fg=COLOR_WORD, anchor="w"
        )
        self._word_label.pack(anchor="w")

        self._confidence_label = tk.Label(
            right, text="", font=FONT_CONFIDENCE, bg=COLOR_BACKGROUND, fg=COLOR_SECONDARY, anchor="w"
        )
        self._confidence_label.pack(anchor="w")

        self._top_k_label = tk.Label(
            right, text="", font=FONT_TOP_K, bg=COLOR_BACKGROUND, fg=COLOR_TEXT,
            justify="left", anchor="w",
        )
        self._top_k_label.pack(anchor="w", pady=(20, 0))

        self._info_label = tk.Label(
            right, text="", font=FONT_INFO, bg=COLOR_BACKGROUND, fg=COLOR_TEXT, anchor="w"
        )
        self._info_label.pack(anchor="w", pady=(20, 0))

        buttons = tk.Frame(self._root, bg=COLOR_BACKGROUND)
        buttons.pack(fill="x", padx=12, pady=(0, 6))

        self._start_button = tk.Button(buttons, text="Старт", width=12, command=self._on_start)
        self._start_button.pack(side="left")
        self._stop_button = tk.Button(
            buttons, text="Стоп", width=12, command=self._on_stop, state="disabled"
        )
        self._stop_button.pack(side="left", padx=(8, 0))

        self._status_label = tk.Label(
            self._root, text="", font=FONT_STATUS, bg=COLOR_BACKGROUND, fg=COLOR_SECONDARY, anchor="w"
        )
        self._status_label.pack(fill="x", padx=12, pady=(0, 10))

    def run(self) -> None:
        self._tick()
        self._root.mainloop()

    def _tick(self) -> None:
        self._render(self._state.read())
        self._after_id = self._root.after(UI_REFRESH_MS, self._tick)

    def _render(self, snapshot: UiSnapshot) -> None:
        if snapshot.frame_bgr is not None:
            frame_rgb = cv2.cvtColor(snapshot.frame_bgr, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self._video_label.configure(image=self._photo, text="")

        word_text, word_color = word_display_state(
            snapshot.word, snapshot.word_shown_at, time.monotonic()
        )
        self._word_label.configure(text=word_text, fg=word_color)
        self._confidence_label.configure(
            text=f"уверенность: {snapshot.confidence:.2f}" if word_text else ""
        )

        if snapshot.top_k:
            top_k_text = "\n".join(f"{label:<20} {probability:.2f}" for label, probability in snapshot.top_k)
        else:
            top_k_text = "гипотез пока нет"
        self._top_k_label.configure(text=top_k_text)

        self._info_label.configure(
            text=f"FPS: {snapshot.fps:.1f}\nрук в кадре: {snapshot.hands_count}"
        )
        self._status_label.configure(text=snapshot.status)

        self._start_button.configure(state="disabled" if snapshot.running else "normal")
        self._stop_button.configure(state="normal" if snapshot.running else "disabled")

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = CameraWorker(self._state, self._camera_index)
        self._worker.start()

    def _on_stop(self) -> None:
        self._stop_worker()
        self._state.publish(status="Остановлено", running=False, frame_bgr=None)

    def _stop_worker(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()
        self._worker.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
        self._worker = None

    def _on_close(self) -> None:
        # Сначала снять запланированный after(), иначе колбэк выстрелит уже
        # после destroy() и Tk ругнётся на несуществующие виджеты.
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
            self._after_id = None
        self._stop_worker()
        self._root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="ArmSL: распознавание жестов в реальном времени")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="индекс камеры")
    args = parser.parse_args()

    AppWindow(camera_index=args.camera).run()


if __name__ == "__main__":
    main()
