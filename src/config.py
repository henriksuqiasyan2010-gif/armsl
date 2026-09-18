"""Глобальные настройки проекта: пути, параметры камеры, гиперпараметры модели.

Здесь и только здесь живут все константы проекта. В остальных файлах
никаких «магических чисел» — импортируем отсюда.
"""

from pathlib import Path

# --- Пути ------------------------------------------------------------------

# Корень проекта (папка armsl/) вычисляется от расположения этого файла,
# поэтому пути работают одинаково, откуда бы ни запускали скрипт.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Файл модели детекции рук MediaPipe Tasks.
# Лежит локально, в git не хранится (см. .gitignore), скачивается вручную
# один раз. Во время работы программы ничего не скачиваем.
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# --- Параметры детекции рук ------------------------------------------------

# Обе руки: часть жестов армянского жестового языка двуручные.
NUM_HANDS = 2

# Число ключевых точек на одной руке в модели MediaPipe Hand Landmarker.
# Это свойство самой модели, а не наш выбор.
NUM_LANDMARKS = 21

# Пороги уверенности.
# ВАЖНО: это ДЕФОЛТНЫЕ значения из документации MediaPipe Tasks API
# (HandLandmarkerOptions), а не подобранные нами числа. Вынесены сюда явно,
# чтобы потом было видно, что именно можно крутить, если детекция шумит.
MIN_HAND_DETECTION_CONFIDENCE = 0.5   # дефолт MediaPipe: min_hand_detection_confidence
MIN_HAND_PRESENCE_CONFIDENCE = 0.5    # дефолт MediaPipe: min_hand_presence_confidence
MIN_TRACKING_CONFIDENCE = 0.5         # дефолт MediaPipe: min_tracking_confidence
