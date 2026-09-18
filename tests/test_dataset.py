"""Тесты загрузки и разбиения датасета (src/dataset.py)."""

import numpy as np
import pytest

from src import config
from src.dataset import load_dataset, load_labels, record_sample
from src.features import frame_to_vector, normalize_window
from src.landmarks import HandResult

KNOWN_LABEL = "barev"
UNKNOWN_LABEL = "not_a_real_label"


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Подменяет пути config.* на временную папку — тесты не должны
    трогать настоящие data/ проекта."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)

    labels_csv = data_dir / "labels.csv"
    labels_csv.write_text(
        "label_id,armenian,pronunciation,meaning\n"
        f"{KNOWN_LABEL},Բարև,barev,привет\n"
        "jur,Ջուր,jur,вода\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RAW_DATA_DIR", data_dir / "raw")
    monkeypatch.setattr(config, "META_CSV_PATH", data_dir / "meta.csv")
    monkeypatch.setattr(config, "LABELS_CSV_PATH", labels_csv)

    return data_dir


def _base_hand_landmarks(wrist=(0.5, 0.5, 0.0)) -> np.ndarray:
    """Та же синтетическая рука, что и в test_features.py: 21 точка
    с детерминированным ненулевым смещением от запястья."""
    wrist_arr = np.array(wrist, dtype=np.float32)
    offsets = np.array(
        [[0.01 * i, 0.02 * i, 0.0] for i in range(config.NUM_LANDMARKS)],
        dtype=np.float32,
    )
    return wrist_arr + offsets


def _valid_window(num_frames: int = config.WINDOW_LENGTH) -> list:
    """Окно из num_frames кадров, в каждом — одна найденная правая рука."""
    hand = HandResult(handedness="Right", landmarks=_base_hand_landmarks(), handedness_score=0.99)
    return [[hand] for _ in range(num_frames)]


def _empty_window(num_frames: int = config.WINDOW_LENGTH) -> list:
    """Окно, где ни на одном кадре не найдено ни одной руки."""
    return [[] for _ in range(num_frames)]


# --- load_labels -------------------------------------------------------------


def test_load_labels_reads_known_label(isolated_data_dir):
    labels = load_labels()
    assert KNOWN_LABEL in labels
    assert labels[KNOWN_LABEL]["meaning"] == "привет"


# --- record_sample + load_dataset: round-trip -------------------------------


def test_record_and_load_round_trip(isolated_data_dir):
    window = _valid_window()

    record_sample(KNOWN_LABEL, window, person="p1", session="s1")
    X, y, groups = load_dataset()

    assert X.shape == (1, config.WINDOW_LENGTH, config.FEATURE_VECTOR_SIZE)
    assert list(y) == [KNOWN_LABEL]
    assert list(groups) == ["p1__s1"]

    expected = normalize_window(
        np.stack([frame_to_vector(frame_hands) for frame_hands in window]).astype(np.float32)
    )
    assert np.allclose(X[0], expected)


def test_record_sample_appends_without_overwriting(isolated_data_dir):
    """Два сэмпла подряд для одного person+session не должны затирать друг друга."""
    record_sample(KNOWN_LABEL, _valid_window(), person="p1", session="s1")
    record_sample(KNOWN_LABEL, _valid_window(), person="p1", session="s1")

    X, y, groups = load_dataset()

    assert X.shape[0] == 2
    assert list(y) == [KNOWN_LABEL, KNOWN_LABEL]
    assert list(groups) == ["p1__s1", "p1__s1"]


# --- load_dataset: пустой и повреждённый датасет -----------------------------


def test_load_dataset_on_empty_dir_returns_empty_arrays(isolated_data_dir):
    X, y, groups = load_dataset()

    assert X.shape == (0, config.WINDOW_LENGTH, config.FEATURE_VECTOR_SIZE)
    assert y.shape == (0,)
    assert groups.shape == (0,)


def test_load_dataset_skips_corrupted_file_but_loads_the_rest(isolated_data_dir):
    # Один нормальный сэмпл.
    record_sample(KNOWN_LABEL, _valid_window(), person="p1", session="s1")

    # И один "битый" .npy, вручную дописанный в meta.csv мимо record_sample —
    # имитация обрыва записи/повреждённого файла.
    label_dir = config.RAW_DATA_DIR / KNOWN_LABEL
    corrupt_path = label_dir / "p2_s1_0.npy"
    corrupt_path.write_bytes(b"not a real npy file")

    with open(config.META_CSV_PATH, "a", encoding="utf-8", newline="") as meta_file:
        meta_file.write(
            f"raw/{KNOWN_LABEL}/p2_s1_0.npy,{KNOWN_LABEL},p2,s1,2026-01-01,,,, \n"
        )

    with pytest.warns(UserWarning):
        X, y, groups = load_dataset()

    # Битая строка пропущена, хороший сэмпл всё равно загрузился.
    assert X.shape[0] == 1
    assert list(y) == [KNOWN_LABEL]
    assert list(groups) == ["p1__s1"]


# --- record_sample: проверки контракта ---------------------------------------


def test_record_sample_wrong_window_length_raises(isolated_data_dir):
    too_short_window = _valid_window(num_frames=config.WINDOW_LENGTH - 1)

    with pytest.raises(ValueError):
        record_sample(KNOWN_LABEL, too_short_window, person="p1", session="s1")


def test_record_sample_no_hand_at_all_raises(isolated_data_dir):
    with pytest.raises(ValueError):
        record_sample(KNOWN_LABEL, _empty_window(), person="p1", session="s1")


def test_record_sample_unknown_label_raises(isolated_data_dir):
    with pytest.raises(ValueError):
        record_sample(UNKNOWN_LABEL, _valid_window(), person="p1", session="s1")


# --- groups: person+session ---------------------------------------------------


def test_groups_distinguish_sessions_but_not_duplicate_recordings(isolated_data_dir):
    # Один и тот же person+session, два сэмпла -> одна и та же группа.
    record_sample(KNOWN_LABEL, _valid_window(), person="p1", session="s1")
    record_sample(KNOWN_LABEL, _valid_window(), person="p1", session="s1")
    # Тот же person, другая session -> другая группа.
    record_sample(KNOWN_LABEL, _valid_window(), person="p1", session="s2")

    _, _, groups = load_dataset()

    assert list(groups) == ["p1__s1", "p1__s1", "p1__s2"]
    assert len(set(groups)) == 2
