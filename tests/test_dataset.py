"""Тесты загрузки и разбиения датасета (src/dataset.py)."""

import numpy as np
import pytest

from src import config
from src.dataset import augment, load_dataset, load_labels, record_sample, split_by_group
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


def _raw_window_right_hand_only(num_frames: int = config.WINDOW_LENGTH) -> np.ndarray:
    """Сырое окно (T,128): правая рука есть на всех кадрах, левой нет ни на одном
    (нули + флаг 0) — для проверки, что augment не трогает отсутствующую руку."""
    return np.stack([frame_to_vector(frame_hands) for frame_hands in _valid_window(num_frames)]).astype(
        np.float32
    )


# --- load_labels -------------------------------------------------------------


def test_load_labels_reads_known_label(isolated_data_dir):
    labels = load_labels()
    assert KNOWN_LABEL in labels
    assert labels[KNOWN_LABEL]["meaning"] == "привет"


def test_load_labels_preserves_row_order(isolated_data_dir):
    """Порядок строк labels.csv = порядок классов = индекс выхода softmax."""
    labels = load_labels()
    assert list(labels) == [KNOWN_LABEL, "jur"]


def test_load_labels_rejects_duplicate_label_id(isolated_data_dir):
    """Дубликат label_id сдвинул бы индексы классов — должен быть ValueError."""
    config.LABELS_CSV_PATH.write_text(
        "label_id,armenian,pronunciation,meaning\n"
        f"{KNOWN_LABEL},Բարև,barev,привет\n"
        "jur,Ջուր,jur,вода\n"
        f"{KNOWN_LABEL},Բարև,barev,привет ещё раз\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_labels()


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


def test_npy_on_disk_is_raw_normalization_happens_on_read(isolated_data_dir):
    """На диск сохраняется СЫРОЙ вектор (frame_to_vector), а normalize_window
    применяется только при чтении в load_dataset.

    Если бы нормализация случайно уехала обратно в record_sample, файл на
    диске совпал бы с нормализованным — этот тест это поймает.
    """
    window = _valid_window()
    record_sample(KNOWN_LABEL, window, person="p1", session="s1")

    raw_expected = np.stack(
        [frame_to_vector(frame_hands) for frame_hands in window]
    ).astype(np.float32)
    normalized_expected = normalize_window(raw_expected)

    on_disk = np.load(config.RAW_DATA_DIR / KNOWN_LABEL / "p1_s1_0.npy")

    # На диске — сырое, и это не то же самое, что нормализованное.
    assert np.allclose(on_disk, raw_expected)
    assert not np.allclose(on_disk, normalized_expected)

    # А load_dataset отдаёт уже нормализованное.
    X, _, _ = load_dataset()
    assert np.allclose(X[0], normalized_expected)
    assert not np.allclose(X[0], on_disk)


def test_load_dataset_normalize_false_returns_raw(isolated_data_dir):
    """load_dataset(normalize=False) отдаёт то же, что лежит на диске,
    без вызова normalize_window. Нужно train.py — augment() применяется
    к сырому окну, до нормализации."""
    window = _valid_window()
    record_sample(KNOWN_LABEL, window, person="p1", session="s1")

    raw_expected = np.stack(
        [frame_to_vector(frame_hands) for frame_hands in window]
    ).astype(np.float32)

    X_raw, y, groups = load_dataset(normalize=False)

    assert np.allclose(X_raw[0], raw_expected)
    assert list(y) == [KNOWN_LABEL]
    assert list(groups) == ["p1__s1"]


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


# --- split_by_group -----------------------------------------------------------


def _synthetic_dataset():
    """Небольшой синтетический (X, y, groups) без обращения к диску:
    3 группы, по одному сэмплу на группу, X кодирует номер сэмпла,
    чтобы легко проверить, какие строки куда попали."""
    X = np.arange(3, dtype=np.float32).reshape(3, 1, 1) * np.ones(
        (3, config.WINDOW_LENGTH, config.FEATURE_VECTOR_SIZE), dtype=np.float32
    )
    y = np.array(["barev", "jur", "barev"], dtype=str)
    groups = np.array(["p1__s1", "p1__s2", "p2__s1"], dtype=str)
    return X, y, groups


def test_split_by_group_splits_exactly_by_group_membership():
    X, y, groups = _synthetic_dataset()
    val_groups = {"p1__s2"}

    X_train, y_train, X_val, y_val = split_by_group(X, y, groups, val_groups)

    is_val = np.isin(groups, list(val_groups))
    assert np.array_equal(X_train, X[~is_val])
    assert np.array_equal(y_train, y[~is_val])
    assert np.array_equal(X_val, X[is_val])
    assert np.array_equal(y_val, y[is_val])
    # Ничего не потеряно и не задвоено.
    assert X_train.shape[0] + X_val.shape[0] == X.shape[0]


def test_split_by_group_empty_dataset_raises():
    X = np.zeros((0, config.WINDOW_LENGTH, config.FEATURE_VECTOR_SIZE), dtype=np.float32)
    y = np.array([], dtype=str)
    groups = np.array([], dtype=str)

    with pytest.raises(ValueError, match="пуст"):
        split_by_group(X, y, groups, val_groups={"p1__s1"})


def test_split_by_group_train_empty_when_val_groups_covers_everything():
    X, y, groups = _synthetic_dataset()

    with pytest.raises(ValueError, match="train пуст"):
        split_by_group(X, y, groups, val_groups=set(groups.tolist()))


def test_split_by_group_val_empty_when_val_groups_unknown():
    X, y, groups = _synthetic_dataset()

    with pytest.raises(ValueError, match="val пуст"):
        split_by_group(X, y, groups, val_groups={"typo_person__typo_session"})


# --- augment --------------------------------------------------------------------


def test_augment_preserves_shape():
    window = _raw_window_right_hand_only()
    augmented = augment(window, seed=0)
    assert augmented.shape == window.shape


def test_augment_does_not_touch_visibility_flags():
    window = _raw_window_right_hand_only()
    augmented = augment(window, seed=0)
    assert np.array_equal(augmented[:, -2:], window[:, -2:])


def test_augment_does_not_touch_absent_hand():
    """Левой руки в этом окне нет ни на одном кадре (нули + флаг 0) —
    после аугментации она должна остаться точным нулём, а не "зашумиться"."""
    window = _raw_window_right_hand_only()
    augmented = augment(window, seed=0)

    left_slice = slice(0, config.HAND_VECTOR_SIZE)
    assert np.array_equal(augmented[:, left_slice], window[:, left_slice])
    assert np.all(augmented[:, left_slice] == 0.0)

    # А присутствующая (правая) рука реально поменялась — иначе тест
    # ничего бы не проверял.
    right_slice = slice(config.HAND_VECTOR_SIZE, 2 * config.HAND_VECTOR_SIZE)
    assert not np.array_equal(augmented[:, right_slice], window[:, right_slice])


def test_augment_seed_none_gives_different_results():
    window = _raw_window_right_hand_only()
    result_a = augment(window, seed=None)
    result_b = augment(window, seed=None)
    assert not np.array_equal(result_a, result_b)


def test_augment_same_seed_gives_same_result():
    window = _raw_window_right_hand_only()
    result_a = augment(window, seed=123)
    result_b = augment(window, seed=123)
    assert np.array_equal(result_a, result_b)
