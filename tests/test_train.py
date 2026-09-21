"""Тесты обучения (src/train.py) — всё на синтетическом датасете в tmp_path.

Реального датасета для обучения пока нет — это нормально (см. CLAUDE.md):
модуль проверяется на маленьком фейковом наборе, который тесты строят сами
через record_sample, как и в test_dataset.py.
"""

import csv
import itertools
import json

import numpy as np
import pytest

from src import config, train
from src.dataset import record_sample
from src.features import frame_to_vector
from src.landmarks import HandResult


@pytest.fixture
def isolated_project_dir(tmp_path, monkeypatch):
    """Подменяет пути config.* (data/labels) на временную папку и отдаёт
    отдельные reports_dir/models_dir — run_experiment принимает их явно,
    поэтому эти два подменять через config не нужно."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)

    labels_csv = data_dir / "labels.csv"
    labels_csv.write_text(
        "label_id,armenian,pronunciation,meaning\n"
        "barev,Բարև,barev,привет\n"
        "jur,Ջուր,jur,вода\n"
        "lav,Լավ,lav,хорошо\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RAW_DATA_DIR", data_dir / "raw")
    monkeypatch.setattr(config, "META_CSV_PATH", data_dir / "meta.csv")
    monkeypatch.setattr(config, "LABELS_CSV_PATH", labels_csv)

    return {
        "data_dir": data_dir,
        "reports_dir": tmp_path / "reports",
        "models_dir": tmp_path / "models",
    }


def _base_hand_landmarks(wrist=(0.5, 0.5, 0.0)) -> np.ndarray:
    """Та же синтетическая рука, что в test_features.py/test_dataset.py."""
    wrist_arr = np.array(wrist, dtype=np.float32)
    offsets = np.array(
        [[0.01 * i, 0.02 * i, 0.0] for i in range(config.NUM_LANDMARKS)],
        dtype=np.float32,
    )
    return wrist_arr + offsets


def _window(num_frames: int = config.WINDOW_LENGTH, wrist_x: float = 0.5) -> list:
    """wrist_x различает сэмплы друг от друга — иначе все окна были бы
    побитово идентичны, и проверка "это окно из train, а не из val" не
    имела бы смысла (любое окно совпало бы с любым другим)."""
    hand = HandResult(
        handedness="Right", landmarks=_base_hand_landmarks(wrist=(wrist_x, 0.5, 0.0)), handedness_score=0.99
    )
    return [[hand] for _ in range(num_frames)]


def _raw_window(wrist_x: float = 0.5) -> np.ndarray:
    """Сырое окно (T,128) — то же самое, что record_sample считает внутри себя,
    но без записи на диск (для тестов чистых функций augment_and_normalize_train)."""
    return np.stack(
        [frame_to_vector(frame_hands) for frame_hands in _window(wrist_x=wrist_x)]
    ).astype(np.float32)


_sample_counter = itertools.count()


def _record_samples(label: str, person: str, session: str, count: int) -> None:
    for _ in range(count):
        # Каждый сэмпл — с чуть другой позицией руки, чтобы никакие два
        # окна во всём тестовом датасете не совпадали побитово.
        wrist_x = 0.3 + 0.001 * next(_sample_counter)
        record_sample(label, _window(wrist_x=wrist_x), person=person, session=session)


# --- get_class_list / _labels_to_indices -------------------------------------


def test_get_class_list_matches_labels_csv_order(isolated_project_dir):
    assert train.get_class_list() == ["barev", "jur", "lav"]


def test_labels_to_indices_raises_on_unknown_label():
    with pytest.raises(ValueError):
        train._labels_to_indices(np.array(["not_a_class"]), ["barev", "jur"])


# --- prepare_train_val: непересечение групп ВНУТРИ train.py -----------------


def test_prepare_train_val_groups_do_not_overlap(isolated_project_dir):
    _record_samples("barev", "p1", "s1", 2)
    _record_samples("jur", "p1", "s2", 2)
    _record_samples("barev", "p2", "s1", 2)
    _record_samples("jur", "p2", "s1", 2)

    split = train.prepare_train_val({"p2__s1"})

    assert set(split.groups_train) & set(split.groups_val) == set()
    assert set(split.groups_val) == {"p2__s1"}
    assert set(split.groups_train) == {"p1__s1", "p1__s2"}
    assert split.X_train_raw.shape[0] + split.X_val_raw.shape[0] == 8


# --- augment_and_normalize_train: чистые функции, без диска -----------------


def test_augment_and_normalize_train_seed_reproducible():
    X_train_raw = np.stack([_raw_window(), _raw_window()])
    y_train_idx = np.array([0, 1], dtype=np.int64)

    X1, y1 = train.augment_and_normalize_train(
        X_train_raw, y_train_idx, seed=7, add_noise=True, add_shift=True, add_scale=True
    )
    X2, y2 = train.augment_and_normalize_train(
        X_train_raw, y_train_idx, seed=7, add_noise=True, add_shift=True, add_scale=True
    )

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_augment_and_normalize_train_disabled_keeps_original_size():
    X_train_raw = np.stack([_raw_window(), _raw_window()])
    y_train_idx = np.array([0, 1], dtype=np.int64)

    X, y = train.augment_and_normalize_train(
        X_train_raw, y_train_idx, seed=0, add_noise=False, add_shift=False, add_scale=False
    )

    assert X.shape[0] == 2
    assert list(y) == [0, 1]


def test_augment_and_normalize_train_enabled_doubles_size():
    X_train_raw = np.stack([_raw_window(), _raw_window()])
    y_train_idx = np.array([0, 1], dtype=np.int64)

    X, y = train.augment_and_normalize_train(
        X_train_raw, y_train_idx, seed=0, add_noise=True, add_shift=False, add_scale=False
    )

    assert X.shape[0] == 4
    assert list(y) == [0, 1, 0, 1]


# --- augment применяется ТОЛЬКО к train (структурная проверка) --------------


def test_augment_never_called_with_val_data(isolated_project_dir, monkeypatch):
    """Подмена augment на шпион: проверяем, что каждый вызов получил окно
    из X_train_raw, и ни один — из X_val_raw."""
    _record_samples("barev", "p1", "s1", 2)
    _record_samples("jur", "p1", "s2", 2)
    _record_samples("barev", "p2", "s1", 2)

    calls: list[np.ndarray] = []

    def fake_augment(window, seed=None, *, add_noise=True, add_shift=True, add_scale=True):
        calls.append(np.array(window, copy=True))
        return window

    monkeypatch.setattr(train, "augment", fake_augment)

    split = train.prepare_train_val({"p2__s1"})
    y_train_idx = np.zeros(len(split.y_train), dtype=np.int64)

    train.augment_and_normalize_train(
        split.X_train_raw, y_train_idx, seed=0, add_noise=True, add_shift=True, add_scale=True
    )
    train.normalize_val(split.X_val_raw)  # не должно трогать augment вообще

    assert len(calls) == split.X_train_raw.shape[0]
    for called_window in calls:
        assert not any(np.array_equal(called_window, val_window) for val_window in split.X_val_raw)


# --- save_model_with_labels_snapshot: всё или ничего -------------------------


def test_save_model_with_labels_snapshot_success(tmp_path):
    model_path = tmp_path / "model.bin"
    snapshot_path = tmp_path / "model.classes.json"

    def ok_save(path):
        path.write_text("weights", encoding="utf-8")

    train.save_model_with_labels_snapshot(ok_save, ["a", "b"], model_path, snapshot_path)

    assert model_path.read_text(encoding="utf-8") == "weights"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["classes"] == ["a", "b"]


def test_save_model_with_labels_snapshot_rolls_back_on_failure(tmp_path):
    model_path = tmp_path / "model.bin"
    snapshot_path = tmp_path / "model.classes.json"

    def failing_save(path):
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("диск кончился на середине")

    with pytest.raises(RuntimeError):
        train.save_model_with_labels_snapshot(failing_save, ["a", "b"], model_path, snapshot_path)

    assert not model_path.exists()
    assert not snapshot_path.exists()


# --- run_experiment: baseline сквозной прогон --------------------------------


def test_run_experiment_baseline_end_to_end(isolated_project_dir):
    dirs = isolated_project_dir
    _record_samples("barev", "p1", "s1", 3)
    _record_samples("jur", "p1", "s2", 3)
    _record_samples("barev", "p2", "s1", 2)
    _record_samples("jur", "p2", "s1", 2)

    result = train.run_experiment(
        model_type="baseline",
        val_groups={"p2__s1"},
        reports_dir=dirs["reports_dir"],
        models_dir=dirs["models_dir"],
    )

    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["confusion_matrix_path"].exists()
    assert result["training_curves_path"] is None  # у RandomForest нет эпох
    assert result["model_path"] is None  # baseline не сохраняется как артефакт


def test_experiments_csv_appends_not_overwrites(isolated_project_dir):
    dirs = isolated_project_dir
    _record_samples("barev", "p1", "s1", 3)
    _record_samples("jur", "p1", "s2", 3)
    _record_samples("barev", "p2", "s1", 2)
    _record_samples("jur", "p2", "s1", 2)

    result1 = train.run_experiment(
        model_type="baseline",
        val_groups={"p2__s1"},
        reports_dir=dirs["reports_dir"],
        models_dir=dirs["models_dir"],
    )
    result2 = train.run_experiment(
        model_type="baseline",
        val_groups={"p2__s1"},
        reports_dir=dirs["reports_dir"],
        models_dir=dirs["models_dir"],
    )

    csv_path = dirs["reports_dir"] / config.EXPERIMENTS_CSV_PATH.name
    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["run_id"] == result1["run_id"]
    assert rows[1]["run_id"] == result2["run_id"]


def test_run_experiment_rejects_unknown_model_type(isolated_project_dir):
    with pytest.raises(ValueError):
        train.run_experiment(model_type="svm", val_groups={"p1__s1"})


# --- run_experiment: GRU (пропускается, если tensorflow не установлен) ------


def test_run_experiment_gru_saves_model_and_consistent_snapshot(isolated_project_dir):
    pytest.importorskip("tensorflow")
    dirs = isolated_project_dir
    _record_samples("barev", "p1", "s1", 3)
    _record_samples("jur", "p1", "s2", 3)
    _record_samples("barev", "p2", "s1", 2)
    _record_samples("jur", "p2", "s1", 2)

    result = train.run_experiment(
        model_type="gru",
        val_groups={"p2__s1"},
        epochs=1,
        reports_dir=dirs["reports_dir"],
        models_dir=dirs["models_dir"],
    )

    assert result["model_path"].exists()
    assert result["snapshot_path"].exists()
    assert result["training_curves_path"].exists()

    snapshot = json.loads(result["snapshot_path"].read_text(encoding="utf-8"))
    assert snapshot["classes"] == train.get_class_list()
