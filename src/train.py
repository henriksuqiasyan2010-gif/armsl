"""Обучение модели распознавания жестов: baseline (RandomForest) и основная
модель (GRU), с честным train/val-разбиением по группам person+session и
логированием результатов в reports/experiments.csv.

Порядок обработки данных (см. также docstring augment_and_normalize_train):
    load_dataset(normalize=False) -> split_by_group -> augment ТОЛЬКО train
    -> normalize_window обеих частей -> обучение -> метрики -> отчёты.
TensorFlow импортируется лениво (внутри функций GRU), поэтому baseline
и весь конвейер данных работают и без него.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import numpy as np

from src import config
from src.dataset import augment, load_dataset, load_labels, split_by_group
from src.features import normalize_window


# --- Классы -----------------------------------------------------------------


def get_class_list(labels_csv_path: Path | None = None) -> list[str]:
    """Полный упорядоченный список классов из data/labels.csv.

    Именно ПОЛНЫЙ список (все строки labels.csv), а не только те метки,
    что реально встретились в записанных данных — индекс класса должен
    совпадать с индексом выхода softmax независимо от того, сколько
    сэмплов на этот класс успели записать (см. CLAUDE.md).
    """
    return list(load_labels(labels_csv_path))


def _labels_to_indices(labels: np.ndarray, class_list: list[str]) -> np.ndarray:
    """Строковые метки -> индексы по позиции в class_list.

    ValueError с понятным текстом, если встретилась метка, которой нет
    в class_list -- это означает, что labels.csv и данные разошлись
    (см. CLAUDE.md про единственный источник истины по классам).
    """
    index_by_label = {label: i for i, label in enumerate(class_list)}
    indices = []
    for label in labels:
        if label not in index_by_label:
            raise ValueError(
                f"Метка {label!r} встретилась в данных, но не найдена в "
                f"data/labels.csv (классы: {class_list}). "
                "Список меток и данные разошлись."
            )
        indices.append(index_by_label[label])
    return np.array(indices, dtype=np.int64)


# --- Данные: загрузка, разрез, аугментация ----------------------------------


class DataSplit(NamedTuple):
    """Результат prepare_train_val -- ВСЁ ЕЩЁ сырое (до normalize_window)."""

    X_train_raw: np.ndarray
    y_train: np.ndarray
    groups_train: np.ndarray
    X_val_raw: np.ndarray
    y_val: np.ndarray
    groups_val: np.ndarray


def prepare_train_val(val_groups: Iterable[str], *, data_dir: Path | None = None) -> DataSplit:
    """Загрузить датасет и разбить на train/val по группам person__session.

    Тонкая обёртка вокруг load_dataset(normalize=False) + split_by_group,
    но дополнительно возвращает groups_train/groups_val -- сам split_by_group
    их не отдаёт (группы уже сделали свою работу при разбиении), а здесь
    они нужны, чтобы ВНУТРИ train.py можно было проверить непересечение
    групп напрямую, не только в тестах на сам split_by_group.
    """
    X_raw, y, groups = load_dataset(data_dir, normalize=False)
    X_train_raw, y_train, X_val_raw, y_val = split_by_group(X_raw, y, groups, val_groups)

    is_val = np.isin(groups, list(set(val_groups)))
    groups_train, groups_val = groups[~is_val], groups[is_val]

    return DataSplit(X_train_raw, y_train, groups_train, X_val_raw, y_val, groups_val)


def augment_and_normalize_train(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    add_noise: bool,
    add_shift: bool,
    add_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Единственное место во всём train.py, где вызывается augment().

    У этой функции в параметрах нет X_val/y_val -- валидационных данных
    здесь физически не существует, их нечем аугментировать по ошибке.
    Второй барьер -- normalize_val() ниже: она понятия не имеет об
    augment и не может его вызвать, даже случайно.

    Если хотя бы один из add_noise/add_shift/add_scale включён: каждый
    исходный сэмпл ДОПОЛНЯЕТСЯ одним аугментированным (train увеличивается
    вдвое), а не заменяется им -- модель видит и настоящие, чистые записи,
    и их шумные варианты. Если все три флага выключены -- аугментации нет,
    train остаётся исходного размера.

    seed -- один мастер-сид на весь train.py (NFR-4). Каждому сэмплу нужен
    СВОЙ случайный сдвиг/шум (иначе все аугментированные сэмплы получат
    одно и то же смещение), поэтому из mastseed через generator извлекается
    отдельный подсид на каждый сэмпл -- одинаковый seed всегда даёт
    одинаковую последовательность подсидов, а значит одинаковый результат.
    """
    normalized = (
        np.stack([normalize_window(w) for w in X_train_raw])
        if len(X_train_raw)
        else X_train_raw.copy()
    )

    if not (add_noise or add_shift or add_scale):
        return normalized, y_train

    rng = np.random.default_rng(seed)
    augmented_raw = np.stack(
        [
            augment(
                w,
                seed=int(rng.integers(0, 2**31 - 1)),
                add_noise=add_noise,
                add_shift=add_shift,
                add_scale=add_scale,
            )
            for w in X_train_raw
        ]
    )
    augmented_normalized = np.stack([normalize_window(w) for w in augmented_raw])

    X_combined = np.concatenate([normalized, augmented_normalized], axis=0)
    y_combined = np.concatenate([y_train, y_train], axis=0)
    return X_combined, y_combined


def normalize_val(X_val_raw: np.ndarray) -> np.ndarray:
    """Нормализовать val -- и ТОЛЬКО нормализовать.

    В сигнатуре нет флагов аугментации и seed: этой функции физически
    нечем аугментировать val, даже по ошибке. Единственный путь, которым
    val превращается в вектор признаков.
    """
    if len(X_val_raw) == 0:
        return X_val_raw.copy()
    return np.stack([normalize_window(w) for w in X_val_raw])


# --- Baseline: RandomForest --------------------------------------------------


def train_baseline(X_train: np.ndarray, y_train_idx: np.ndarray, *, seed: int):
    """RandomForestClassifier на расплющенном окне (WINDOW_LENGTH*FEATURE_VECTOR_SIZE)."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=config.RF_N_ESTIMATORS, random_state=seed)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train_idx)
    return model


def predict_baseline(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X.reshape(X.shape[0], -1))


# --- GRU (TensorFlow импортируется лениво, только здесь) --------------------


def build_gru_model(num_classes: int):
    """Masking -> GRU -> Dropout -> GRU -> Dense -> Dense(softmax).

    Masking(mask_value=0.0) ложится на контракт данных без дополнительного
    кода: кадр, где ни одна рука не найдена, -- это ровно 128 нулей
    (координаты нулевые по контракту, оба флага видимости тоже 0), и такие
    кадры GRU автоматически пропускает, не считая их частью жеста.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Masking(
                mask_value=0.0,
                input_shape=(config.WINDOW_LENGTH, config.FEATURE_VECTOR_SIZE),
            ),
            layers.GRU(config.GRU_UNITS_1, return_sequences=True),
            layers.Dropout(config.GRU_DROPOUT),
            layers.GRU(config.GRU_UNITS_2),
            layers.Dense(config.DENSE_UNITS, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(config.GRU_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_gru(
    X_train: np.ndarray,
    y_train_idx: np.ndarray,
    X_val: np.ndarray,
    y_val_idx: np.ndarray,
    *,
    num_classes: int,
    seed: int,
    epochs: int | None = None,
):
    """Обучить GRU с EarlyStopping по val_loss и фиксированным сидом (NFR-4)."""
    from tensorflow import keras

    keras.utils.set_random_seed(seed)

    model = build_gru_model(num_classes)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.GRU_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train_idx,
        validation_data=(X_val, y_val_idx),
        epochs=config.GRU_EPOCHS if epochs is None else epochs,
        batch_size=config.GRU_BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0,
    )
    return model, history


def predict_gru(model, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict(X, verbose=0)
    return np.argmax(probabilities, axis=1)


# --- Метрики и графики -------------------------------------------------------


def _compute_metrics(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, class_list: list[str]) -> dict:
    """accuracy + macro/per-class F1, всегда по ПОЛНОМУ списку классов.

    labels=range(len(class_list)) и zero_division=0 -- иначе классы,
    которых не было в val (или в train), просто выпали бы из отчёта, и
    baseline с GRU оказались бы несравнимы построчно в experiments.csv,
    если у них разный набор реально предсказанных классов.
    """
    from sklearn.metrics import f1_score

    labels_range = list(range(len(class_list)))
    accuracy = float(np.mean(y_true_idx == y_pred_idx)) if len(y_true_idx) else 0.0
    macro_f1 = float(
        f1_score(y_true_idx, y_pred_idx, labels=labels_range, average="macro", zero_division=0)
    )
    per_class_f1 = f1_score(y_true_idx, y_pred_idx, labels=labels_range, average=None, zero_division=0)
    per_class_f1_map = {class_list[i]: float(per_class_f1[i]) for i in labels_range}

    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class_f1": per_class_f1_map}


def _plot_confusion_matrix(
    y_true_idx: np.ndarray, y_pred_idx: np.ndarray, class_list: list[str], out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)  # без этого попытка открыть окно уронит headless/тесты
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    labels_range = list(range(len(class_list)))
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=labels_range)

    size = max(6, len(class_list) * 0.5)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels_range)
    ax.set_xticklabels(class_list, rotation=90)
    ax.set_yticks(labels_range)
    ax.set_yticklabels(class_list)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинное")
    fig.colorbar(image)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_training_curves(history, out_path: Path) -> None:
    """Loss и accuracy по эпохам. Только для GRU -- у RandomForest нет эпох."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # В разных версиях keras метрика могла называться "accuracy" или "acc".
    train_acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Эпоха")
    axes[0].legend()

    axes[1].plot(history.history[train_acc_key], label="train")
    axes[1].plot(history.history[val_acc_key], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Эпоха")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# --- Отчёт experiments.csv ---------------------------------------------------


def _make_run_id() -> str:
    """Временная метка + короткий случайный суффикс -- уникален даже если
    два запуска стартовали в одну и ту же секунду (как в тестах)."""
    return f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _append_experiment_row(experiments_csv_path: Path, row: dict) -> None:
    """Дописать одну строку в reports/experiments.csv (тот же приём, что
    _append_meta_row в dataset.py: создать с заголовком, если файла нет,
    иначе только дописать -- предыдущие запуски не перезатираются)."""
    experiments_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_is_new = not experiments_csv_path.exists() or experiments_csv_path.stat().st_size == 0

    with open(experiments_csv_path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=config.EXPERIMENTS_CSV_COLUMNS)
        if file_is_new:
            writer.writeheader()
        writer.writerow(row)


# --- Сохранение модели + снимка классов --------------------------------------


def _classes_fingerprint(class_list: list[str]) -> str:
    payload = json.dumps(class_list, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_model_with_labels_snapshot(
    save_fn: Callable[[Path], None],
    class_list: list[str],
    model_path: Path,
    snapshot_path: Path,
) -> None:
    """Сохранить модель и снимок списка классов ВМЕСТЕ -- либо оба, либо ни одного.

    save_fn -- вызывается с model_path и должен сохранить туда модель
               (например, keras_model.save или functools.partial(joblib.dump, model)).
               Эта функция не знает, как сериализуется конкретный тип модели --
               это решает вызывающий код.

    .classes.json -- СНИМОК для проверки (список классов + sha256-отпечаток +
    время сохранения), а НЕ второй редактируемый список: единственный
    источник истины по классам -- data/labels.csv (см. CLAUDE.md).
    predictor.py должен сверять этот снимок со свежим load_labels() и
    отказываться работать при несовпадении.

    ГАРАНТИЯ СОГЛАСОВАННОСТИ -- честно, не "атомарно": на Windows нет
    надёжного способа атомарно заменить два разных файла одной операцией.
    Вместо этого: сохранить оба -> проверить, что оба на месте и снимок
    совпадает с переданным class_list -> при любой проблеме удалить ОБА
    файла и поднять исключение. После сбоя не остаётся половины модели
    без соответствующего снимка классов (или наоборот).
    """
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        save_fn(model_path)

        snapshot = {
            "classes": class_list,
            "classes_sha256": _classes_fingerprint(class_list),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        if not model_path.exists():
            raise RuntimeError(f"Модель не сохранилась: {model_path}")
        saved_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if saved_snapshot["classes"] != class_list:
            raise RuntimeError(
                "Сохранённый снимок классов не совпадает с переданным списком -- "
                "модель и labels.json/labels.csv разошлись бы."
            )
    except Exception:
        model_path.unlink(missing_ok=True)
        snapshot_path.unlink(missing_ok=True)
        raise


# --- Оркестрация одного запуска ----------------------------------------------


def run_experiment(
    *,
    model_type: str,
    val_groups: Iterable[str],
    seed: int = config.RANDOM_SEED,
    augment_noise: bool = True,
    augment_shift: bool = True,
    augment_scale: bool = True,
    epochs: int | None = None,
    data_dir: Path | None = None,
    reports_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict:
    """Полный конвейер одного запуска: данные -> обучение -> метрики -> отчёты.

    model_type -- "baseline" (RandomForest) или "gru".
    val_groups -- группы person__session, уходящие в val (см. split_by_group).
    epochs     -- только для GRU; None -> config.GRU_EPOCHS (используется
                  тестами, чтобы не гонять полное обучение).
    data_dir, reports_dir, models_dir -- переопределения путей для тестов
                  (по умолчанию -- config.DATA_DIR/REPORTS_DIR/MODELS_DIR,
                  резолвятся внутри функции, а не как значение параметра
                  по умолчанию, иначе monkeypatch config.* в тестах не сработал бы).

    Возвращает dict с run_id, метриками и путями к сохранённым артефактам.
    """
    if model_type not in ("baseline", "gru"):
        raise ValueError(f'model_type должен быть "baseline" или "gru", получено {model_type!r}')

    if reports_dir is None:
        reports_dir = config.REPORTS_DIR
    if models_dir is None:
        models_dir = config.MODELS_DIR

    random.seed(seed)
    np.random.seed(seed)

    class_list = get_class_list()

    split = prepare_train_val(val_groups, data_dir=data_dir)

    # Тот же принцип, что и в split_by_group, но проверен здесь заново,
    # на уровне train.py, а не только доверием к чужой функции.
    if set(split.groups_train) & set(split.groups_val):
        raise RuntimeError(
            "Группы train и val пересекаются — это баг в разбиении, "
            "обучать модель в таком виде нельзя."
        )

    missing_in_train = [c for c in class_list if c not in set(split.y_train)]
    if missing_in_train:
        warnings.warn(
            f"В train нет ни одного сэмпла для классов: {missing_in_train}. "
            "Это ожидаемо на раннем этапе съёмки, но модель никогда не увидит "
            "эти жесты во время обучения."
        )

    y_train_idx = _labels_to_indices(split.y_train, class_list)
    y_val_idx = _labels_to_indices(split.y_val, class_list)

    # Аугментация -- ТОЛЬКО train (см. augment_and_normalize_train), val --
    # только нормализация (см. normalize_val). Это два разных пути, и
    # augment вызывается ровно в одном месте всего модуля.
    X_train, y_train_idx = augment_and_normalize_train(
        split.X_train_raw,
        y_train_idx,
        seed=seed,
        add_noise=augment_noise,
        add_shift=augment_shift,
        add_scale=augment_scale,
    )
    X_val = normalize_val(split.X_val_raw)

    history = None
    if model_type == "baseline":
        model = train_baseline(X_train, y_train_idx, seed=seed)
        y_pred_idx = predict_baseline(model, X_val)
    else:
        model, history = train_gru(
            X_train, y_train_idx, X_val, y_val_idx,
            num_classes=len(class_list), seed=seed, epochs=epochs,
        )
        y_pred_idx = predict_gru(model, X_val)

    metrics = _compute_metrics(y_val_idx, y_pred_idx, class_list)

    run_id = _make_run_id()

    confusion_matrix_path = reports_dir / f"{run_id}_confusion_matrix.png"
    _plot_confusion_matrix(y_val_idx, y_pred_idx, class_list, confusion_matrix_path)

    training_curves_path = None
    if history is not None:
        training_curves_path = reports_dir / f"{run_id}_training_curves.png"
        _plot_training_curves(history, training_curves_path)

    _append_experiment_row(
        reports_dir / config.EXPERIMENTS_CSV_PATH.name,
        {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model_type,
            "augment_noise": augment_noise,
            "augment_shift": augment_shift,
            "augment_scale": augment_scale,
            "n_train": X_train.shape[0],
            "n_val": X_val.shape[0],
            "n_classes": len(class_list),
            "val_groups": ";".join(sorted(set(val_groups))),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": json.dumps(metrics["per_class_f1"], ensure_ascii=False),
        },
    )

    model_path = None
    snapshot_path = None
    if model_type == "gru":
        model_path = models_dir / config.GRU_MODEL_PATH.name
        snapshot_path = models_dir / config.GRU_CLASSES_SNAPSHOT_PATH.name
        save_model_with_labels_snapshot(
            save_fn=model.save,
            class_list=class_list,
            model_path=model_path,
            snapshot_path=snapshot_path,
        )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "model": model,
        "confusion_matrix_path": confusion_matrix_path,
        "training_curves_path": training_curves_path,
        "model_path": model_path,
        "snapshot_path": snapshot_path,
    }


# --- CLI ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение модели распознавания жестов")
    parser.add_argument("--model", choices=["baseline", "gru"], required=True)
    parser.add_argument(
        "--val-groups",
        required=True,
        nargs="+",
        metavar="PERSON__SESSION",
        help="группы person__session, уходящие в валидацию (см. src/dataset.py: load_dataset)",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=None, help="только для --model gru")
    parser.add_argument("--no-augment-noise", dest="augment_noise", action="store_false", default=True)
    parser.add_argument("--no-augment-shift", dest="augment_shift", action="store_false", default=True)
    parser.add_argument("--no-augment-scale", dest="augment_scale", action="store_false", default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_experiment(
        model_type=args.model,
        val_groups=args.val_groups,
        seed=args.seed,
        augment_noise=args.augment_noise,
        augment_shift=args.augment_shift,
        augment_scale=args.augment_scale,
        epochs=args.epochs,
    )
    print(f"run_id={result['run_id']}")
    print(f"accuracy={result['metrics']['accuracy']:.4f}  macro_f1={result['metrics']['macro_f1']:.4f}")
    print(f"confusion matrix: {result['confusion_matrix_path']}")
    if result["training_curves_path"] is not None:
        print(f"training curves: {result['training_curves_path']}")
    if result["model_path"] is not None:
        print(f"модель: {result['model_path']}")
        print(f"снимок классов: {result['snapshot_path']}")


if __name__ == "__main__":
    main()
