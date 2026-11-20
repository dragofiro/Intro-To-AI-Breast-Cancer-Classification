"""
BreaKHis benign/malignant classifier: EfficientNetB0, multi-task head,
patient-grouped stratified K-fold CV, cached images, sensitivity-targeted evaluation.

Outputs (in OUT_DIR):
  fold{k}_best.keras      model for each fold
  oof_predictions.csv     out-of-fold image-level predictions (with TTA)
  patient_predictions.csv out-of-fold patient-level predictions
  threshold.json          operating threshold chosen for the target sensitivity
"""
import os
import re
import glob
import json
import math
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix
from evaluation_metrics import threshold_for_sensitivity

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "BreaKHis_v1/histology_slides/breast"
OUT_DIR = "outputs"
CACHE_PATH = os.path.join(OUT_DIR, "images_uint8_224x336.npy")

CROP = 224                      # network input size (square)
CACHE_H, CACHE_W = 224, 336     # stored size; keeps ~3:2 aspect ratio, random crops are taken from this
BATCH_SIZE = 32

N_FOLDS = 5
SEED = 42
GROUP_BY_BIOPSY = True          # group folders sharing a biopsy number (e.g. 14-22549AB / CD / G)

# Training schedule
HEAD_EPOCHS = 8                 # phase 1: frozen backbone
FT_EPOCHS = 20                  # phase 2: partial unfreeze (early stopping will usually end it sooner)
UNFREEZE_LAST_N = 40            # backbone layers to unfreeze in phase 2 (BatchNorm stays frozen)
HEAD_LR = 1e-3
FT_LR = 3e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.4
L2_HEAD = 1e-4
SUBTYPE_LOSS_WEIGHT = 0.3       # auxiliary head weight
EARLY_STOP_PATIENCE = 5
PATIENT_BALANCED_WEIGHTS = True  # weight each image by 1 / (images per patient)

# Evaluation
TARGET_SENSITIVITY = 0.95
PATIENT_TOPK_FRAC = 1.0         # 1.0 = mean of a patient's image probabilities; <1 = mean of top fraction
USE_MIXED_PRECISION = True      # only applied if a GPU is present

os.makedirs(OUT_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

if USE_MIXED_PRECISION and tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled.")


# ============================================================
# 1. METADATA
# ============================================================
def build_dataframe():
    rows = []
    for fp in glob.glob(os.path.join(DATA_DIR, "**/*.png"), recursive=True):
        parts = os.path.normpath(fp).split(os.sep)
        try:
            i = parts.index("SOB")
            rows.append({
                "filepath": fp,
                "tumor_class": parts[i - 1],       # 'benign' / 'malignant'
                "subtype": parts[i + 1],           # adenosis, ductal_carcinoma, ...
                "patient_id": parts[i + 2],
            })
        except (ValueError, IndexError):
            continue
    if not rows:
        raise RuntimeError(f"No PNGs found under {DATA_DIR}")

    # Sort so the row order (and therefore the cache) is deterministic
    df = pd.DataFrame(rows).sort_values("filepath").reset_index(drop=True)
    df = df[df["tumor_class"].isin(["benign", "malignant"])].reset_index(drop=True)

    df["y"] = (df["tumor_class"] == "malignant").astype(np.float32)
    df["subtype_id"] = LabelEncoder().fit_transform(df["subtype"]).astype(np.int32)

    if GROUP_BY_BIOPSY:
        def biopsy(pid):
            m = re.search(r"\d+-\d+", pid)
            return m.group() if m else pid
        df["group"] = df["patient_id"].map(biopsy)
    else:
        df["group"] = df["patient_id"]

    # Per-patient sample weight so patients with many images don't dominate
    counts = df.groupby("patient_id")["filepath"].transform("count")
    w = 1.0 / counts
    df["sample_weight"] = (w / w.mean()).astype(np.float32)
    return df


df = build_dataframe()
NUM_SUBTYPES = int(df["subtype_id"].nunique())
print(f"Images: {len(df)} | patients: {df['patient_id'].nunique()} | "
      f"groups: {df['group'].nunique()} | subtypes: {NUM_SUBTYPES}")
print(f"Malignant fraction (majority-class baseline accuracy): "
      f"{max(df['y'].mean(), 1 - df['y'].mean()):.3f}")


# ============================================================
# 2. IMAGE CACHE (decode + resize once, store as uint8 on disk and in RAM)
# ============================================================
def _load_and_resize(path):
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, [CACHE_H, CACHE_W], antialias=True)
    return tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)


def build_or_load_cache(paths):
    t0 = time.perf_counter()
    if os.path.exists(CACHE_PATH):
        arr = np.load(CACHE_PATH, mmap_mode="r")
        if arr.shape[0] == len(paths):
            print(f"Loaded image cache: {CACHE_PATH} {arr.shape} ({time.perf_counter() - t0:.2f}s)")
            return arr
        print("Cache size mismatch; rebuilding.")

    print("Building image cache (one-time cost)...")
    ds = (tf.data.Dataset.from_tensor_slices(paths)
          .map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(64)
          .prefetch(tf.data.AUTOTUNE))
    arr = np.concatenate(list(ds.as_numpy_iterator()), axis=0)
    np.save(CACHE_PATH, arr)
    print(f"Saved cache: {CACHE_PATH} {arr.shape} ({arr.nbytes / 1e9:.2f} GB) "
          f"in {time.perf_counter() - t0:.2f}s")
    return np.load(CACHE_PATH, mmap_mode="r")


IMAGES = build_or_load_cache(df["filepath"].values)


# ============================================================
# 3. tf.data PIPELINES (read from cache, augment on the fly)
# ============================================================
Y_MAL = tf.constant(df["y"].values.reshape(-1, 1), tf.float32)
Y_SUB = tf.constant(df["subtype_id"].values, tf.int32)
W_ALL = tf.constant(df["sample_weight"].values, tf.float32)
CENTER_OFFSET = (CACHE_W - CROP) // 2


def _fetch(i):
    img = tf.numpy_function(lambda j: np.ascontiguousarray(IMAGES[j]), [i], tf.uint8)
    img.set_shape([CACHE_H, CACHE_W, 3])
    return img


def _augment(img):
    """Random crop, lossless 90-degree rotations/flips, mild stain/colour jitter. Input uint8."""
    img = tf.image.random_crop(img, [CROP, CROP, 3])
    img = tf.cast(img, tf.float32)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 20.0)       # on the 0-255 scale
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.04)
    return tf.clip_by_value(img, 0.0, 255.0)


def _center(img):
    img = tf.image.crop_to_bounding_box(img, 0, CENTER_OFFSET, CROP, CROP)
    return tf.cast(img, tf.float32)


def make_ds(idx, training):
    ds = tf.data.Dataset.from_tensor_slices(idx.astype(np.int64))
    if training:
        ds = ds.shuffle(len(idx), seed=SEED, reshuffle_each_iteration=True)

    def fn(i):
        img = _fetch(i)
        img = _augment(img) if training else _center(img)
        labels = {"malignant": Y_MAL[i], "subtype": Y_SUB[i]}
        if training and PATIENT_BALANCED_WEIGHTS:
            return img, labels, W_ALL[i]
        return img, labels

    ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ============================================================
# 4. MODEL (EfficientNetB0 backbone, binary head + subtype auxiliary head)
# ============================================================
def make_optimizer(lr):
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)
    try:
        opt.exclude_from_weight_decay(var_names=["bias", "beta", "gamma"])
    except Exception:
        pass
    return opt


def compile_model(model, lr):
    model.compile(
        optimizer=make_optimizer(lr),
        loss={
            "malignant": tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
            "subtype": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"malignant": 1.0, "subtype": SUBTYPE_LOSS_WEIGHT},
        metrics={
            "malignant": [tf.keras.metrics.AUC(name="auc"),
                          tf.keras.metrics.Recall(name="recall")],
            "subtype": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        },
    )


def build_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(CROP, CROP, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(CROP, CROP, 3))      # raw [0, 255]; EfficientNet rescales internally
    x = base(inputs, training=False)                    # BatchNorm always in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    reg = tf.keras.regularizers.l2(L2_HEAD)
    mal = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg,
                                name="malignant", dtype="float32")(x)
    sub = tf.keras.layers.Dense(NUM_SUBTYPES, activation="softmax", kernel_regularizer=reg,
                                name="subtype", dtype="float32")(x)
    model = tf.keras.Model(inputs, {"malignant": mal, "subtype": sub})
    return model, base


def unfreeze_top(base, n):
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    for layer in base.layers:                           # keep BatchNorm frozen
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def early_stop():
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_malignant_auc", mode="max",
        patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1)


# ============================================================
# 5. INFERENCE WITH TEST-TIME AUGMENTATION
# ============================================================
def _make_forward_fn(model):
    @tf.function(reduce_retracing=True)
    def forward(x):
        out = model(x, training=False)
        return (
            tf.cast(out["malignant"], tf.float32),
            tf.cast(out["subtype"], tf.float32),
        )
    return forward


def predict_tta(model, idx, batch=64):
    """Average over 3 horizontal crops x (original + h-flip + v-flip) = 9 views."""
    offsets = sorted({0, CENTER_OFFSET, CACHE_W - CROP})
    forward = _make_forward_fn(model)
    p_mal, p_sub = [], []
    for s in range(0, len(idx), batch):
        sub_idx = idx[s:s + batch]
        chunk = np.asarray(IMAGES[np.sort(sub_idx)])
        order = np.argsort(np.argsort(sub_idx))   # restore requested order
        chunk = chunk[order].astype(np.float32)
        pm = np.zeros((len(chunk), 1), dtype=np.float32)
        ps = np.zeros((len(chunk), NUM_SUBTYPES), dtype=np.float32)
        n = 0
        for off in offsets:
            view = chunk[:, :CROP, off:off + CROP, :]
            for v in (view, view[:, :, ::-1], view[:, ::-1]):
                out_mal, out_sub = forward(tf.convert_to_tensor(np.ascontiguousarray(v), dtype=tf.float32))
                pm += out_mal.numpy()
                ps += out_sub.numpy()
                n += 1
        p_mal.append((pm / n).ravel())
        p_sub.append(ps / n)
    return np.concatenate(p_mal), np.concatenate(p_sub, axis=0)


# ============================================================
# 6. METRICS HELPERS
# ============================================================
def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def agg_patient(series):
    v = np.sort(series.values)[::-1]
    k = max(1, math.ceil(PATIENT_TOPK_FRAC * len(v)))
    return v[:k].mean()


def sens_spec(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec, (tn, fp, fn, tp)


# ============================================================
# 7. CROSS-VALIDATION LOOP
# ============================================================
def run_fold(fold, train_idx, val_idx):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + fold)

    train_ds, val_ds = make_ds(train_idx, True), make_ds(val_idx, False)
    steps = math.ceil(len(train_idx) / BATCH_SIZE)
    model, base = build_model()

    # Phase 1: train the heads on a frozen backbone
    t_p1 = time.perf_counter()
    compile_model(model, HEAD_LR)
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
              callbacks=[early_stop()], verbose=2)
    p1_sec = time.perf_counter() - t_p1

    # Phase 2: unfreeze the top of the backbone, cosine LR decay
    t_p2 = time.perf_counter()
    unfreeze_top(base, UNFREEZE_LAST_N)
    schedule = tf.keras.optimizers.schedules.CosineDecay(FT_LR, FT_EPOCHS * steps, alpha=0.05)
    compile_model(model, schedule)
    model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS,
              callbacks=[early_stop()], verbose=2)
    p2_sec = time.perf_counter() - t_p2

    model.save(os.path.join(OUT_DIR, f"fold{fold}_best.keras"))

    t_tta = time.perf_counter()
    p_mal, p_sub = predict_tta(model, val_idx)
    tta_sec = time.perf_counter() - t_tta

    print(f"[Fold {fold}] timing -> phase1: {p1_sec:.1f}s | phase2: {p2_sec:.1f}s | TTA: {tta_sec:.1f}s")

    out = df.iloc[val_idx][["filepath", "patient_id", "group", "subtype", "tumor_class", "y"]].copy()
    out["fold"] = fold
    out["p_malignant"] = p_mal
    out["pred_subtype"] = p_sub.argmax(axis=1)
    out["subtype_id"] = df["subtype_id"].values[val_idx]
    return out


def main():
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # Stratify on subtype so rare subtypes are spread across folds; groups keep patients/biopsies intact
    splits = list(skf.split(df, df["subtype_id"], groups=df["group"]))

    results = []
    for k, (tr, va) in enumerate(splits):
        print(f"\n{'=' * 20} FOLD {k + 1}/{N_FOLDS} | train imgs {len(tr)} | "
              f"val imgs {len(va)} | val patients {df.iloc[va]['patient_id'].nunique()} {'=' * 20}")
        assert set(df.iloc[tr]["group"]).isdisjoint(df.iloc[va]["group"]), "group leakage!"
        results.append(run_fold(k, tr, va))

    oof = pd.concat(results, ignore_index=True)
    oof.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

    # Patient-level aggregation
    pat = (oof.groupby(["fold", "patient_id", "subtype", "tumor_class"])
              .agg(y=("y", "first"), p_malignant=("p_malignant", agg_patient),
                   n_images=("y", "size"))
              .reset_index())
    pat.to_csv(os.path.join(OUT_DIR, "patient_predictions.csv"), index=False)

    # Operating threshold chosen on pooled out-of-fold PATIENT predictions
    thr = threshold_for_sensitivity(pat["y"].values, pat["p_malignant"].values, TARGET_SENSITIVITY)
    with open(os.path.join(OUT_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": thr, "target_sensitivity": TARGET_SENSITIVITY,
                   "aggregation_topk_frac": PATIENT_TOPK_FRAC}, f, indent=2)

    # ---- Per-fold report ----
    print("\n" + "=" * 30 + " RESULTS " + "=" * 30)
    rows = []
    for k in range(N_FOLDS):
        o, p = oof[oof.fold == k], pat[pat.fold == k]
        se_i, sp_i, _ = sens_spec(o["y"].values, o["p_malignant"].values, thr)
        se_p, sp_p, _ = sens_spec(p["y"].values, p["p_malignant"].values, thr)
        rows.append({
            "fold": k,
            "img_auc": safe_auc(o["y"], o["p_malignant"]),
            "img_acc@0.5": ((o["p_malignant"] >= 0.5) == (o["y"] == 1)).mean(),
            "img_sens@thr": se_i, "img_spec@thr": sp_i,
            "pat_auc": safe_auc(p["y"], p["p_malignant"]),
            "pat_sens@thr": se_p, "pat_spec@thr": sp_p,
            "subtype_acc": (o["pred_subtype"] == o["subtype_id"]).mean(),
        })
    report = pd.DataFrame(rows).set_index("fold")
    print(report.round(3).to_string())
    print("\nmean +/- std across folds")
    print(pd.DataFrame({"mean": report.mean(), "std": report.std()}).round(3).to_string())

    # ---- Pooled out-of-fold, sensitivity-targeted ----
    print(f"\nTarget patient-level sensitivity {TARGET_SENSITIVITY:.0%} -> threshold {thr:.4f} "
          f"(chosen on pooled OOF predictions; metrics at this threshold are not independent test results)")
    for name, d in (("IMAGE", oof), ("PATIENT", pat)):
        se, sp, (tn, fp, fn, tp) = sens_spec(d["y"].values, d["p_malignant"].values, thr)
        print(f"{name:8s} pooled AUC {safe_auc(d['y'], d['p_malignant']):.3f} | "
              f"sensitivity {se:.3f} | specificity {sp:.3f} | TN {tn} FP {fp} FN {fn} TP {tp}")

    # ---- Where the false negatives come from ----
    fn_pat = pat[(pat["y"] == 1) & (pat["p_malignant"] < thr)]
    if len(fn_pat):
        print("\nMissed malignant patients (false negatives):")
        print(fn_pat[["patient_id", "subtype", "p_malignant", "n_images"]]
              .sort_values("p_malignant").round(3).to_string(index=False))
        print("\nFalse negatives by subtype:")
        print(fn_pat["subtype"].value_counts().to_string())
    else:
        print("\nNo patient-level false negatives at this threshold.")


if __name__ == "__main__":
    main()
