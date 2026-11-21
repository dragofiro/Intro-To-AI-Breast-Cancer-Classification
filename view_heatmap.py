"""
Preview what the spatial-attention heatmap looks like. Read-only: it does not
change or retrain anything.

Usage (from the project folder):
    python view_heatmap.py
    python view_heatmap.py --model other.keras --n 4 --seed 1

Needs: pip install matplotlib
"""
import argparse
import glob
import os
import random

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="breakhis_heatmap_classifier.keras")
parser.add_argument("--data", default="BreaKHis_v1/histology_slides/breast")
parser.add_argument("--n", type=int, default=3, help="images per tumor class (benign / malignant)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--scale", type=float, default=1 / 255,
                    help="input scaling the model was trained with (heat_map.py: 1/255; EfficientNet: 1.0)")
parser.add_argument("--out", default="heatmap_preview.png")
args = parser.parse_args()

SIZE = 224

# Collect images; subtype names sorted alphabetically = LabelEncoder order used in heat_map.py
items = []
for fp in glob.glob(os.path.join(args.data, "**/*.png"), recursive=True):
    parts = os.path.normpath(fp).split(os.sep)
    if "SOB" in parts:
        i = parts.index("SOB")
        items.append((fp, parts[i - 1], parts[i + 1]))
if not items:
    raise SystemExit(f"No PNGs found under {args.data}")
subtypes = sorted({s for _, _, s in items})

rng = random.Random(args.seed)
picked = []
for cls in ("benign", "malignant"):
    pool = [t for t in items if t[1] == cls]
    picked += rng.sample(pool, min(args.n, len(pool)))

model = tf.keras.models.load_model(args.model)
hm_model = tf.keras.Model(model.input, [model.get_layer("spatial_heatmap").output, model.output])


def load(path):
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    return tf.image.resize(img, [SIZE, SIZE]).numpy()          # float32, 0-255


fig, axes = plt.subplots(len(picked), 3, figsize=(9, 3 * len(picked)))
axes = np.atleast_2d(axes)
for row, (fp, cls, sub) in enumerate(picked):
    img = load(fp)
    heat, probs = hm_model.predict((img * args.scale)[None], verbose=0)
    heat = heat[0, ..., 0]
    up = tf.image.resize(heat[..., None], [SIZE, SIZE]).numpy()[..., 0]
    pred = subtypes[int(np.argmax(probs[0]))] if probs.shape[-1] == len(subtypes) else f"p={probs.ravel()[0]:.2f}"
    print(f"{cls:9s} {sub:20s} -> pred {pred:20s} | heat min {heat.min():.2f} max {heat.max():.2f} std {heat.std():.3f}")

    axes[row, 0].imshow(img.astype("uint8")); axes[row, 0].set_title(f"{cls} / {sub}", fontsize=8)
    axes[row, 1].imshow(up, cmap="jet", vmin=0, vmax=1); axes[row, 1].set_title("heatmap (0-1, fixed scale)", fontsize=8)
    axes[row, 2].imshow(img.astype("uint8")); axes[row, 2].imshow(up, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[row, 2].set_title(f"overlay | pred: {pred}", fontsize=8)
    for a in axes[row]:
        a.axis("off")
plt.tight_layout()
plt.savefig(args.out, dpi=130)
print(f"\nSaved {args.out}")
if matplotlib.get_backend().lower() != "agg":
    plt.show()