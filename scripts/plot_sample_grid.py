"""Sample grid of labeled images with ground-truth boxes.

Pulls from BOTH train and val; each image's split is shown by a colored border and
its title. Boxes are colored per class. Ground truth only - no model needed.

Usage: python plot_sample_grid.py            (defaults below)
       python plot_sample_grid.py --n 6 --seed 1
"""
import os
import glob
import random
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
CLASS_COLORS = {0: "limegreen", 1: "deepskyblue", 2: "red"}
SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange"}


def collect(root):
    items = []
    for split in ("train", "val"):
        img_dir = os.path.join(root, "images", split)
        for img in sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                          glob.glob(os.path.join(img_dir, "*.png"))):
            lbl = img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
            lbl = os.path.splitext(lbl)[0] + ".txt"
            items.append((img, lbl, split))
    return items


def read_boxes(lbl):
    boxes = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) >= 5:
                boxes.append((int(float(p[0])), *map(float, p[1:5])))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/datashare/HW1/labeled_image_data")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sample_grid.png")
    args = ap.parse_args()

    items = collect(args.root)
    assert len(items) >= args.n, f"only {len(items)} labeled images found under {args.root!r}"
    random.seed(args.seed)
    sample = random.sample(items, args.n)

    ncols = 3
    nrows = (args.n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (img_path, lbl, split) in zip(axes, sample):
        im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        ax.imshow(im)
        for c, cx, cy, w, h in read_boxes(lbl):
            ax.add_patch(Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H,
                                   fill=False, edgecolor=CLASS_COLORS.get(c, "white"), lw=2))
        ax.set_title(f"[{split}]  {os.path.basename(img_path)}", fontsize=9, color=SPLIT_COLORS[split])
        for sp in ax.spines.values():                      # colored border = split
            sp.set_edgecolor(SPLIT_COLORS[split]); sp.set_linewidth(4)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[args.n:]:                                # hide unused cells
        ax.set_visible(False)

    handles = [Patch(color=CLASS_COLORS[c], label=NAMES[c]) for c in sorted(NAMES)]
    handles += [Patch(color=SPLIT_COLORS[s], label=f"{s} (border)") for s in ("train", "val")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Labeled samples with ground-truth boxes", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  ({args.n} images from train+val)")


if __name__ == "__main__":
    main()
