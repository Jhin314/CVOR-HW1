"""Ground truth vs final-model predictions, side by side, on labeled images.
Left = ground-truth boxes, right = the model's predictions (with confidence).
Defaults to the stage-3 (final) model and the val split (where real GT exists).
Usage: python plot_gt_vs_pred.py
       python plot_gt_vs_pred.py --weights experiments/stage3/weights/best.pt --n 6 --conf 0.25
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
from ultralytics import YOLO
NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
CLASS_COLORS = {0: "limegreen", 1: "deepskyblue", 2: "red"}
def collect(root, split):
    img_dir = os.path.join(root, "images", split)
    items = []
    for img in sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                      glob.glob(os.path.join(img_dir, "*.png"))):
        lbl = img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        items.append((img, os.path.splitext(lbl)[0] + ".txt"))
    return items
def read_gt(lbl, W, H):
    boxes = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) >= 5:
                c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
                boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, None))
    return boxes
def draw(ax, im, boxes):
    ax.imshow(im)
    for c, x, y, w, h, conf in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor=CLASS_COLORS.get(c, "white"), lw=2))
        label = NAMES.get(c, str(c)) + (f" {conf:.2f}" if conf is not None else "")
        ax.text(x, y - 3, label, color=CLASS_COLORS.get(c, "white"), fontsize=7, va="bottom")
    ax.set_xticks([]); ax.set_yticks([])
def predict_boxes(model, img_path, conf):
    r = model.predict(img_path, conf=conf, verbose=False)[0]
    out = []
    if r.boxes is not None:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, cf):
            out.append((c, x1, y1, x2 - x1, y2 - y1, float(p)))
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="experiments/stage3/weights/best.pt")
    ap.add_argument("--root", default="/datashare/HW1/labeled_image_data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default="gt_vs_pred.png")
    args = ap.parse_args()
    items = collect(args.root, args.split)
    assert items, f"no images under {os.path.join(args.root, 'images', args.split)!r}"
    n = min(args.n, len(items))
    random.seed(args.seed)
    sample = random.sample(items, n)
    model = YOLO(args.weights)
    per_row = 2
    ncols = 2 * per_row
    nrows = (n + per_row - 1) // per_row
    h0, w0 = cv2.imread(sample[0][0]).shape[:2]      # match cell shape to image aspect
    cell_w = 3.4
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell_w * ncols, cell_w * (h0 / w0) * nrows + 0.9),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, (img_path, lbl) in enumerate(sample):
        im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        r = i // per_row
        c0 = (i % per_row) * 2
        gt_ax, pr_ax = axes[r][c0], axes[r][c0 + 1]
        gt_ax.set_visible(True); pr_ax.set_visible(True)
        draw(gt_ax, im, read_gt(lbl, W, H))
        draw(pr_ax, im, predict_boxes(model, img_path, args.conf))
        gt_ax.set_title(f"GT  {os.path.basename(img_path)}", fontsize=8)
        pr_ax.set_title("prediction", fontsize=8)
    handles = [Patch(color=CLASS_COLORS[c], label=NAMES[c]) for c in sorted(NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"GT vs prediction - {args.split} split", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96], pad=0.3, w_pad=0.2, h_pad=0.4)
    fig.subplots_adjust(wspace=0.02, hspace=0.08)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  ({n} images)")
if __name__ == "__main__":
    main()
