"""EDA for the surgical-tools dataset.

Produces seven figures (saved to OUT_DIR) + a printed summary:
  1. class distribution           -> class_distribution.png
  2. boxes-per-image distribution -> boxes_per_image.png
  3. class co-occurrence          -> cooccurrence.png
  4. spatial heatmap of centers   -> spatial_heatmap.png
  5. ID vs OOD appearance shift    -> id_ood_shift.png (HSV)
  6. ID vs OOD RGB + intensity     -> rgb_intensity_shift.png

Analyses 1-4 use the full labeled set (train+val) since val alone is tiny.
Analyses 5-7 sample frames from the ID/OOD videos.
"""
import os
import glob
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- config ----------------
ROOT       = "/datashare/HW1/labeled_image_data"
LABEL_DIRS = [os.path.join(ROOT, "labels", "train"),
              os.path.join(ROOT, "labels", "val")]
NAMES      = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}

ID_VIDEOS  = ["/datashare/HW1/id_video_data/4_2_24_B_2.mp4",
              "/datashare/HW1/id_video_data/20_2_24_1.mp4"]
OOD_VIDEO  = "/datashare/HW1/ood_video_data/surg_1.mp4"
FRAMES_PER_SRC = 80          # frames sampled per source for the color analysis

OUT_DIR    = "eda_out"
# -----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def load_labels(label_dirs):
    """Return list of per-image detections: [[(cls, cx, cy, w, h), ...], ...]."""
    per_image = []
    for d in label_dirs:
        for txt in sorted(glob.glob(os.path.join(d, "*.txt"))):
            dets = []
            with open(txt) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    c = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    dets.append((c, cx, cy, w, h))
            per_image.append(dets)
    return per_image


def fig1_class_distribution(per_image, classes):
    inst = Counter()
    imgs = Counter()                       # images containing >=1 of the class
    for dets in per_image:
        present = set()
        for c, *_ in dets:
            inst[c] += 1
            present.add(c)
        for c in present:
            imgs[c] += 1

    labels = [NAMES.get(c, str(c)) for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, [inst[c] for c in classes], 0.4, label="instances")
    ax.bar(x + 0.2, [imgs[c] for c in classes], 0.4, label="images w/ class")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("count"); ax.set_title("Class distribution (train+val)")
    ax.legend()
    for i, c in enumerate(classes):
        ax.text(i - 0.2, inst[c], str(inst[c]), ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, imgs[c], str(imgs[c]), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)
    print("class instances:", dict(inst))
    print("images per class:", dict(imgs))


def fig2_boxes_per_image(per_image):
    counts = [len(d) for d in per_image]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(1, max(counts) + 2) - 0.5
    ax.hist(counts, bins=bins, rwidth=0.9, color="red", edgecolor="black")
    ax.set_xlabel("boxes per image"); ax.set_ylabel("# images")
    ax.set_title(f"Boxes per image (mean={np.mean(counts):.2f})")
    ax.set_xticks(range(1, max(counts) + 1))
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "boxes_per_image.png"), dpi=150)
    plt.close(fig)
    print("boxes/image: mean %.2f, min %d, max %d" % (np.mean(counts), min(counts), max(counts)))


def fig3_cooccurrence(per_image, classes):
    idx = {c: i for i, c in enumerate(classes)}      # classes = [0,1,2] -> Empty, Tweezers, Needle_driver
    n = len(classes)
    m = np.zeros((n, n), dtype=int)
    for dets in per_image:
        present = sorted({c for c, *_ in dets})       # distinct classes in this image
        for a in present:
            m[idx[a], idx[a]] += 1                     # diagonal: # images containing class a
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                m[idx[present[i]], idx[present[j]]] += 1   # off-diagonal: # images containing BOTH
                m[idx[present[j]], idx[present[i]]] += 1

    labels = [NAMES.get(c, str(c)) for c in classes]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(m, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_title("Co-occurrence (diag = # images w/ class)")
    thr = m.max() * 0.6
    for i in range(n):
        for j in range(n):
            ax.text(j, i, m[i, j], ha="center", va="center",
                    color="white" if m[i, j] < thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cooccurrence.png"), dpi=150)
    plt.close(fig)


def fig4_spatial_heatmap(per_image, classes):
    fig, ax = plt.subplots(figsize=(4, 4))
    xs = [cx for dets in per_image for cc, cx, cy, w, h in dets]
    ys = [cy for dets in per_image for cc, cx, cy, w, h in dets]
    if xs:
        ax.hist2d(xs, ys, bins=20, range=[[0, 1], [0, 1]], cmap="hot")
    ax.set_title(f"all (n={len(xs)})")
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)          # image coords: y down
    ax.set_aspect("equal")
    fig.suptitle("Box-center spatial distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spatial_heatmap.png"), dpi=150)
    plt.close(fig)
    

def _avg_hsv_hist(videos, n_frames):
    """Average, over sampled frames, of per-channel HSV histograms (normalized)."""
    import cv2
    RANGES = [180, 256, 256]   # cv2 Hue spans 0-179; Saturation/Value span 0-255
    hs = np.zeros((3, 256))
    total = 0
    for v in videos:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print(f"  (skip, cannot open {v})")
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames
        for fi in np.linspace(0, max(n - 1, 0), min(n_frames, n)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for ch in range(3):
                hs[ch] += cv2.calcHist([hsv], [ch], None, [256], [0, RANGES[ch]]).ravel()
            total += 1
        cap.release()
    if total:
        hs /= total
        hs /= hs.sum(axis=1, keepdims=True) + 1e-9
    return hs, total


def fig5_id_ood_shift():
    id_h, id_n   = _avg_hsv_hist(ID_VIDEOS, FRAMES_PER_SRC)
    ood_h, ood_n = _avg_hsv_hist([OOD_VIDEO], FRAMES_PER_SRC)
    if id_n == 0 or ood_n == 0:
        print("  (id/ood shift skipped: missing video frames)")
        return
    chans = ["Hue", "Saturation", "Value"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, ch in enumerate(chans):            # i: int index; ch: channel label
        a = axes[i]
        a.plot(id_h[i],  label=f"ID (n={id_n})")
        a.plot(ood_h[i], label=f"OOD (n={ood_n})")
        a.set_title(ch); a.set_xlabel("bin"); a.legend()
    fig.suptitle("ID vs OOD appearance shift (avg normalized HSV histograms)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "id_ood_shift.png"), dpi=150)
    plt.close(fig)
    # quick scalar summary: mean brightness/saturation
    bins = np.arange(256)
    for name, h in [("ID", id_h), ("OOD", ood_h)]:
        print(f"  {name}: mean V={np.sum(h[2]*bins):.1f}, mean S={np.sum(h[1]*bins):.1f}")


def _avg_rgb_gray_hist(videos, n_frames):
    """One pass: averaged normalized RGB (3,256) and grayscale-intensity (256,) hists."""
    import cv2
    rgb = np.zeros((3, 256)); gray = np.zeros(256); total = 0
    CH = [2, 1, 0]   # frame is BGR -> map to R, G, B
    for v in videos:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print(f"  (skip, cannot open {v})"); continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames
        for fi in np.linspace(0, max(n - 1, 0), min(n_frames, n)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            for out_i, ch in enumerate(CH):
                rgb[out_i] += cv2.calcHist([frame], [ch], None, [256], [0, 256]).ravel()
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray += cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
            total += 1
        cap.release()
    if total:
        rgb /= total; rgb /= rgb.sum(axis=1, keepdims=True) + 1e-9
        gray /= total; gray /= gray.sum() + 1e-9
    return rgb, gray, total


def fig6_7_rgb_intensity_shift():
    id_rgb, id_gray, id_n     = _avg_rgb_gray_hist(ID_VIDEOS, FRAMES_PER_SRC)
    ood_rgb, ood_gray, ood_n  = _avg_rgb_gray_hist([OOD_VIDEO], FRAMES_PER_SRC)
    if id_n == 0 or ood_n == 0:
        print("  (rgb/intensity shift skipped: missing video frames)")
        return

    # --- one figure: RGB (3 panels) + intensity (1 panel); ID solid, OOD dashed ---
    chans = ["Red", "Green", "Blue"]; colors = ["tab:red", "tab:green", "tab:blue"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, ch in enumerate(chans):
        a = axes[i]
        a.plot(id_rgb[i],  color=colors[i], label=f"ID (n={id_n})")
        a.plot(ood_rgb[i], color=colors[i], ls="--", label=f"OOD (n={ood_n})")
        a.set_title(ch); a.set_xlabel("pixel value"); a.legend()
    axes[3].plot(id_gray,  color="black", label=f"ID (n={id_n})")
    axes[3].plot(ood_gray, color="black", ls="--", label=f"OOD (n={ood_n})")
    axes[3].set_title("Intensity (grayscale)"); axes[3].set_xlabel("value (0-255)"); axes[3].legend()
    fig.suptitle("ID vs OOD RGB + intensity distribution (avg normalized histograms)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "rgb_intensity_shift.png"), dpi=150)
    plt.close(fig)

    bins = np.arange(256)
    print(f"  mean intensity: ID={np.sum(id_gray*bins):.1f}  OOD={np.sum(ood_gray*bins):.1f}")
    for i, ch in enumerate(chans):
        print(f"  mean {ch}: ID={np.sum(id_rgb[i]*bins):.1f}  OOD={np.sum(ood_rgb[i]*bins):.1f}")



def main():
    per_image = load_labels(LABEL_DIRS)
    classes = sorted({c for dets in per_image for c, *_ in dets})
    print(f"loaded {len(per_image)} labeled images, classes={classes}")

    fig1_class_distribution(per_image, classes)
    fig2_boxes_per_image(per_image)
    fig3_cooccurrence(per_image, classes)
    fig4_spatial_heatmap(per_image, classes)

    print("id/ood appearance shift (sampling video frames)...")
    try:
        fig5_id_ood_shift()
    except Exception as e:
        print(f"  (id/ood shift failed: {e})")

    print("rgb + intensity shift (sampling video frames)...")
    try:
        fig6_7_rgb_intensity_shift()
    except Exception as e:
        print(f"  (rgb/intensity shift failed: {e})")

    print(f"\nfigures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()