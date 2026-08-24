"""Custom results plotting - keeps Ultralytics untouched.

Call plot_run(save_dir) right after model.train() to (re)generate results.png with:
  - loss components (box/cls/dfl; train solid, val dashed)
  - total loss (sum of components; train vs val)
  - one panel per metric (precision, recall, mAP50, mAP50-95)

plot_stages([...csvs]) makes the concatenated across-stages version for the report.
Reads CSVs with the stdlib only (no polars/pandas dependency).
"""
import os
import csv as _csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(csv_path):
    rows = list(_csv.DictReader(open(csv_path)))
    cols = list(rows[0].keys())
    data = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    x = np.array([int(r["epoch"]) for r in rows])
    return cols, data, x


def _components(cols):
    """Pair loss columns into (name, train_col, val_col_or_None), generic across tasks."""
    comps = []
    for c in cols:
        if "loss" in c and c.startswith("train/"):
            comp = c[len("train/"):]
            val_c = f"val/{comp}"
            comps.append((comp.replace("_loss", ""), c, val_c if val_c in cols else None))
    return comps


def plot_run(save_dir, out_name="results.png"):
    """Regenerate results.png for a single training run directory."""
    save_dir = str(save_dir)
    csv_path = os.path.join(save_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"[plot_run] no results.csv in {save_dir}")
        return
    cols, d, x = _read(csv_path)
    comps = _components(cols)
    metric_cols = [c for c in cols if "metric" in c]

    n_panels = 2 + len(metric_cols)
    ncols = (n_panels + 1) // 2
    fig, ax = plt.subplots(2, ncols, figsize=(4 * ncols, 8), tight_layout=True)
    ax = ax.ravel()
    colors = plt.cm.tab10.colors

    for k, (name, tr, va) in enumerate(comps):
        c = colors[k % len(colors)]
        ax[0].plot(x, d[tr], color=c, lw=1.6, label=f"{name} (train)")
        if va:
            ax[0].plot(x, d[va], color=c, lw=1.4, ls="--", label=f"{name} (val)")
    ax[0].set_title("loss components", fontsize=12)
    ax[0].legend(fontsize=7, ncol=len(comps))

    tr_total = sum(d[tr] for _, tr, _ in comps)
    ax[1].plot(x, tr_total, color="tab:red", lw=1.7, label="total (train)")
    if all(va for _, _, va in comps):
        ax[1].plot(x, sum(d[va] for _, _, va in comps),
                   color="tab:purple", lw=1.5, ls="--", label="total (val)")
    ax[1].set_title("total loss", fontsize=12)
    ax[1].legend(fontsize=8)

    for i, m in enumerate(metric_cols):
        ax[2 + i].plot(x, d[m], color="tab:blue", lw=1.6)
        ax[2 + i].set_title(m, fontsize=12)

    for j in range(n_panels, len(ax)):
        ax[j].set_visible(False)

    out = os.path.join(save_dir, out_name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plot_run] wrote {out}")
    return out


def plot_stages(csvs, out_path="results_all_stages.png",
                stage_labels=("Stage 1", "Stage 2", "Stage 3")):
    """Concatenate several run CSVs on one epoch axis (for the report)."""
    epochs, boundaries, offset = [], [], 0
    series = None
    for fn in csvs:
        cols, d, _ = _read(fn)
        if series is None:
            series = {c: [] for c in cols}
        n = len(d[cols[0]])
        for c in cols:
            series[c].extend(list(d[c]))
        epochs.extend([offset + i + 1 for i in range(n)])
        offset += n
        boundaries.append(offset)
    x = np.array(epochs)
    d = {c: np.array(v) for c, v in series.items()}
    comps = _components(list(d.keys()))
    metric_cols = [c for c in d if "metric" in c]

    def marks(a, label=False):
        y0, y1 = a.get_ylim(); start = 0
        for i, b in enumerate(boundaries):
            a.axvspan(start + 0.5, b + 0.5, alpha=0.04, color="grey" if i % 2 else "blue")
            if label and i < len(stage_labels):
                a.text((start + b) / 2, y0 + (y1 - y0) * 0.03, stage_labels[i],
                       ha="center", va="bottom", fontsize=7, color="dimgrey")
            start = b
        for b in boundaries[:-1]:
            a.axvline(b + 0.5, color="grey", ls=":", lw=0.9)

    n_panels = 2 + len(metric_cols); ncols = (n_panels + 1) // 2
    fig, ax = plt.subplots(2, ncols, figsize=(4 * ncols, 8), tight_layout=True)
    ax = ax.ravel(); colors = plt.cm.tab10.colors

    for k, (name, tr, va) in enumerate(comps):
        c = colors[k % len(colors)]
        ax[0].plot(x, d[tr], color=c, lw=1.5, label=f"{name} (train)")
        if va:
            ax[0].plot(x, d[va], color=c, lw=1.3, ls="--", label=f"{name} (val)")
    ax[0].set_title("loss components"); ax[0].legend(fontsize=7, ncol=len(comps)); marks(ax[0], True)

    tr_total = sum(d[tr] for _, tr, _ in comps)
    ax[1].plot(x, tr_total, color="tab:red", lw=1.6, label="total (train)")
    if all(va for _, _, va in comps):
        ax[1].plot(x, sum(d[va] for _, _, va in comps),
                   color="tab:purple", lw=1.4, ls="--", label="total (val)")
    ax[1].set_title("total loss"); ax[1].legend(fontsize=8); marks(ax[1], True)

    for i, m in enumerate(metric_cols):
        ax[2 + i].plot(x, d[m], color="tab:olive", lw=1.5); ax[2 + i].set_title(m); marks(ax[2 + i])

    for j in range(n_panels, len(ax)):
        ax[j].set_visible(False)
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[plot_stages] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "stages":
        plot_stages(sys.argv[2:])
    elif len(sys.argv) > 1:
        plot_run(sys.argv[1])
    else:
        print("usage: python custom_plots.py <run_dir>   |   python custom_plots.py stages a.csv b.csv c.csv")
