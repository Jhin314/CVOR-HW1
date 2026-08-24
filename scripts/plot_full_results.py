"""Full training figure per stage - all losses and all mAPs, train AND val.
Reads per stage:
  - results.csv    -> train/val box,cls,dfl losses + val mAP50, mAP50-95
  - train_map.csv  -> train mAP50, mAP50-95 (written by train_model.py --train-eval-data)
Layout (1x3): total loss | mAP50 | mAP50-95.
Train = solid, Val = dashed. One figure per stage: full_results_<name>.png
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ---------------- config ----------------
# (folder name, run dir, figure title describing what the stage trained on)
STAGES = [
    ("stage1", "experiments/stage1", "Stage 1 :Labeled data"),
    ("stage2", "experiments/stage2", "Stage 2 :Labeled data + Pseudo-labeled ID data"),
    ("stage3", "experiments/stage3", "Stage 3: Labeled data + Pseudo-labeled OOD data"),
]
# -----------------------------------------
def read_results(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "results.csv"))))
    ep = [int(r["epoch"]) for r in rows]
    col = lambda k: [float(r[k]) for r in rows]
    return ep, col
def read_train_map(run_dir):
    p = os.path.join(run_dir, "train_map.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    # a re-run may have appended a second block of epochs; keep the LAST value per
    # epoch (most recent run, matching results.csv) and sort, to avoid diagonal artifacts
    d = {}
    for r in rows:
        d[int(r["epoch"])] = (float(r["train_mAP50"]), float(r["train_mAP50-95"]))
    eps = sorted(d)
    return eps, [d[e][0] for e in eps], [d[e][1] for e in eps]
def plot_stage(name, run_dir, title):
    ep, col = read_results(run_dir)
    tm = read_train_map(run_dir)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # --- total loss ---
    ttr = [b + c + d for b, c, d in zip(col("train/box_loss"), col("train/cls_loss"), col("train/dfl_loss"))]
    tva = [b + c + d for b, c, d in zip(col("val/box_loss"), col("val/cls_loss"), col("val/dfl_loss"))]
    ax[0].plot(ep, ttr, color="tab:red", lw=1.7, label="train")
    ax[0].plot(ep, tva, color="tab:purple", lw=1.5, ls="--", label="val")
    ax[0].set_title("total loss (box+cls+dfl)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8)
    # --- mAP: train (from train_map.csv) solid, val (from results.csv) dashed ---
    for a, val_key, tmi, mtitle in [(ax[1], "metrics/mAP50(B)", 1, "mAP50"),
                                    (ax[2], "metrics/mAP50-95(B)", 2, "mAP50-95")]:
        if tm is not None:
            a.plot(tm[0], tm[tmi], color="tab:blue", lw=1.6, label="train")
        a.plot(ep, col(val_key), color="tab:orange", lw=1.6, ls="--", label="val")
        a.set_ylim(0, 1); a.set_title(mtitle); a.set_xlabel("epoch"); a.legend(fontsize=8)
    if tm is None:
        ax[1].text(0.5, 0.1, "no train_map.csv (rerun with --train-eval-data)",
                   transform=ax[1].transAxes, ha="center", fontsize=8, color="grey")
    fig.suptitle(f"{title}: training results", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"full_results_{name}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")
def main():
    for name, run_dir, title in STAGES:
        if os.path.exists(os.path.join(run_dir, "results.csv")):
            plot_stage(name, run_dir, title)
        else:
            print(f"(skip {name}: no results.csv in {run_dir})")
if __name__ == "__main__":
    main()