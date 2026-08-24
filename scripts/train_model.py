"""Train a YOLO model - one primitive of the SSL pipeline: (weights + data) -> best.pt.

Output goes to <project>/<name>/  (default project: <repo>/experiments), so each
stage lands in its OWN folder and stages never clobber each other. Re-running the
same --name overwrites that folder (exist_ok=True).

Pass --train-eval-data <labeled yaml> to also log per-epoch TRAIN mAP (evaluated on
the real labeled train split), so you can plot train+val mAP curves afterwards.

Examples
--------
python scripts/train_model.py --name stage1 --data configs/data_stage1.yaml \
       --weights yolo11s.pt --epochs 150 --lr0 1e-3 --train-eval-data configs/data_stage1.yaml

python scripts/train_model.py --name stage2 --data data_stage2_combined.yaml \
       --weights experiments/stage1/weights/best.pt --lr0 5e-4 \
       --train-eval-data configs/data_stage2.yaml
"""
import os
import csv
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent   # repo root (scripts/ -> ..)


def _stage_local(train_eval_data, dest):
    """Copy the labeled train images+labels to a local writable dir once and return a
    yaml pointing at it, so the val label-cache persists (no repeated /datashare scans)."""
    import glob
    import shutil
    import yaml
    with open(train_eval_data) as f:
        d = yaml.safe_load(f)
    src_img = os.path.join(d["path"], d["train"])
    src_lbl = src_img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    img_dst = os.path.join(dest, "images"); lbl_dst = os.path.join(dest, "labels")
    os.makedirs(img_dst, exist_ok=True); os.makedirs(lbl_dst, exist_ok=True)
    for img in glob.glob(os.path.join(src_img, "*.jpg")) + glob.glob(os.path.join(src_img, "*.png")):
        shutil.copy(img, img_dst)
        t = os.path.join(src_lbl, os.path.splitext(os.path.basename(img))[0] + ".txt")
        if os.path.exists(t):
            shutil.copy(t, lbl_dst)
    y = dict(d); y["path"] = os.path.abspath(dest); y["train"] = "images"; y["val"] = "images"
    yp = os.path.join(dest, "data.yaml")
    with open(yp, "w") as f:
        yaml.safe_dump(y, f)
    return yp


def make_train_map_logger(train_eval_data):
    """Callback: each epoch, load last.pt into a SEPARATE model and eval it on the
    LABELED train split, appending train mAP to <save_dir>/train_map.csv.

    A separate instance is REQUIRED: validating the live training model fuses it
    (Conv+BN merge), which breaks backprop. Labeled data is staged locally once so it
    isn't re-scanned from /datashare. Fully guarded so it never kills training."""
    state = {"local_yaml": None}
    def cb(trainer):
        try:
            sd = str(trainer.save_dir)
            out = os.path.join(sd, "train_map.csv")
            if state["local_yaml"] is None:                       # first call this run
                state["local_yaml"] = _stage_local(train_eval_data, os.path.join(sd, "train_eval_data"))
                with open(out, "w", newline="") as f:             # overwrite: a re-run starts clean
                    csv.writer(f).writerow(["epoch", "train_mAP50", "train_mAP50-95"])
            ckpt = getattr(trainer, "last", None)                 # last.pt (updated each epoch)
            if not (ckpt and os.path.exists(ckpt)):
                return
            res = YOLO(str(ckpt)).val(                            # separate instance -> live model untouched
                data=state["local_yaml"], split="train", verbose=False, plots=False,
                save_json=False, project=sd, name="train_eval", exist_ok=True,
            )
            with open(out, "a", newline="") as f:
                csv.writer(f).writerow([int(trainer.epoch) + 1, float(res.box.map50), float(res.box.map)])
        except Exception as e:
            print(f"(train-mAP logger skipped this epoch: {e})")
    return cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run/stage name -> output subfolder")
    ap.add_argument("--data", required=True, help="dataset yaml (may be a combined labeled+pseudo yaml)")
    ap.add_argument("--weights", default="yolo11s.pt",
                    help="starting weights: COCO (yolo11s.pt) or a prior stage's best.pt")
    ap.add_argument("--project", default=str(ROOT / "experiments"),
                    help="output root (default: <repo>/experiments)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = AutoBatch")
    ap.add_argument("--lr0", type=float, default=1e-3, help="lower (e.g. 5e-4) when continuing a model")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--cls", type=float, default=1.0, help="classification loss weight")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"],
                    help="image cache (ram fixes /datashare I/O starvation)")
    ap.add_argument("--train-eval-data", default=None, dest="train_eval_data",
                    help="labeled yaml to eval per-epoch TRAIN mAP on (enables train_map.csv)")
    args = ap.parse_args()

    assert os.path.exists(args.data), f"data yaml not found: {args.data!r} (run from repo root?)"
    if args.train_eval_data:
        assert os.path.exists(args.train_eval_data), \
            f"train-eval-data not found: {args.train_eval_data!r}"
    cache = {"ram": True, "disk": "disk", "none": False}[args.cache]

    model = YOLO(args.weights)
    if args.train_eval_data:
        model.add_callback("on_fit_epoch_end", make_train_map_logger(args.train_eval_data))

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cache=cache,
        patience=args.patience,
        cos_lr=True,
        optimizer="AdamW",
        lr0=args.lr0,
        cls=args.cls,
        # augmentation (main regularizer / OOD generalization lever on tiny data)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, close_mosaic=15, mixup=0.1, copy_paste=0.1,
        project=args.project, name=args.name, exist_ok=True,
    )
    print("best weights:", model.trainer.best)

    try:
        from custom_plots import plot_run
        plot_run(model.trainer.save_dir)
    except Exception as e:
        print(f"(plot_run skipped: {e})")


if __name__ == "__main__":
    main()