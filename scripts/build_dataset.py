"""Build a combined training set - the glue between generate and train:
(labeled yaml + pseudo dir) -> combined data yaml.

Writes a train list (labeled x N, then pseudo once) and a YOLO data yaml whose
train: points at that list. Validation stays the real labeled val set.
The output yaml is separate from the input, so it never clobbers configs/.

Example
-------
python scripts/build_dataset.py --labeled-yaml configs/data_stage1.yaml \
       --pseudo-dir pseudo_id --labeled-repeat 5 --out-yaml data_stage2_combined.yaml
"""
import os
import glob
import argparse

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-yaml", required=True, dest="labeled_yaml",
                    help="base data yaml (path/train/val/names for the real labeled set)")
    ap.add_argument("--pseudo-dir", required=True, dest="pseudo_dir",
                    help="pseudo-label dir containing images/ (from generate_psd_labels.py)")
    ap.add_argument("--out-yaml", required=True, dest="out_yaml",
                    help="combined data yaml to write (keep distinct from the input yaml!)")
    ap.add_argument("--labeled-repeat", type=int, default=5, dest="labeled_repeat",
                    help="oversample factor for real labels vs pseudo")
    ap.add_argument("--train-list", default=None, dest="train_list",
                    help="path for the image-list txt (default: <out-yaml>_train.txt)")
    args = ap.parse_args()

    assert os.path.exists(args.labeled_yaml), f"labeled yaml not found: {args.labeled_yaml!r}"
    assert os.path.abspath(args.out_yaml) != os.path.abspath(args.labeled_yaml), \
        "out-yaml must differ from labeled-yaml (else you overwrite your input config)"

    with open(args.labeled_yaml) as f:
        d = yaml.safe_load(f)
    root = d["path"]
    labeled_train = os.path.join(root, d["train"])
    labeled = sorted(glob.glob(os.path.join(labeled_train, "*.jpg")) +
                     glob.glob(os.path.join(labeled_train, "*.png")))
    assert labeled, f"no labeled images under {labeled_train!r} (check labeled-yaml path/train)"

    pimg = os.path.join(args.pseudo_dir, "images")
    pseudo = sorted(glob.glob(os.path.join(pimg, "*.jpg")) +
                    glob.glob(os.path.join(pimg, "*.png")))
    assert pseudo, f"no pseudo images under {pimg!r} (did generate_psd_labels.py run?)"

    train_list = os.path.abspath(args.train_list or (os.path.splitext(args.out_yaml)[0] + "_train.txt"))
    with open(train_list, "w") as f:
        for _ in range(args.labeled_repeat):
            for p in labeled:
                f.write(os.path.abspath(p) + "\n")
        for p in pseudo:
            f.write(os.path.abspath(p) + "\n")

    out = dict(d)
    out["path"] = ""                                # absolute paths below
    out["train"] = train_list
    out["val"] = os.path.join(root, d["val"])       # keep validating on real labels
    with open(args.out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    n = len(labeled) * args.labeled_repeat + len(pseudo)
    print(f"labeled {len(labeled)} x{args.labeled_repeat} + pseudo {len(pseudo)} = {n} train images")
    print(f"wrote {args.out_yaml}  (train list: {train_list})")


if __name__ == "__main__":
    main()