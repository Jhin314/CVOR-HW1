"""Generate pseudo-labels from video(s) - the other pipeline primitive:
(model + video(s)) -> <out>/images + <out>/labels.  No training, no dataset building.

Selection heuristic: keep only tracks that persist >= --min-track-len frames with
mean confidence >= --conf-keep. With --majority-vote (default), every box in a
track is relabeled to the track's most-frequent class, removing class flicker.

ID vs OOD is just which videos you pass (and looser thresholds for OOD):
  ID  : --conf-track 0.35 --conf-keep 0.60 --min-track-len 8
  OOD : --conf-track 0.25 --conf-keep 0.45 --min-track-len 5

Examples
--------
python scripts/generate_psd_labels.py --weights experiments/stage1/weights/best.pt \
       --videos /datashare/HW1/id_video_data/*.mp4 --out pseudo_id \
       --conf-track 0.35 --conf-keep 0.60 --min-track-len 8

python scripts/generate_psd_labels.py --weights experiments/stage2/weights/best.pt \
       --videos /datashare/HW1/ood_video_data/surg_1.mp4 --out pseudo_ood \
       --conf-track 0.25 --conf-keep 0.45 --min-track-len 5 --vid-stride 2 --imgsz 480 --half
"""
import os
import glob
import shutil
import argparse
from collections import defaultdict, Counter

import cv2
import numpy as np
from ultralytics import YOLO


def gen_from_video(model, video, tag, args, img_dir, lbl_dir):
    track_conf = defaultdict(list)   # tid -> [conf,...]   (all frames)
    track_cls  = defaultdict(list)   # tid -> [class,...]  (all frames)
    keep_boxes = defaultdict(list)   # fi -> [(tid, cls, box), ...]  (STRIDE frames)
    keep_imgs  = {}                  # fi -> image                    (STRIDE frames)

    print(f"\n[{tag}] tracking {video} ...", flush=True)
    for fi, r in enumerate(model.track(source=video, conf=args.conf_track, persist=True,
                                       stream=True, verbose=False,
                                       vid_stride=args.vid_stride, imgsz=args.imgsz,
                                       half=args.half)):
        if fi % 100 == 0:
            print(f"  [{tag}] frame {fi:>6} | tracks so far: {len(track_conf)}", flush=True)
        if r.boxes is None or r.boxes.id is None:
            continue
        ids   = r.boxes.id.cpu().numpy().astype(int)
        cls   = r.boxes.cls.cpu().numpy().astype(int)
        conf  = r.boxes.conf.cpu().numpy()
        xywhn = r.boxes.xywhn.cpu().numpy()
        for tid, c, cf, box in zip(ids, cls, conf, xywhn):
            track_conf[int(tid)].append(float(cf))
            track_cls[int(tid)].append(int(c))
            if fi % args.stride == 0:
                keep_boxes[fi].append((int(tid), int(c), box))
        if fi % args.stride == 0:
            keep_imgs[fi] = r.orig_img

    good = {tid for tid, cs in track_conf.items()
            if len(cs) >= args.min_track_len and np.mean(cs) >= args.conf_keep}
    majority = {tid: Counter(track_cls[tid]).most_common(1)[0][0] for tid in good}

    if args.majority_vote:
        flips = sum(1 for tid in good for c in track_cls[tid] if c != majority[tid])
        total = sum(len(track_cls[tid]) for tid in good) or 1
        print(f"  majority vote reassigned {flips}/{total} boxes to their track's class")

    written = 0
    for fi in sorted(keep_boxes):
        dets = []
        for (tid, c, box) in keep_boxes[fi]:
            if tid not in good:
                continue
            dets.append((majority[tid] if args.majority_vote else c, box))
        if not dets:
            continue
        name = f"{tag}_{fi:06d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), keep_imgs[fi])
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            for c, (cx, cy, w, h) in dets:
                f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        written += 1
    print(f"{video}: kept {len(good)} tracks -> {written} pseudo frames")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="model used to generate labels")
    ap.add_argument("--videos", nargs="+", required=True, help="one or more video paths")
    ap.add_argument("--out", required=True, help="output dir (gets images/ and labels/)")

    ap.add_argument("--conf-track", type=float, default=0.30, dest="conf_track",
                    help="detection conf during tracking (recall)")
    ap.add_argument("--conf-keep", type=float, default=0.50, dest="conf_keep",
                    help="min mean track confidence to trust a track")
    ap.add_argument("--min-track-len", type=int, default=5, dest="min_track_len",
                    help="min frames a track must persist")
    ap.add_argument("--stride", type=int, default=10, help="write every Nth frame (dedup)")
    ap.add_argument("--majority-vote", action=argparse.BooleanOptionalAction, default=True,
                    help="relabel each track to its majority class")

    # tracking speed knobs
    ap.add_argument("--vid-stride", type=int, default=1, dest="vid_stride",
                    help="process every Nth frame during tracking (speed)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 inference")

    ap.add_argument("--append", action="store_true",
                    help="add to an existing --out instead of clearing it first")
    args = ap.parse_args()

    if not args.append and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for v in args.videos:
        assert os.path.exists(v), f"video not found: {v!r}"

    model = YOLO(args.weights)
    total = 0
    for i, v in enumerate(args.videos):
        tag = "vid" if len(args.videos) == 1 else f"vid{i}"
        total += gen_from_video(model, v, tag, args, img_dir, lbl_dir)
    print(f"\nDONE: {total} pseudo frames written to {img_dir}")


if __name__ == "__main__":
    main()