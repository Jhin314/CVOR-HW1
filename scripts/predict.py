"""predict.py - run the trained model on a single image.

Prints one line per detection in the assignment's YOLO format:
    x_center y_center w h conf class
where x_center, y_center, w, h are normalized to [0, 1] and class is the class id.
Optionally writes the same lines to a .txt and/or saves an annotated image.

Example:
    python scripts/predict.py --weights experiments/stage3/weights/best.pt \
                              --source frame.jpg --out preds.txt --save
"""
import argparse

import cv2
from ultralytics import YOLO

NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
COLORS = [(0, 200, 0), (0, 128, 255), (255, 64, 64),
          (200, 0, 200), (0, 200, 200), (64, 64, 255)]  # BGR, cycled per class id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="experiments/stage3/weights/best.pt", help="path to best.pt")
    ap.add_argument("--source", required=True, help="input image")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None, help="optional .txt to write detections to")
    ap.add_argument("--save", action="store_true", help="also save an annotated image")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names

    r = model.predict(args.source, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

    lines, dets = [], []
    if r.boxes is not None:
        xywhn = r.boxes.xywhn.cpu().numpy()          # normalized cx, cy, w, h
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        for (cx, cy, w, h), c, cf in zip(xywhn, cls, conf):
            lines.append(f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cf:.4f} {c}")  # x_center y_center w h conf class
            dets.append((c, cx, cy, w, h, cf))

    print("\n".join(lines) if lines else "# no detections")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"# wrote {len(lines)} detections -> {args.out}")

    if args.save:
        img = cv2.imread(args.source)
        H, W = img.shape[:2]
        thick, fscale = max(1, round(W / 640)), max(0.5, W / 1280)
        for c, cx, cy, w, h, cf in dets:
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            color = COLORS[c % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            label = f"{names.get(c, NAMES.get(c, c))} {cf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), thick, cv2.LINE_AA)
        out_img = args.source.rsplit(".", 1)[0] + "_pred.jpg"
        cv2.imwrite(out_img, img)
        print(f"# saved annotated image -> {out_img}")


if __name__ == "__main__":
    main()
