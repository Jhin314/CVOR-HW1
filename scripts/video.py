"""video.py - run a trained YOLO model on a video and write an annotated copy.

Boxes are drawn with OpenCV, colored per class, labeled "<class> <conf>".

Example:
    python video.py --weights experiments/stage3/weights/best.pt \
                    --source /datashare/HW1/ood_video_data/surg_1.mp4 \
                    --out ood_stage3.mp4 --conf 0.25
"""
import argparse

import cv2
from ultralytics import YOLO

COLORS = [(0, 200, 0), (0, 128, 255), (255, 64, 64),
          (200, 0, 200), (0, 200, 200), (64, 64, 255)]  # BGR, cycled per class id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--source", required=True, help="input video")
    ap.add_argument("--out", default="annotated.mp4", help="output video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    thick  = max(1, round(w / 640))
    fscale = max(0.5, w / 1280)

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
                color = COLORS[c % len(COLORS)]
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, color, thick)
                label = f"{names[c]} {cf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
                cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw, p1[1]), color, -1)
                cv2.putText(frame, label, (p1[0], p1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), thick, cv2.LINE_AA)

        writer.write(frame)
        fi += 1
        if fi % 100 == 0:
            print(f"  frame {fi}/{total}", flush=True)

    cap.release()
    writer.release()
    print(f"done: {fi} frames -> {args.out}")


if __name__ == "__main__":
    main()
