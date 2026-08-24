# ── Step 1: train initial model on labeled data ────────────────────────────
python scripts/train_model.py --name stage1 --data configs/data_stage1.yaml \
  --weights yolo11s.pt --epochs 150 --lr0 1e-3 \
  --train-eval-data configs/data_stage1.yaml

# ── Step 2: generate ID pseudo-labels ──────────────────────────────────────
python scripts/generate_psd_labels.py \
  --weights experiments/stage1/weights/best.pt \
  --videos /datashare/HW1/id_video_data/4_2_24_B_2.mp4 \
           /datashare/HW1/id_video_data/20_2_24_1.mp4 \
          /datashare/HW1/ood_video_data/4_2_24_A_1.mp4 \
  --out pseudo_id --conf-track 0.35 --conf-keep 0.60 --min-track-len 8 --stride 10

# ── Step 3: refine with ID pseudo-labels (build combined set, then retrain) ─
python scripts/build_dataset.py --labeled-yaml configs/data_stage1.yaml \
  --pseudo-dir pseudo_id --labeled-repeat 5 --out-yaml data_stage2_combined.yaml
python scripts/train_model.py --name stage2 --data data_stage2_combined.yaml \
  --weights experiments/stage1/weights/best.pt --lr0 5e-4 \
  --train-eval-data configs/data_stage2.yaml

# ── Step 4: repeat 2–3 on the OOD video ────────────────────────────────────
python scripts/generate_psd_labels.py \
  --weights experiments/stage2/weights/best.pt \
  --videos /datashare/HW1/ood_video_data/surg_1.mp4 \
  --out pseudo_ood --conf-track 0.25 --conf-keep 0.45 --min-track-len 5 \
  --vid-stride 2 --imgsz 480 --half
python scripts/build_dataset.py --labeled-yaml configs/data_stage1.yaml \
  --pseudo-dir pseudo_ood --labeled-repeat 5 --out-yaml data_stage3_combined.yaml
python scripts/train_model.py --name stage3 --data data_stage3_combined.yaml \
  --weights experiments/stage2/weights/best.pt --lr0 5e-4 \
  --train-eval-data configs/data_stage3.yaml
