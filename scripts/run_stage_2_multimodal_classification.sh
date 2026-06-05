#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
MEDICALNET34_WEIGHTS="third_party/pretrained/resnet_34_23dataset.pth"

COMMON_ARGS=(
  --manifest outputs/phase2_manifest_gt_mask.csv
  --clinical-num-cols age menopause_missing
  --clinical-cat-cols HR HER2 menopause
  --batch-size 4
  --epochs 80
  --lr 1e-5
  --weight-decay 5e-4
  --roi-size 96 160 160
  --roi-margin 16
  --min-component-size 16
  --d-model 256
  --n-heads 4
  --num-layers 1
  --dropout 0.3
  --dim-feedforward 512
  --encoder-type official_medicalnet_resnet34
  --encoder-base-channels 32
  --encoder-out-channels 128
  --medicalnet-pretrained-path "$MEDICALNET34_WEIGHTS"
  --normalize-mode zscore
  --target-col pCR
  --num-workers 0
  --amp
  --seed 42
  --scheduler plateau
  --grad-clip 1.0
  --early-stop-patience 15
  --monitor-metric auroc
  --threshold-objective balanced_accuracy
  --loss-bce-weight 1.0
  --loss-focal-weight 0.0
  --focal-gamma 2.0
  --focal-alpha 0.25
)

# 1. Build GT-mask manifest
./venv/bin/python scripts/build_phase2_manifest.py \
  --csv outputs/BreastDCEDL_ISPY1_ISPY2_noDuke_80_10_10_split.csv \
  --spy1-root data/BreastDCEDL_ISPY1_min_crop \
  --spy2-root data/BreastDCEDL_ISPY2_min_crop \
  --mask-source gt \
  --out outputs/phase2_manifest_gt_mask.csv

# 2. Build predicted-mask manifest for internal_test evaluation
./venv/bin/python scripts/build_phase2_manifest.py \
  --csv outputs/BreastDCEDL_ISPY1_ISPY2_noDuke_80_10_10_split.csv \
  --spy1-root data/BreastDCEDL_ISPY1_min_crop \
  --spy2-root data/BreastDCEDL_ISPY2_min_crop \
  --mask-source pred_nnunet \
  --pred-mask-root outputs/nnunet_predictions/Dataset112_BreastTumorISPY_internal_test \
  --out outputs/phase2_manifest_pred_mask.csv

# 3. Image-only GT ROI
"$PYTHON_BIN" -m pcr_phase2.train \
  "${COMMON_ARGS[@]}" \
  --output-dir outputs_v2/phase2_mednet34_image_only_gt_roi_margin16_bs4 \
  --image-only \
  --mask-mode gt \
  --fusion-type concat \
  --enable-augmentation \
  --augmentation-strength light

# 4. Clinical-only
"$PYTHON_BIN" -m pcr_phase2.train \
  "${COMMON_ARGS[@]}" \
  --output-dir outputs_v2/phase2_mednet34_clinical_only_margin16_bs4 \
  --clinical-only \
  --mask-mode none \
  --disable-roi-crop \
  --fusion-type attention

# 5. GT ROI + clinical + concat
"$PYTHON_BIN" -m pcr_phase2.train \
  "${COMMON_ARGS[@]}" \
  --output-dir outputs_v2/phase2_mednet34_gt_roi_concat_margin16_bs4 \
  --mask-mode gt \
  --fusion-type concat \
  --enable-augmentation \
  --augmentation-strength light

# 6. GT ROI + clinical + attention
"$PYTHON_BIN" -m pcr_phase2.train \
  "${COMMON_ARGS[@]}" \
  --output-dir outputs_v2/phase2_mednet34_gt_roi_clinical_attention_margin16_bs4_bce_light \
  --mask-mode gt \
  --fusion-type attention \
  --enable-augmentation \
  --augmentation-strength light

# 7. Predicted-mask internal-test evaluation for MedNet34 attention model
"$PYTHON_BIN" -m pcr_phase2.evaluate \
  --manifest outputs/phase2_manifest_pred_mask.csv \
  --checkpoint outputs_v2/phase2_mednet34_gt_roi_clinical_attention_margin16_bs4_bce_light/checkpoints/best.pt \
  --split internal_test \
  --output-dir outputs_v2/phase2_mednet34_gt_roi_clinical_attention_margin16_bs4_bce_light_eval_pred_mask_internal_test

# 8. GT-mask internal-test evaluation for MedNet34 attention model
"$PYTHON_BIN" -m pcr_phase2.evaluate \
  --manifest outputs/phase2_manifest_gt_mask.csv \
  --checkpoint outputs_v2/phase2_mednet34_gt_roi_clinical_attention_margin16_bs4_bce_light/checkpoints/best.pt \
  --split internal_test \
  --output-dir outputs_v2/phase2_mednet34_gt_roi_clinical_attention_margin16_bs4_bce_light_eval_gt_mask_internal_test
