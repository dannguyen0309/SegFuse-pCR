#!/usr/bin/env bash
set -euo pipefail

export nnUNet_raw=/home/serverai/dannguyen/BreastCancerDetection-MRI/nnUNet_raw
export nnUNet_preprocessed=/home/serverai/dannguyen/BreastCancerDetection-MRI/nnUNet_preprocessed
export nnUNet_results=/home/serverai/dannguyen/BreastCancerDetection-MRI/nnUNet_results

python scripts/prepare_ispy_nnunet_dataset.py \
  --csv outputs/BreastDCEDL_ISPY1_ISPY2_noDuke_80_10_10_split.csv \
  --spy1-root data/BreastDCEDL_ISPY1_min_crop \
  --spy2-root data/BreastDCEDL_ISPY2_min_crop \
  --dataset-id 112 \
  --dataset-name BreastTumorISPY \
  --out-audit-dir outputs/nnunet_ispy_audit \
  --copy-mode copy \
  --overwrite

nnUNetv2_plan_and_preprocess -d 112 --verify_dataset_integrity

# Create splits_final.json after plan_and_preprocess, because nnUNet_preprocessed/Dataset112_BreastTumorISPY
# is created during preprocessing.
python scripts/create_ispy_nnunet_splits.py \
  --csv outputs/BreastDCEDL_ISPY1_ISPY2_noDuke_80_10_10_split.csv \
  --case-map outputs/nnunet_ispy_audit/case_id_mapping.csv \
  --dataset-id 112 \
  --dataset-name BreastTumorISPY

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 112 3d_fullres 0 --npz

# internal_test cases are placed in imagesTs and predicted only after training.
nnUNetv2_predict \
  -i $nnUNet_raw/Dataset112_BreastTumorISPY/imagesTs \
  -o outputs/nnunet_predictions/Dataset112_BreastTumorISPY_internal_test \
  -d 112 \
  -c 3d_fullres \
  -f 0
