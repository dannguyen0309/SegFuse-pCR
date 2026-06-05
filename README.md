# BreastCancerDetection-MRI Code Bundle

This `code/` folder contains the paper-facing source code for the breast MRI segmentation and multimodal pCR classification pipeline.

## 1. Download the dataset

Download the two cropped I-SPY dataset archives from Zenodo into `data/`:

```bash
mkdir -p data
cd data
wget "https://zenodo.org/records/18114231/files/BreastDCEDL_ISPY1_min_crop.tar.gz?download=1" -O BreastDCEDL_ISPY1_min_crop.tar.gz
wget "https://zenodo.org/records/18114231/files/BreastDCEDL_ISPY2_min_crop.tar.gz?download=1" -O BreastDCEDL_ISPY2_min_crop.tar.gz
```

Extract them:

```bash
tar -xzf BreastDCEDL_ISPY1_min_crop.tar.gz
tar -xzf BreastDCEDL_ISPY2_min_crop.tar.gz
cd ..
```

This code also expects the metadata CSV at `data/BreastDCEDL_metadata_min_crop.csv`.

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r code/requirements.txt
```

Notes:

- Install a CUDA-matched PyTorch build if you plan to train on GPU.
- The MedicalNet source tree is expected under `third_party/MedicalNet`.
- The MedNet34 pretrained weights are expected at `third_party/pretrained/resnet_34_23dataset.pth`.

## 3. Build the split and manifests

Create the train/val/internal-test split:

```bash
./venv/bin/python code/create_ispy_noDuke_split.py
```

Build the GT-mask manifest:

```bash
./venv/bin/python code/scripts/build_phase2_manifest.py \
  --csv outputs/BreastDCEDL_ISPY1_ISPY2_noDuke_80_10_10_split.csv \
  --spy1-root data/BreastDCEDL_ISPY1_min_crop \
  --spy2-root data/BreastDCEDL_ISPY2_min_crop \
  --mask-source gt \
  --out outputs/phase2_manifest_gt_mask.csv
```

Build the predicted-mask manifest:

```bash
./venv/bin/python code/scripts/build_phase2_manifest.py \
  --csv outputs/BreastDCEDL_ISPY1_ISPY2_noDuke_80_10_10_split.csv \
  --spy1-root data/BreastDCEDL_ISPY1_min_crop \
  --spy2-root data/BreastDCEDL_ISPY2_min_crop \
  --mask-source pred_nnunet \
  --pred-mask-root outputs/nnunet_predictions/Dataset112_BreastTumorISPY_internal_test \
  --out outputs/phase2_manifest_pred_mask.csv
```

## 4. Paper results

### Validation-set comparison

| Method | AUROC | AUPRC | Balanced Acc | Recall | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | 
| Image only | 0.65 | 0.50 | 0.63 | 0.55 | 0.70 | 
| Clinical only | 0.74 | 0.57 | 0.70 | 0.67 | 0.74 |
| GT mask ROI + clinical + concat | 0.74 | 0.56 | 0.72 | 0.85 | 0.64 |
| GT mask ROI + clinical + attention | 0.76 | 0.61 | 0.72 | 0.85 | 0.56 |

### Test-set performance with GT and predicted masks

| Metric | GT mask | Predicted mask 
| --- | ---: | ---: |
| AUROC | 0.70 | 0.71 |
| AUPRC | 0.49 | 0.50 |
| Balanced accuracy | 0.64 | 0.63 |
| Recall / Sensitivity | 0.67 | 0.64 |
| Specificity | 0.62 | 0.62 |

These tables use the paper-ready rounded values you provided.

## 5. Reproduction commands

### Validation experiments

To reproduce the four validation experiments reported above, run:

```bash
bash code/scripts/run_stage_2_multimodal_classification.sh
```

This script includes:

- image-only with GT ROI
- clinical-only
- GT ROI + clinical + concat fusion
- GT ROI + clinical + attention fusion

### Internal-test mask comparison

To reproduce the internal-test comparison between GT masks and predicted masks, run:

```bash
bash code/scripts/run_stage_1_nnunet.sh
bash code/scripts/run_stage_2_multimodal_classification.sh
```

`run_stage_1_nnunet.sh` generates the nnU-Net predictions used for the predicted-mask evaluation, and `run_stage_2_multimodal_classification.sh` trains the MedNet34 models and runs the GT-mask and predicted-mask evaluations.
