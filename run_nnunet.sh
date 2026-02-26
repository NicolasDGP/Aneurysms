#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User paths
# -----------------------------
RAW_DIR="/Users/nicolas/Desktop/nnUNet_raw"
PREPRO_DIR="/Users/nicolas/Desktop/nnUNet_preprocessed"
RESULTS_DIR="/Users/nicolas/Desktop/nnUNet_results"

REF_IMAGE="/Users/nicolas/Desktop/ref_image.nii.gz"
MASK_IMAGE="/Users/nicolas/Desktop/mask.nii.gz"

DATASET_ID="000"
DATASET_NAME="brain"
DATASET_FOLDER="Dataset${DATASET_ID}_${DATASET_NAME}"

CONFIG="3d_fullres"
FOLD="0"

# nnUNet expects 4-digit case ids and modality suffix _0000 for images
CASE_ID="0000"
IMAGE_TR_NAME="case_${CASE_ID}_0000.nii.gz"
LABEL_TR_NAME="case_${CASE_ID}.nii.gz"   # label must match case id

# -----------------------------
# Export env vars
# -----------------------------
export nnUNet_raw="$RAW_DIR"
export nnUNet_preprocessed="$PREPRO_DIR"
export nnUNet_results="$RESULTS_DIR"

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

# -----------------------------
# Create folder structure
# -----------------------------
IMAGES_TR="$nnUNet_raw/$DATASET_FOLDER/imagesTr"
LABELS_TR="$nnUNet_raw/$DATASET_FOLDER/labelsTr"
mkdir -p "$IMAGES_TR" "$LABELS_TR"

# -----------------------------
# Copy files into nnUNet naming scheme
# -----------------------------
cp -v "$REF_IMAGE"  "$IMAGES_TR/$IMAGE_TR_NAME"
cp -v "$MASK_IMAGE" "$LABELS_TR/$LABEL_TR_NAME"

# -----------------------------
# Create dataset.json
# (Single-channel input, binary segmentation)
# -----------------------------
DATASET_JSON="$nnUNet_raw/$DATASET_FOLDER/dataset.json"
cat > "$DATASET_JSON" <<EOF
{
  "channel_names": {
    "0": "image"
  },
  "labels": {
    "background": 0,
    "foreground": 1
  },
  "numTraining": 1,
  "file_ending": ".nii.gz"
}
EOF

echo "Wrote: $DATASET_JSON"
cat "$DATASET_JSON"

# -----------------------------
# Verify dataset integrity
# -----------------------------
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

# -----------------------------
# Plan + preprocess
# -----------------------------
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c "$CONFIG"

# -----------------------------
# splits_final.json workaround (single-case training)
# Put the same case in train + val so training doesn't fail
# -----------------------------
PREPRO_DS_DIR="$nnUNet_preprocessed/$DATASET_FOLDER"
mkdir -p "$PREPRO_DS_DIR"

cat > "$PREPRO_DS_DIR/splits_final.json" <<JSON
[
  {
    "train": ["case_${CASE_ID}"],
    "val":   ["case_${CASE_ID}"]
  }
]
JSON

echo "Wrote: $PREPRO_DS_DIR/splits_final.json"
cat "$PREPRO_DS_DIR/splits_final.json"

# -----------------------------
# Train
# -----------------------------
nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD"
