#!/bin/bash

PYTHON_SCRIPT="/home/kacharya33/nanoGenRec/src/eval_ssearch_lasttext.py"

# Default parameters
DEFAULT_K="5"
DEFAULT_SPLIT="test"
DEFAULT_ENCODER_NAME="intfloat/e5-small-v2"

# Function to run a job
run_job() {
  local dataset_name="$1"
  local data_family="$2"
  local encoder_name="$3"
  local k="$4"
  local split="$5"

  echo "Running job for dataset: $dataset_name, data family: $data_family, encoder: $encoder_name, k: $k, split: $split"
  CUDA_VISIBLE_DEVICES=0 python "$PYTHON_SCRIPT" \
    --dataset_name "$dataset_name" \
    --data_family "$data_family" \
    --encoder_name "$encoder_name" \
    --k "$k" \
    --split "$split"
  if [ $? -ne 0 ]; then
    echo "Error running job for dataset: $dataset_name, encoder: $encoder_name, split: $split" >&2
    return 1 # Return non-zero exit code on error
  fi
  return 0
}

# ---
## Jobs for Amazon family datasets

### Beauty dataset
run_job "beauty" "amazon" "$DEFAULT_ENCODER_NAME" "$DEFAULT_K" "$DEFAULT_SPLIT" || exit 1

### Toys dataset
run_job "toys" "amazon" "$DEFAULT_ENCODER_NAME" "$DEFAULT_K" "$DEFAULT_SPLIT" || exit 1

### Sports dataset
run_job "sports" "amazon" "$DEFAULT_ENCODER_NAME" "$DEFAULT_K" "$DEFAULT_SPLIT" || exit 1