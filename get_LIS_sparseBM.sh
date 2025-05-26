#!/bin/bash

PYTHON_SCRIPT="/home/kacharya33/nanoGenRec/src/eval_spsearch_lasttext.py"

# Default parameters
DEFAULT_K="5"
DEFAULT_SPLIT="test"

# Function to run a job
run_job() {
  local dataset_name="$1"
  local data_family="$2"
  local k="$3"
  local split="$4"

  # Construct bm25_index_name from dataset_name
  local bm25_index_name="${dataset_name}_index"

  echo "Running job for dataset: $dataset_name, data family: $data_family, k: $k, split: $split, bm25_index_name: $bm25_index_name"
  CUDA_VISIBLE_DEVICES=0 python "$PYTHON_SCRIPT" \
    --dataset "$dataset_name" \
    --data_family "$data_family" \
    --k "$k" \
    --split "$split" \
    --bm25_index_name "$bm25_index_name"
  if [ $? -ne 0 ]; then
    echo "Error running job for dataset: $dataset_name, data_family: $data_family, split: $split" >&2
    return 1 # Return non-zero exit code on error
  fi
  return 0
}

# ---
## Jobs for Amazon family datasets

### Beauty dataset
run_job "beauty" "amazon" "$DEFAULT_K" "$DEFAULT_SPLIT" || exit 1

### Toys dataset
run_job "toys" "amazon" "$DEFAULT_K" "$DEFAULT_SPLIT" || exit 1

### Sports dataset
run_job "sports" "amazon" "$DEFAULT_K" "$DEFAULT_SPLIT" || exit 1