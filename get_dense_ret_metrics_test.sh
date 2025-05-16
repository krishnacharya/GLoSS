#!/bin/bash

PYTHON_SCRIPT="/home/kacharya33/nanoGenRec/src/calcirmetrics_dense-minilm.py"
CUDA_VISIBLE_DEVICES="0"

# Function to run a job
run_job() {
  local category="$1"
  local generated_file="$2"
  local split="$3"
  local short_model_name="$4"
  echo "Running job for category: $category, model: $short_model_name, split: $split"
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python "$PYTHON_SCRIPT" \
    --category "$category" \
    --generated_file "$generated_file" \
    --split "$split" \
    --short_model_name "$short_model_name"
  if [ $? -ne 0 ]; then
    echo "Error running job for category: $category, model: $short_model_name, split: $split" >&2
    return 1 # Return non-zero exit code on error
  fi
  return 0
}

# Jobs for beauty category
run_job "beauty" "llama-1b/llama-1b-test_beam5_max_seq1024.json" "test" "llama-1b" || exit 1
run_job "beauty" "llama-3b/llama-3b-test_beam5_max_seq1024_bs8_numret5.json" "test" "llama-3b" || exit 1
run_job "beauty" "llama-8b/llama-8b-8.3k_test_beam5_max_seq1024_bs8_numret5.json" "test" "llama-8b" || exit 1

# Jobs for toys category
run_job "toys" "llama-1b/Llama-3.2-1B-fttoys_ep10_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "test" "llama-1b" || exit 1
run_job "toys" "llama-3b/Llama-3.2-3B-fttoys_ep10_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "test" "llama-3b" || exit 1
run_job "toys" "llama-8b/llama-8b-toys-7.2kcpt_test_beam5_max_seq1024_bs8_numret5.json" "test" "llama-8b" || exit 1

# Jobs for sports category
run_job "sports" "llama-1b/Llama-3.2-1B-ftsports_ep7_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "test" "llama-1b" || exit 1
run_job "sports" "llama-3b/Llama-3.2-3B-ftsports_ep7_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "test" "llama-3b" || exit 1
run_job "sports" "llama-8b/Llama-3.1-8B-ftsports_ep7_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "test" "llama-8b" || exit 1
