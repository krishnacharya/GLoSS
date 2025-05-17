#!/bin/bash

PYTHON_SCRIPT="/home/kacharya33/nanoGenRec/src/calcirmetrics_dense-minilm_dataset_agno.py"
CUDA_VISIBLE_DEVICES="0"

# Default parameters
DEFAULT_NUM_SEQUENCES="5"
DEFAULT_ENCODER_NAME="sentence-transformers/all-MiniLM-L6-v2"

# Function to run a job
run_job() {
  local dataset_name="$1"
  local data_family="$2"
  local generated_file="$3"
  local num_sequences="$4"
  local split="$5"
  local encoder_name="$6"
  local short_model_name="$7"

  echo "Running job for dataset: $dataset_name, data family: $data_family, model: $short_model_name, split: $split"
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python "$PYTHON_SCRIPT" \
    --dataset_name "$dataset_name" \
    --data_family "$data_family" \
    --generated_file "$generated_file" \
    --num_sequences "$num_sequences" \
    --split "$split" \
    --encoder_name "$encoder_name" \
    --short_model_name "$short_model_name"
  if [ $? -ne 0 ]; then
    echo "Error running job for dataset: $dataset_name, model: $short_model_name, split: $split" >&2
    return 1 # Return non-zero exit code on error
  fi
  return 0
}


## Jobs for ml100k dataset (Movielens family)
# run_job "ml100k" "movielens" "llama-1b/llama-1b-checkpoint-649_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-1b" || exit 1
# run_job "ml100k" "movielens" "llama-3b/llama-3b-checkpoint-590_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-3b" || exit 1
# run_job "ml100k" "movielens" "llama-8b/llama-8b-checkpoint-708_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-8b" || exit 1


## Jobs for Amazon family datasets

### Beauty dataset
run_job "beauty" "amazon" "llama-1b/llama-1b-test_beam5_max_seq1024.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-1b" || exit 1
run_job "beauty" "amazon" "llama-3b/llama-3b-test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-3b" || exit 1
run_job "beauty" "amazon" "llama-8b/llama-8b-8.3k_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-8b" || exit 1

### Toys dataset
run_job "toys" "amazon" "llama-1b/Llama-3.2-1B-fttoys_ep10_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-1b" || exit 1
run_job "toys" "amazon" "llama-3b/Llama-3.2-3B-fttoys_ep10_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-3b" || exit 1
run_job "toys" "amazon" "llama-8b/llama-8b-toys-7.2kcpt_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-8b" || exit 1

### Sports dataset
run_job "sports" "amazon" "llama-1b/Llama-3.2-1B-ftsports_ep7_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-1b" || exit 1
run_job "sports" "amazon" "llama-3b/Llama-3.2-3B-ftsports_ep7_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-3b" || exit 1
run_job "sports" "amazon" "llama-8b/Llama-3.1-8B-ftsports_ep7_maxseq1024_bs4_acc4_test_beam5_max_seq1024_bs8_numret5.json" "$DEFAULT_NUM_SEQUENCES" "test" "$DEFAULT_ENCODER_NAME" "llama-8b" || exit 1