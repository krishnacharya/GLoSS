import os
import torch
from datasets import Dataset, load_from_disk
from typing import List, Dict, Tuple
from unsloth import FastLanguageModel, unsloth_train, is_bfloat16_supported
from src.utils.project_dirs import get_hfdata_dir, get_llama_modelsave_dir
import wandb
from trl import SFTTrainer
from transformers import TrainingArguments
import yaml
import argparse
from transformers.trainer_callback import EarlyStoppingCallback

def add_eos_to_text(example, tokenizer):
    return {"text": example["text"] + tokenizer.eos_token}

def finetune(config):
    model_name = config['finetuning_config']['model_name']
    dataset_name = config['finetuning_config']['dataset_name']
    dataset_split = config['finetuning_config']['dataset_split']
    max_seq_length = config['finetuning_config']['max_seq_length']
    dtype = config['finetuning_config']['dtype']
    load_in_4bit = config['finetuning_config']['load_in_4bit']
    lora_r = config['finetuning_config']['lora_r']
    lora_alpha = config['finetuning_config']['lora_alpha']
    learning_rate = config['finetuning_config']['learning_rate']
    per_device_train_batch_size = config['finetuning_config']['per_device_train_batch_size']
    gradient_accumulation_steps = config['finetuning_config']['gradient_accumulation_steps']
    random_state = config['finetuning_config']['random_state']
    num_train_epochs = config['finetuning_config']['num_train_epochs']
    # max_steps = config['finetuning_config']['max_steps']
    warmup_steps = config['finetuning_config']['warmup_steps']

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    # --- Wandb Setup ---
    project_name = f"{model_name.split('/')[-1]}{config['wandb_config']['project_name_suffix'].format(dataset_name=dataset_name)}"
    # add max_steps, epochs, warmup to run_name
    run_name = config['wandb_config']['run_name_template'].format(
        effective_batch_size=effective_batch_size,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        # max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps
    )
    wandb.init(project=project_name,
        name=run_name,
        config=config['finetuning_config']
    )
    print(f"Wandb run initialized: {wandb.run.name} (Project: {wandb.run.project})")
    print(f"Run URL: {wandb.run.get_url()}")

    # --- Model and Tokenizer Loading ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = random_state,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    if tokenizer.pad_token is None:
        raise ValueError("pad_token is None. Please set pad_token in the tokenizer.")

    tokenizer.truncation_side = "left" # removes tokens from the left
    tokenizer.padding_side = "right" # pad tokens on the right, https://github.com/huggingface/transformers/issues/34842#issuecomment-2490994584

    print(f"Tokenizer Info: padding_side={tokenizer.padding_side}, "
        f"truncation_side={tokenizer.truncation_side}, "
        f"pad_token='{tokenizer.pad_token}', "
        f"eos_token='{tokenizer.eos_token}', "
        f"bos_token='{tokenizer.bos_token}'")

    # Loading dataset
    full_dataset = load_from_disk(os.path.join(get_hfdata_dir(), dataset_name))
    train_dataset = full_dataset['train']
    eval_dataset = full_dataset['validation']

    train_dataset = train_dataset.map(add_eos_to_text, fn_kwargs={"tokenizer": tokenizer})
    eval_dataset = eval_dataset.map(add_eos_to_text, fn_kwargs={"tokenizer": tokenizer})

    # model output directory
    model_output_dir = get_llama_modelsave_dir() / f"{model_name.split('/')[-1]}-ft{dataset_name.split('_')[-1]}_ep{num_train_epochs}_maxseq{max_seq_length}"  # this is a pathlib object
    print(f"Model output directory: {model_output_dir}")

    # training_args = TrainingArguments(
    #     output_dir = str(model_output_dir),
    #     run_name = run_name,
    #     per_device_train_batch_size = per_device_train_batch_size,
    #     gradient_accumulation_steps = gradient_accumulation_steps,
    #     warmup_steps = warmup_steps,
    #     # num_train_epochs = num_train_epochs,
    #     max_steps = max_steps,
    #     learning_rate = learning_rate,
    #     fp16 = not is_bfloat16_supported(),
    #     bf16 = is_bfloat16_supported(),
    #     logging_steps = config['training_arguments']['logging_steps'],
    #     optim = config['training_arguments']['optim'],
    #     weight_decay = config['training_arguments']['weight_decay'],
    #     lr_scheduler_type = config['training_arguments']['lr_scheduler_type'],
    #     seed = random_state,
    #     report_to = config['training_arguments']['report_to'],
    #     evaluation_strategy = config['training_arguments']['evaluation_strategy'],
    #     eval_steps = config['training_arguments']['eval_steps'],
    #     save_strategy = config['training_arguments']['save_strategy'],
    #     metric_for_best_model = config['training_arguments']['metric_for_best_model'],
    #     greater_is_better = config['training_arguments']['greater_is_better'],
    #     save_total_limit = config['training_arguments']['save_total_limit'],
    # )

    # trainer = SFTTrainer(model = model,
    #                     tokenizer = tokenizer,
    #                     train_dataset = train_dataset,
    #                     eval_dataset = eval_dataset, #added eval dataset
    #                     dataset_text_field = "text",
    #                     max_seq_length = max_seq_length,
    #                     dataset_num_proc = 2,
    #                     packing = False, # using False as it's buggy and doesnt count correct number of examples
    #                     args = training_args,
    #                     )
    training_arguments = TrainingArguments(
        output_dir = str(model_output_dir), # Make sure this is defined
        run_name = run_name, # Make sure this is defined
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = warmup_steps,
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        fp16 = not is_bfloat16_supported(), # Make sure is_bfloat16_supported() is defined or handled
        bf16 = is_bfloat16_supported(), # Make sure is_bfloat16_supported() is defined or handled
        logging_steps = config['training_arguments']['logging_steps'], # Assuming this is 1 from your config
        optim = config['training_arguments']['optim'],
        weight_decay = config['training_arguments']['weight_decay'],
        lr_scheduler_type = config['training_arguments']['lr_scheduler_type'],
        seed = random_state,
        report_to = config['training_arguments']['report_to'],
        eval_strategy = config['training_arguments']['eval_strategy'],
        eval_steps = config['training_arguments']['eval_steps'],
        save_strategy = config['training_arguments']['save_strategy'],
        metric_for_best_model = config['training_arguments']['metric_for_best_model'],
        greater_is_better = config['training_arguments']['greater_is_better'],
        save_total_limit = config['training_arguments']['save_total_limit'],
        load_best_model_at_end = config['training_arguments']['load_best_model_at_end'],
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config['early_stopping_config']['patience'], # Or your desired number of epochs
        early_stopping_threshold=config['early_stopping_config']['threshold'], # Or your desired threshold
    )

    trainer = SFTTrainer(
        model = model, # Make sure this is defined
        tokenizer = tokenizer, # Make sure this is defined
        train_dataset = train_dataset, # Make sure this is defined
        eval_dataset = eval_dataset, # Make sure this is defined
        dataset_text_field = "text", # Assuming your text field is named "text"
        max_seq_length = max_seq_length,
        dataset_num_proc = 2, # Or your desired number of processes
        packing = False,
        args = training_arguments,
        callbacks = [early_stopping_callback],
    )
    trainer_stats = unsloth_train(trainer)
    model.save_pretrained(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True, help="Path to the finetuning configuration file")
    args = args.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    finetune(config)

# Llama 3.2-1B, 16 lora, 11M parameters, 0.045 recall@5
# Gemma 3, 1B, 16 lora, 13M parameters
# Qwen 3, 600M, 16 lora, 10M parameters
# SmolLM 160M, 256 lora, 78M parameters

# Llama 3.2 3B, 16 lora, 24M params
