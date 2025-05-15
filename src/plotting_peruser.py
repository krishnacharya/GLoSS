import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from src.utils.project_dirs import *
from src.calcirmetrics import load_data, verify_reviewer_ids, evaluate_retrieval

def plot_metric_vs_sequence_length(metric_per_reviewer, reviewer_sequence_counts, max_seq_length=None, min_reviewers=5, yaxis_name = "Metric", xaxis_name = "Sequence Length"):
    """
    Plots the mean metric against sequence length with standard error bars.

    Args:
        metric_per_reviewer (dict): Dictionary where keys are reviewer IDs and values are their metrics.
        reviewer_sequence_counts (dict): Dictionary where keys are reviewer IDs and values are their sequence lengths.
        max_seq_length (int, optional): The maximum sequence length to consider in the plot. Defaults to None (all lengths).
        min_reviewers (int, optional): The minimum number of reviewers required to calculate mean and SEM for a sequence length. Defaults to 5, just to avoid noise
    """
    metric_by_seqlen = defaultdict(list)
    for rid, metric in metric_per_reviewer.items(): # reviewer ID, metric value
        seq_len = reviewer_sequence_counts.get(rid) # get the item interaction length for the reviewer
        if seq_len is not None and (seq_len <= max_seq_length):
            metric_by_seqlen[seq_len].append(metric)

    metric_summary = {}
    for seq_len, metrics in metric_by_seqlen.items():
        if len(metrics) >= min_reviewers:
            mean_metric = np.mean(metrics)
            sem = np.std(metrics) / np.sqrt(len(metrics)) if len(metrics) > 1 else 0
            metric_summary[seq_len] = {'mean': mean_metric, 'sem': sem}

    if not metric_summary:
        print(f"No sequence lengths with at least {min_reviewers} reviewers within the specified maximum sequence length.")
        return

    sequence_lengths = sorted(metric_summary.keys())
    means = [metric_summary[length]['mean'] for length in sequence_lengths]
    sems_upper = [metric_summary[length]['mean'] + metric_summary[length]['sem'] for length in sequence_lengths]
    sems_lower = [metric_summary[length]['mean'] - metric_summary[length]['sem'] for length in sequence_lengths]

    plt.figure(figsize=(10, 6))
    plt.plot(sequence_lengths, means, marker='o', linestyle='-')
    plt.fill_between(sequence_lengths, sems_lower, sems_upper, color='lightblue', alpha=0.4)

    plt.xlabel(xaxis_name)
    plt.ylabel(yaxis_name)
    plt.xticks(sequence_lengths)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_metric_summary(metric_per_reviewer, reviewer_sequence_counts, max_seq_length=None, min_reviewers=1):
    """
    Calculates the mean metric and SEM for each sequence length.  This function is now a helper.

    Args:
        metric_per_reviewer (dict): Dictionary where keys are reviewer IDs and values are their metrics.
        reviewer_sequence_counts (dict): Dictionary where keys are reviewer IDs and values are their sequence lengths.
        max_seq_length (int, optional): The maximum sequence length to consider in the plot. Defaults to None (all lengths).
        min_reviewers (int, optional): The minimum number of reviewers required to calculate mean and SEM for a sequence length. Defaults to 1.

    Returns:
        dict: A dictionary where keys are sequence lengths and values are dictionaries
              containing 'mean' and 'sem' of the metric for that sequence length.  Returns
              an empty dict if no data meets the criteria.
    """
    metric_by_seqlen = defaultdict(list)
    for rid, metric in metric_per_reviewer.items():  # reviewer ID, metric value
        seq_len = reviewer_sequence_counts.get(rid)  # get the item interaction length for the reviewer
        if seq_len is not None and (seq_len <= max_seq_length):
            metric_by_seqlen[seq_len].append(metric)

    metric_summary = {}
    for seq_len, metrics in metric_by_seqlen.items():
        if len(metrics) >= min_reviewers:
            mean_metric = np.mean(metrics)
            sem = np.std(metrics) / np.sqrt(len(metrics)) if len(metrics) > 1 else 0
            metric_summary[seq_len] = {'mean': mean_metric, 'sem': sem}

    return metric_summary

def plot_across_modelsizes(genfiles, model_names, category, retriever_index, num_sequences,
                            at_k, df_ui, yaxis_name="recall@5", xaxis_name="Sequence Length", max_seq_length=50):
    """
    Plots the mean retrieval metric against sequence length for different generated files from different model sizes on the same plot.

    Args:
        genfiles (list of str): List of filepaths to the generated data.
        category (str): The category of the dataset.
        retriever_index (str): Name of the BM25 index file.
        num_sequences (int): Number of sequences to retrieve.
        at_k (int): The value of k for @k metrics.
        df_ui (pd.DataFrame): DataFrame containing user-item interaction data with 'reviewerID'.
        yaxis_name (str, optional): Label for the y-axis. Defaults to "Recall@5".
        xaxis_name (str, optional): Label for the x-axis. Defaults to "Sequence Length".
        max_seq_length (int, optional):  Maximum sequence length to consider.
    """
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'red']  # TODO hardcoded but should infer from number of genfiles
    meta_filepath = str(processed_data_dir(f'{category}2014') / 'meta_corpus.json')
    retriever_filepath = str(get_bm25_indexes_dir() / retriever_index)

    for i, generated_filepath in enumerate(genfiles):
        asins_compact, genop = load_data(meta_filepath, generated_filepath)
        verify_reviewer_ids(genop)

        qrels, rundR, ans = evaluate_retrieval(genop, retriever_filepath, num_sequences, asins_compact, at_k)
        metric_perreviewer = rundR.scores[yaxis_name]
        reviewer_counts = df_ui['reviewerID'].value_counts().to_dict()

        # Calculate metric summary using the helper function
        metric_summary = get_metric_summary(metric_perreviewer, reviewer_counts, max_seq_length=max_seq_length)

        if not metric_summary:
            print(f"No data for {model_names[i]} meets the criteria (min_reviewers, max_seq_length). Skipping.")
            continue  # Skip to the next model

        sequence_lengths = sorted(metric_summary.keys())
        means = [metric_summary[length]['mean'] for length in sequence_lengths]
        sems_upper = [metric_summary[length]['mean'] + metric_summary[length]['sem'] for length in sequence_lengths]
        sems_lower = [metric_summary[length]['mean'] - metric_summary[length]['sem'] for length in sequence_lengths]
        
        # Plotting
        plt.plot(sequence_lengths, means, marker='o', linestyle='-', color=colors[i], label=model_names[i])
        # plt.fill_between(sequence_lengths, sems_lower, sems_upper, color=colors[i], alpha=0.2)

    plt.xlabel(xaxis_name)
    plt.ylabel(yaxis_name.capitalize())
    # plt.title(f'{yaxis_name} vs. Sequence Length for Different Models')
    plt.xticks(sequence_lengths)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    category = 'beauty'
    generated_file1b = 'llama-1b/llama-1b-test_beam5_max_seq1024.json'
    generated_file3b = 'llama-3b/llama-3b-test_beam5_max_seq1024_bs8_numret5.json'
    generated_file8b = 'llama-8b/llama-8b-8.3k_test_beam5_max_seq1024_bs8_numret5.json'
    model_names = ['Llama-1B', 'Llama-3B', 'Llama-8B']
    retriever_index = 'amznbeauty2014_index'
    num_sequences = 5
    at_k = 5

    generated_filepaths = [str(get_gen_dir_dataset(category) / generated_file) for generated_file in [generated_file1b, generated_file3b, generated_file8b]]
    meta_filepath = str(processed_data_dir(f'{category}2014') / 'meta_corpus.json')
    retriever_filepath = str(get_bm25_indexes_dir() / retriever_index)

    df_ui = pd.read_json(str(processed_data_dir(f'{category}2014') / 'df_dedup.json'), orient='records', lines=True)
    plot_across_modelsizes(generated_filepaths, model_names, category, retriever_index, num_sequences, at_k,\
                            df_ui, yaxis_name="recall@5", xaxis_name="Sequence Length", max_seq_length = 50)
