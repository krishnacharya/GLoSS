import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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