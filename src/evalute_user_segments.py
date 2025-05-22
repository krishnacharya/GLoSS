import pandas as pd
from src.utils.project_dirs import processed_data_dir, get_peruser_metric_dataset_modelname_encoder
#TODO

def get_seqlen_stats(dataset_name, low_seqlen, high_seqlen, user_col='user_id', item_col='item_id'):
    """
    Calculates and prints statistics about user sequence lengths, categorizing users
    into cold-start, regular, and power users based on defined thresholds.
    Also provides statistics on interaction distribution across these user groups.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'beauty').
        low_seqlen (int): The maximum sequence length for cold-start users.
        high_seqlen (int): The maximum sequence length for regular users.
        user_col (str, optional): The name of the user ID column. Defaults to 'user_id'.
        item_col (str, optional): The name of the item ID column. Defaults to 'item_id'.

    Returns:
        tuple: A tuple containing sets of cold-start, regular, and power user IDs.
    """
    df_ui = pd.read_json(str(processed_data_dir(dataset_name) / 'df_ui.json'), orient='records', lines=True)
    usc = df_ui[user_col].value_counts() # user sequence count

    print(f"\n############# Dataset: {dataset_name} #############")
    print(f"User Sequence Length - Mean: {usc.mean():.2f}, Std: {usc.std():.2f}")
    
    # Calculate percentage of users below low_seqlen and above high_seqlen
    pcent_cold_users_raw = (usc <= low_seqlen).sum() / len(usc)
    pcent_power_users_raw = (usc > high_seqlen).sum() / len(usc)
    print(f"Percentage of users <= {low_seqlen}: {pcent_cold_users_raw:.2%}")
    print(f"Percentage of users > {high_seqlen}: {pcent_power_users_raw:.2%}")

    cold_start_users = set(usc[usc <= low_seqlen].index.tolist())
    regular_users = set(usc[(usc > low_seqlen) & (usc <= high_seqlen)].index.tolist())
    power_users = set(usc[usc > high_seqlen].index.tolist())

    print(f"\nTotal number of users: {len(usc)}")
    print(f"Number of cold-start users, pcent usplit: {len(cold_start_users)}, {100*len(cold_start_users)/len(usc):.2f}%")
    print(f"Number of regular users, pcent usplit: {len(regular_users)}, {100*len(regular_users)/len(usc):.2f}%")
    print(f"Number of power users, pcent usplit: {len(power_users)}, {100*len(power_users)/len(usc):.2f}%")
    print(f"Sanity check (user split): {len(usc) == len(cold_start_users) + len(regular_users) + len(power_users)}")

    # Find data split (sum of interactions for users in each split)
    print(f"\nTotal number of interactions: {len(df_ui)}")
    cs_users_itemseqsum = df_ui[df_ui[user_col].isin(cold_start_users)].shape[0]
    rs_users_itemseqsum = df_ui[df_ui[user_col].isin(regular_users)].shape[0]
    ps_users_itemseqsum = df_ui[df_ui[user_col].isin(power_users)].shape[0]
    
    print(f"Cold start user interactions, pcent dsplit: {cs_users_itemseqsum}, {100*cs_users_itemseqsum/len(df_ui):.2f}%")
    print(f"Regular user interactions, pcent dsplit: {rs_users_itemseqsum}, {100*rs_users_itemseqsum/len(df_ui):.2f}%")
    print(f"Power user interactions, pcent dsplit: {ps_users_itemseqsum}, {100*ps_users_itemseqsum/len(df_ui):.2f}%")
    print(f"Sanity check (interaction split): {cs_users_itemseqsum + rs_users_itemseqsum + ps_users_itemseqsum == len(df_ui)}")

    return cold_start_users, regular_users, power_users

def analyze_user_group_metrics(dataset, model_name, encoder_name, cold_users_set, regular_users_set, power_users_set, metrics_cols=['recall@5', 'ndcg@5']):
    """
    Analyzes and prints the average performance metrics for different user groups (cold-start, regular, power).
    Also prints overall average metrics for the dataset and model.

    Args:
        dataset (str): The name of the dataset (e.g., 'beauty').
        model_name (str): The name of the model (e.g., 'llama-1b').
        encoder_name (str): The name of the encoder (e.g., 'sentence-transformers/all-MiniLM-L6-v2').
        cold_users_set (set): A set of user IDs categorized as cold-start.
        regular_users_set (set): A set of user IDs categorized as regular.
        power_users_set (set): A set of user IDs categorized as power.
        metrics_cols (list): A list of metric column names to analyze (e.g., ['recall@5', 'ndcg@5']).
    """
    print(f"\n--- Analyzing Metrics for {dataset} with {model_name} and {encoder_name} ---")
    dir_path = get_peruser_metric_dataset_modelname_encoder(dataset=dataset, model_name=model_name, encoder_name=encoder_name)
    
    # Construct the file path using the dataset and model name
    file_name = f"{dataset}_test_{model_name}.jsonl"
    file_path = str(dir_path / file_name)

    try:
        pum_df = pd.read_json(file_path, orient='records', lines=True)
    except FileNotFoundError:
        print(f"Error: Metric file not found at {file_path}. Please ensure the file exists.")
        return

    # Ensure 'reviewer_id' exists in the DataFrame before filtering
    if 'reviewer_id' not in pum_df.columns:
        print(f"Error: 'reviewer_id' column not found in the metrics file for {dataset}. Please check the data format.")
        return
    
    # --- Overall Metrics ---
    print(f"\nOverall Average Metrics for {dataset} - {model_name} ({len(pum_df)} users):")
    if not pum_df.empty:
        print(pum_df[metrics_cols].mean())
    else:
        print("No metrics data found in the file.")

    # --- Metrics by User Group ---
    cold_start_metrics_df = pum_df[pum_df['reviewer_id'].isin(cold_users_set)]
    regular_metrics_df = pum_df[pum_df['reviewer_id'].isin(regular_users_set)]
    power_metrics_df = pum_df[pum_df['reviewer_id'].isin(power_users_set)]

    print(f"\nAverage metrics for Cold-Start Users ({len(cold_start_metrics_df)} users):")
    if not cold_start_metrics_df.empty:
        print(cold_start_metrics_df[metrics_cols].mean())
    else:
        print("No cold-start users with metrics found.")

    print(f"\nAverage metrics for Regular Users ({len(regular_metrics_df)} users):")
    if not regular_metrics_df.empty:
        print(regular_metrics_df[metrics_cols].mean())
    else:
        print("No regular users with metrics found.")

    print(f"\nAverage metrics for Power Users ({len(power_metrics_df)} users):")
    if not power_metrics_df.empty:
        print(power_metrics_df[metrics_cols].mean())
    else:
        print("No power users with metrics found.")

if __name__ == "__main__":
    # Define dataset configurations
    datasets_config = {
        'beauty': {'low_seqlen': 5, 'high_seqlen': 14, 'user_col': 'reviewerID', 'item_col': 'asin'},
        'toys': {'low_seqlen': 5, 'high_seqlen': 13, 'user_col': 'reviewerID', 'item_col': 'asin'},
        'sports': {'low_seqlen': 5, 'high_seqlen': 13, 'user_col': 'reviewerID', 'item_col': 'asin'},
    }

    user_groups = {} # To store the user sets for each dataset

    # Process each dataset for sequence length statistics
    for ds_name, config in datasets_config.items():
        cold, regular, power = get_seqlen_stats(
            dataset_name=ds_name,
            low_seqlen=config['low_seqlen'],
            high_seqlen=config['high_seqlen'],
            user_col=config['user_col'],
            item_col=config['item_col']
        )
        user_groups[ds_name] = {'cold': cold, 'regular': regular, 'power': power}

    # Define models to evaluate
    models_to_evaluate = ['llama-1b', 'llama-3b', 'llama-8b']
    
    cols_to_analyze = ['recall@5', 'ndcg@5']
    # encoder_name = 'sentence-transformers/all-MiniLM-L6-v2'
    # encoder_name = 'bm25s'
    # encoder_name = 'intfloat/e5-small-v2'
    encoder_name = 'intfloat/e5-base-v2'
    # Analyze metrics for each specified dataset and model
    for target_dataset in datasets_config.keys(): # Loop through all defined datasets
        if target_dataset in user_groups:
            for model_name in models_to_evaluate:
                analyze_user_group_metrics(
                    dataset=target_dataset,
                    model_name=model_name,
                    encoder_name=encoder_name,
                    cold_users_set=user_groups[target_dataset]['cold'],
                    regular_users_set=user_groups[target_dataset]['regular'],
                    power_users_set=user_groups[target_dataset]['power'],
                    metrics_cols=cols_to_analyze
                )
        else:
            print(f"Warning: Sequence length stats not computed for {target_dataset}. Skipping metric analysis.")