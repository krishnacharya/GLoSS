import random
from collections import defaultdict
from typing import List, Dict
import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path
import argparse
from src.utils.project_dirs import processed_data_dir, get_hfdata_dir

def prepare_metadata(meta_corpus_path: str) -> pd.DataFrame:
    """
    Loads and prepares the metadata corpus.

    Args:
        meta_corpus_path (str): Path to the metadata corpus JSON file.

    Returns:
        pd.DataFrame: A DataFrame with 'asin' and 'Title' columns.
    """
    meta_corpus = pd.read_json(meta_corpus_path, orient='records', lines=True)
    meta_corpus.columns = ['asin', 'Title']
    return meta_corpus


def prepare_reviewer_asins(interaction_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Groups ASINs by reviewer ID from the interaction DataFrame.

    Args:
        interaction_df (pd.DataFrame): DataFrame containing reviewer-ASIN interactions
                                        with 'reviewerID' and 'asin' columns.

    Returns:
        Dict[str, List[str]]: A dictionary mapping reviewer IDs to lists of ASINs.
    """
    reviewer_asins = defaultdict(list)
    for _, row in interaction_df.iterrows():
        reviewer_asins[row['reviewerID']].append(row['asin'])
    return reviewer_asins


def create_datasets(
    interaction_df: pd.DataFrame,
    meta_corpus: pd.DataFrame,
    save_path: str,
    category: str,
    Nval: int = 512,
    random_seed: int = 42,
) -> DatasetDict:
    """
    Creates train, validation, and test datasets.

    Args:
        interaction_df (pd.DataFrame): 'reviewerID' and 'asin' interaction dataframe.
        meta_corpus (pd.DataFrame): metadata corpus dataframe with 'asin', 'Title'.
        save_path (str or Path): Path to save the resulting Hugging Face datasets.
        category (str): The name of the dataset category (e.g., 'beauty', 'toys').
                        Used for naming the saved dataset files.
        Nval (int): Number of reviewers to include in the validation set.
        random_seed (int): Seed for random sampling.

    Returns:
        DatasetDict: A Hugging Face DatasetDict containing train, validation, and test splits.
    """
    # Prepare metadata
    meta_corpus.columns = ['asin', 'Title']
    asins_compact = meta_corpus[['asin']].copy()
    asins_compact['nlang'] = "Title: " + meta_corpus['Title']
    asin_dict = asins_compact.set_index('asin')['nlang'].to_dict() # asin to serialized natural language string

    # Group ASINs by reviewer
    reviewer_asins = prepare_reviewer_asins(interaction_df)

    # Define prompt template
    amzn_prompt = (
        "Below is a customer's purchase history on Amazon, listed in chronological order (earliest to latest). \n"
        "Each item is represented by the following format: Title: <item title> \n"
        "Based on this history, predict **only one** item the customer is most likely to purchase next in the same format.\n\n"
        "### Purchase history:\n"
        "{}\n\n"
        "### Next item:\n"
        "{}"
    )

    # Split reviewers into validation and others
    random.seed(random_seed)
    all_reviewers = list(reviewer_asins.keys())
    Nval = min(Nval, len(all_reviewers) // 2) # Ensure Nval is not too large
    val_rewid = set(random.sample(all_reviewers, Nval))

    # Prepare records for each split
    train_records, val_records, test_records = [], [], []
    for reviewer_id, asins_list in reviewer_asins.items():
        n = len(asins_list)
        if n < 3:
            continue  # Skip reviewers with less than 3 interactions
        formatted_asins_list = [asin_dict.get(asin, f"Unknown ASIN: {asin}") for asin in asins_list]

        # Prepare test set for all reviewers
        test_ptext = amzn_prompt.format("\n".join(formatted_asins_list[:n-1]), "")
        test_text = amzn_prompt.format("\n".join(formatted_asins_list[:n-1]), formatted_asins_list[n-1])
        test_seen_asins = asins_list[:n-1]
        test_asin = asins_list[n-1]
        test_asin_text = formatted_asins_list[n-1]
        test_records.append({
            "reviewer_id": reviewer_id,
            "ptext": test_ptext,
            "text": test_text,
            "seen_asins": test_seen_asins,
            "asin": test_asin,
            "asin_text": test_asin_text
        })

        if reviewer_id in val_rewid:  # Validation set
            val_ptext = amzn_prompt.format("\n".join(formatted_asins_list[:n-2]), "")
            val_text = amzn_prompt.format("\n".join(formatted_asins_list[:n-2]), formatted_asins_list[n-2])
            val_seen_asins = asins_list[:n-2]
            val_asin = asins_list[n-2]
            val_asin_text = formatted_asins_list[n-2]
            val_records.append({
                "reviewer_id": reviewer_id,
                "ptext": val_ptext,
                "text": val_text,
                "seen_asins": val_seen_asins,
                "asin": val_asin,
                "asin_text": val_asin_text
            })
            # Train set is shorter for validation users
            train_text = amzn_prompt.format("\n".join(formatted_asins_list[:n-3]), formatted_asins_list[n-3])
            train_records.append({"reviewer_id": reviewer_id, "text": train_text})
        else:  # Train set for non-validation users
            train_text = amzn_prompt.format("\n".join(formatted_asins_list[:n-2]), formatted_asins_list[n-2])
            train_records.append({"reviewer_id": reviewer_id, "text": train_text})

    # Convert lists to Hugging Face datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_records))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_records))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_records))

    # Create a DatasetDict
    dataset_dict = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    # Save dataset locally
    output_path = Path(save_path) / category.lower()
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    print(f"Hugging Face datasets saved to {output_path}")

    return dataset_dict

def main():
    """Main function to orchestrate dataset creation."""
    parser = argparse.ArgumentParser(description="Create Hugging Face dataset for recommendation.")
    parser.add_argument("--category", type=str, required=True, help="The category of the dataset to process (e.g., 'beauty', 'toys').")
    args = parser.parse_args()
    category = args.category.lower()

    processed_dir = processed_data_dir(f"{category}")
    hf_dir = get_hfdata_dir()

    core5_file = str(processed_dir / 'df_withdup.json')
    meta_corpus_file = str(processed_dir / 'meta_corpus.json')

    try:
        core5 = pd.read_json(core5_file, lines=True)
    except FileNotFoundError:
        print(f"Error: Interaction file not found at {core5_file}. Make sure to run the review processing script first.")
        return

    try:
        meta_corpus = prepare_metadata(meta_corpus_file)
    except FileNotFoundError:
        print(f"Error: Metadata corpus file not found at {meta_corpus_file}. Make sure to run the metadata processing script first.")
        return

    # Calculate Nval (5% of unique reviewers)
    Nval = int(0.05 * core5['reviewerID'].nunique())

    # Create datasets
    create_datasets(
        core5, meta_corpus, save_path=str(hf_dir), category=category, Nval=Nval, random_seed=42
    )

if __name__ == "__main__":
    main()