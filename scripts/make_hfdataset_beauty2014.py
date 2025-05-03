import random
from collections import defaultdict
from typing import List, Dict
import pandas as pd
from datasets import Dataset, DatasetDict

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


def prepare_reviewer_asins(core5: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Groups ASINs by reviewer ID from the core interaction DataFrame.

    Args:
        core5 (pd.DataFrame): DataFrame containing reviewer-ASIN interactions.

    Returns:
        Dict[str, List[str]]: A dictionary mapping reviewer IDs to lists of ASINs.
    """
    reviewer_asins = defaultdict(list)
    for _, row in core5.iterrows():
        reviewer_asins[row['reviewerID']].append(row['asin'])
    return reviewer_asins


def create_datasets(
    core5: pd.DataFrame,
    meta_corpus: pd.DataFrame,
    save_path: str,
    Nval: int = 512,
    random_seed: int = 42,
) -> DatasetDict:
    """
    Creates train, validation, and test datasets.

    Args:
        core5 (pd.DataFrame): 'reviewerID' and 'asin' interaction dataframe.
        meta_corpus (pd.DataFrame): metadata corpus dataframe with 'asin', 'Title'
        save_path (str or Path): Path to save the resulting Hugging Face datasets.
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
    reviewer_asins = prepare_reviewer_asins(core5)

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
    val_rewid = set(random.sample(list(reviewer_asins.keys()), Nval))

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
    dataset_dict.save_to_disk(str(save_path))

    return dataset_dict

def main():
    """Main function to orchestrate dataset creation."""
    from src.utils.project_dirs import processed_data_dir, get_hfdata_dir

    core5_file = str(processed_data_dir('beauty2014') / 'df_withdup.json')
    meta_corpus_file = str(processed_data_dir('beauty2014') / 'meta_corpus.json')

    core5 = pd.read_json(core5_file, lines=True)
    meta_corpus = pd.read_json(meta_corpus_file, orient='records', lines=True)

    # Calculate Nval (5% of unique reviewers)
    Nval = int(0.05 * core5['reviewerID'].nunique())
    hf_dir = get_hfdata_dir()

    # Create datasets with duplicates
    dataset_withdup = create_datasets(
        core5, meta_corpus, save_path=str(hf_dir / 'beauty2014'), Nval=Nval, random_seed=42
    )
if __name__ == "__main__":
    main()