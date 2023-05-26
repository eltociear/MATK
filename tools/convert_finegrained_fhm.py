import os
import argparse
import pandas as pd
import shutil

# binary classification
HATEFULNESS = {
    v:k for k,v in enumerate([
        "not_hateful",
        "hateful"
    ])
}

# 6-class multi-label classification
PROTECTED_CATEGORY = {
    v:k for k,v in enumerate([
        "pc_empty",
        "disability",
        "nationality",
        "race",
        "religion",
        "sex"
    ])
}

# 8-class multi-label classification
PROTECTED_ATTACK = {
    v:k for k,v in enumerate([
        "attack_empty",
        "contempt",
        "dehumanizing",
        "exclusion",
        "inciting_violence",
        "inferiority",
        "mocking",
        "slurs"
    ])
}

def main(github_dir: str, dataset_dir: str):

    # remap intensity and target
    train_fp = os.path.join(github_dir, "data", "annotations", "train.json")
    dev_seen_fp = os.path.join(github_dir, "data", "annotations", "dev_seen.json")
    dev_unseen_fp = os.path.join(github_dir, "data", "annotations", "dev_unseen.json")
    test_fp = os.path.join(github_dir, "data", "annotations", "test.jsonl")

    train_df = pd.read_json(train_fp, lines=True)
    dev_seen_df = pd.read_json(dev_seen_fp, lines=True)
    dev_unseen_df = pd.read_json(dev_unseen_fp, lines=True)
    test_df = pd.read_json(test_fp, lines=True)

    train_df['hate'] = train_df['gold_hate'].apply(lambda x: HATEFULNESS[x[0]])
    train_df['pc'] = train_df['gold_pc'].apply(lambda x: [PROTECTED_CATEGORY[i] for i in x])
    train_df['attack'] = train_df['gold_attack'].apply(lambda x: [PROTECTED_ATTACK[i] for i in x])
    
    dev_seen_df['hate'] = dev_seen_df['gold_hate'].apply(lambda x: HATEFULNESS[x[0]])
    dev_seen_df['pc'] = dev_seen_df['gold_pc'].apply(lambda x: [PROTECTED_CATEGORY[i] for i in x])
    dev_seen_df['attack'] = dev_seen_df['gold_attack'].apply(lambda x: [PROTECTED_ATTACK[i] for i in x])

    dev_unseen_df['hate'] = dev_unseen_df['gold_hate'].apply(lambda x: HATEFULNESS[x[0]])
    dev_unseen_df['pc'] = dev_unseen_df['gold_pc'].apply(lambda x: [PROTECTED_CATEGORY[i] for i in x])
    dev_unseen_df['attack'] = dev_unseen_df['gold_attack'].apply(lambda x: [PROTECTED_ATTACK[i] for i in x])

    # create the new original file
    new_train_fp = os.path.join(dataset_dir, "fhm_finegrained", "annotations", "train.jsonl")
    new_dev_seen_fp = os.path.join(dataset_dir, "fhm_finegrained", "annotations", "dev_seen.jsonl")
    new_dev_unseen_fp = os.path.join(dataset_dir, "fhm_finegrained", "annotations", "dev_unseen.jsonl")
    os.makedirs(os.path.dirname(new_train_fp), exist_ok=True)

    train_df.to_json(new_train_fp, orient="records", lines=True)
    dev_seen_df.to_json(new_dev_seen_fp, orient="records", lines=True)
    dev_unseen_df.to_json(new_dev_unseen_fp, orient="records", lines=True)
    test_df.to_json(new_dev_unseen_fp, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting FHM's finegrained dataset to specified format")
    parser.add_argument("--github-dir", help="Folder path to the Facebook's Hateful Memes Fine-Grain directory")
    parser.add_argument("--dataset-dir", help="Folder path to the dataset directory")
    args = parser.parse_args()

    main(
        args.github_dir,
        args.dataset_dir
    )