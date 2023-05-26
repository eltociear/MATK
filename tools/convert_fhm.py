import os
import argparse
import pandas as pd
import shutil
from tqdm import tqdm

def copy_folder_with_progress(src_folder, dst_folder):
    total_files = sum([len(files) for _, _, files in os.walk(src_folder)])
    
    with tqdm(total=total_files, unit='file') as pbar:
        for root, dirs, files in os.walk(src_folder):
            # Create corresponding directories in the destination folder
            dst_root = os.path.join(dst_folder, os.path.relpath(root, src_folder))
            os.makedirs(dst_root, exist_ok=True)

            # Copy files to the destination folder
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_root, file)
                shutil.copy2(src_file, dst_file)
                pbar.update(1)

def main(github_dir: str, dataset_dir: str):

    img_dir = img_dir = os.path.join(github_dir, "img")
    img_out_dir = os.path.join(dataset_dir, "fhm", "images")
    os.makedirs(img_out_dir, exist_ok=True)

    copy_folder_with_progress(img_dir, img_out_dir)
    
    train_fp = os.path.join(github_dir, "train.jsonl")
    dev_seen_fp = os.path.join(github_dir, "dev_seen.jsonl")
    dev_unseen_fp = os.path.join(github_dir, "dev_unseen.jsonl")
    test_fp = os.path.join(github_dir, "test_unseen.jsonl")

    train_df = pd.read_json(train_fp, lines=True)
    dev_seen_df = pd.read_json(dev_seen_fp, lines=True)
    dev_unseen_df = pd.read_json(dev_unseen_fp, lines=True)
    test_df = pd.read_json(test_fp, lines=True)

    # create the new original file
    new_train_fp = os.path.join(dataset_dir, "fhm", "annotations", "train.jsonl")
    new_dev_seen_fp = os.path.join(dataset_dir, "fhm", "annotations", "dev_seen.jsonl")
    new_dev_unseen_fp = os.path.join(dataset_dir, "fhm", "annotations", "dev_unseen.jsonl")
    new_test_fp = os.path.join(dataset_dir, "fhm", "annotations", "test.jsonl")
    os.makedirs(os.path.dirname(new_train_fp), exist_ok=True)

    train_df.to_json(new_train_fp, orient="records", lines=True)
    dev_seen_df.to_json(new_dev_seen_fp, orient="records", lines=True)
    dev_unseen_df.to_json(new_dev_unseen_fp, orient="records", lines=True)
    test_df.to_json(new_test_fp, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting FHM's finegrained dataset to specified format")
    parser.add_argument("--github-dir", help="Folder path to the Facebook's Hateful Memes Fine-Grain directory")
    parser.add_argument("--dataset-dir", help="Folder path to the dataset directory")
    args = parser.parse_args()

    main(
        args.github_dir,
        args.dataset_dir
    )