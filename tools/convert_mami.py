import os
import shutil
import argparse
import pandas as pd

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

    train_fp = os.path.join(github_dir, "training", "TRAINING", "training.csv")
    test_fp = os.path.join(github_dir, "test", "test", "Test.csv")
    label_fp = os.path.join(github_dir, "test_labels.txt")
    val_fp = os.path.join(github_dir, "trial", "Users", "fersiniel", "Desktop", "MAMI - TO LABEL", "TRIAL DATASET", "trial.csv")
    
    ## copy images
    train_img_dir = os.path.join(github_dir, "training", "TRAINING")
    test_img_dir = os.path.join(github_dir, "test", "test")
    val_img_dir = os.path.join(github_dir, "trial", "Users", "fersiniel", "Desktop", "MAMI - TO LABEL", "TRIAL DATASET")

    img_out_dir = os.path.join(dataset_dir, "mami", "images")
    os.makedirs(img_out_dir, exist_ok=True)

    copy_folder_with_progress(train_img_dir, img_out_dir)
    copy_folder_with_progress(test_img_dir, img_out_dir)
    copy_folder_with_progress(val_img_dir, img_out_dir)

    # derive misogynous, shaming, stereotype, objectification and violence

    train_df = pd.read_csv(train_fp, sep="\t")
    train_df = train_df.rename({
        "file_name": "img",
        "Text Transcription": "text"
    }, axis=1)

    test_df = pd.read_csv(test_fp, sep="\t")
    test_df = test_df.rename({
        "file_name": "img",
        "Text Transcription": "text"
    }, axis=1)

    ## handle the merging of annotations and labels for test
    col_names = ["file_name", "misogynous","shaming","stereotype","objectification","violence"]
    label_df = pd.read_csv(label_fp, sep="\t", names=col_names)

    test_df["misogynous"] = label_df["misogynous"].copy()
    test_df["shaming"] = label_df["shaming"].copy()
    test_df["stereotype"] = label_df["stereotype"].copy()
    test_df["objectification"] = label_df["objectification"].copy()
    test_df["violence"] = label_df["violence"].copy()
    
    val_df = pd.read_csv(val_fp, sep="\t")
    val_df = val_df.rename({
        "file_name": "img",
        "Text Transcription": "text"
    }, axis=1)
    
    # create the new original file
    new_train_fp = os.path.join(dataset_dir, "mami", "annotations", "train.jsonl")
    new_val_fp = os.path.join(dataset_dir, "mami", "annotations", "val.jsonl")
    new_test_fp = os.path.join(dataset_dir, "mami", "annotations", "test.jsonl")
    os.makedirs(os.path.dirname(new_train_fp), exist_ok=True)

    train_df.to_json(new_train_fp, orient="records", lines=True)
    test_df.to_json(new_test_fp, orient="records", lines=True)
    val_df.to_json(new_val_fp, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting FHM's finegrained dataset to specified format")
    parser.add_argument("--github-dir", help="Folder path to the Facebook's Hateful Memes Fine-Grain directory")
    parser.add_argument("--dataset-dir", help="Folder path to the dataset directory")
    args = parser.parse_args()

    main(
        args.github_dir,
        args.dataset_dir
    )