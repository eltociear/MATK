import os
import shutil
import argparse
import pandas as pd

def main(
        annotations_dir: str,
        train_filename: str,
        test_filename: str
    ):

    # train dataframe
    train_fp = os.path.join(annotations_dir, train_filename)
    train_df = pd.read_csv(train_fp, sep="\t")
    train_df = train_df.rename({
        "file_name": "img",
        "Text Transcription": "text"
    }, axis=1)

    # derive intensity and target
    test_fp = os.path.join(annotations_dir, test_filename)
    test_df = pd.read_csv(test_fp, sep="\t", names=["misogynous", "shaming", "stereotype", "objectification", "violence"])
    test_df = test_df.reset_index().rename({"index": "img"}, axis=1)

    # shift the original file
    archive_dir = os.path.join(annotations_dir, "original")
    os.makedirs(archive_dir, exist_ok=True)

    archived_train_fp = os.path.join(archive_dir, train_filename)
    shutil.move(train_fp, archived_train_fp)

    archived_test_fp = os.path.join(archive_dir, test_filename)
    shutil.move(test_fp, archived_test_fp)
    
    # create the new original file
    new_train_fp = os.path.join(annotations_dir, "train.jsonl")
    new_test_fp = os.path.join(annotations_dir, "test.jsonl")
    train_df.to_json(new_train_fp, orient="records", lines=True)
    test_df.to_json(new_test_fp, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting MAMI dataset to specified format")
    parser.add_argument("--annotation-dir", help="path to the annotation directory")
    parser.add_argument("--train-filename", default="training.csv", help="filepath to the HarMemes train annotations")
    parser.add_argument("--test-filename", default="test_labels.txt", help="filepath to the HarMemes test annotations")
    args = parser.parse_args()

    main(args.annotation_dir, args.train_filename, args.test_filename)