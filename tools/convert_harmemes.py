import os
import shutil
import argparse
import pandas as pd

INTENSITY_MAP = {
    'not harmful': 0, 
    'somewhat harmful': 1, 
    'very harmful': 2
}

TARGET_MAP = {
    'individual': 0, 
    'organization': 1, 
    'community': 2 , 
    'society': 3
}

def main(
        annotations_dir: str,
        train_filename: str,
        val_filename: str,
        test_filename: str
    ):

    # remap intensity and target
    train_fp = os.path.join(annotations_dir, train_filename)
    val_fp = os.path.join(annotations_dir, val_filename)
    test_fp = os.path.join(annotations_dir, test_filename)

    train_df = pd.read_json(train_fp, lines=True)
    val_df = pd.read_json(val_fp, lines=True)
    test_df = pd.read_json(test_fp, lines=True)

    train_df['intensity'] = train_df['labels'].apply(lambda x: INTENSITY_MAP[x[0]])
    train_df['target'] = train_df['labels'].apply(lambda x: TARGET_MAP[x[1]] if len(x) > 1 else -1)
    train_df = train_df.rename({"image": "img"}, axis=1)

    val_df['intensity'] = val_df['labels'].apply(lambda x: INTENSITY_MAP[x[0]])
    val_df['target'] = val_df['labels'].apply(lambda x: TARGET_MAP[x[1]] if len(x) > 1 else -1)
    val_df = val_df.rename({"image": "img"}, axis=1)
    
    test_df['intensity'] = test_df['labels'].apply(lambda x: INTENSITY_MAP[x[0]])
    test_df['target'] = test_df['labels'].apply(lambda x: TARGET_MAP[x[1]] if len(x) > 1 else -1)
    test_df = test_df.rename({"image": "img"}, axis=1)

    # shift the original file
    archive_dir = os.path.join(annotations_dir, "original")
    os.makedirs(archive_dir, exist_ok=True)

    archived_train_fp = os.path.join(archive_dir, train_filename)
    shutil.move(train_fp, archived_train_fp)

    archived_val_fp = os.path.join(archive_dir, val_filename)
    shutil.move(val_fp, archived_val_fp)

    archived_test_fp = os.path.join(archive_dir, test_filename)
    shutil.move(test_fp, archived_test_fp)
    
    # create the new original file
    new_train_fp = os.path.join(annotations_dir, "train.jsonl")
    new_val_fp = os.path.join(annotations_dir, "val.jsonl")
    new_test_fp = os.path.join(annotations_dir, "test.jsonl")
    train_df.to_json(new_train_fp, orient="records", lines=True)
    val_df.to_json(new_val_fp, orient="records", lines=True)
    test_df.to_json(new_test_fp, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Harmemes to Specified Format")
    parser.add_argument("--annotation-dir", help="path to the annotation directory")
    parser.add_argument("--train-filename", default="train.jsonl", help="filepath to the HarMemes train annotations")
    parser.add_argument("--val-filename", default="val.jsonl", help="filepath to the HarMemes test annotations")
    parser.add_argument("--test-filename", default="test.jsonl", help="filepath to the HarMemes test annotations")
    args = parser.parse_args()

    main(
        args.annotation_dir,
        args.train_filename,
        args.val_filename,
        args.test_filename,
    )