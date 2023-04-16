# gqa_lxmert

1. 
```
pip install -r requirements.txt
```

2. Download images from https://nlp.stanford.edu/data/gqa/images.zip and unzip to GQA_IMAGES_DIR

3. Run the following to get only the features for the testdev in hf_generated.datasets_testdev.datasets:
```
python extracting_data.py -i GQA_IMAGES_DIR -o hf_generated.datasets -t hf_generated.datasets_testdev.datasets -b BATCH_SIZE
```

4. Run the following to get GQA testdev questions:
```
wget https://nlp.stanford.edu/data/gqa/questions1.2.zip
unzip -j questions1.2.zip testdev_balanced_questions.json
```

5. Get features from the original LXMERT repo:
```
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/gqa_testdev_obj36.zip
unzip gqa_testdev_obj36.zip && rm gqa_testdev_obj36.zip
```

6. Run evaluation:
```
python gqa_evaluate.py
```

Accuracy -
With `visual_features_file_path = "hf_generated.datasets_testdev.datasets"`: 0.5853871839720146 (when `extracting_data.py` was ran with batch size of 1)
With `visual_features_file_path = "vg_gqa_imgfeat/gqa_testdev_obj36.tsv"`: 0.592940054062649
