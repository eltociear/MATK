import os
import glob
import json
import shutil
from multiprocessing import Pool

import fire
import easyocr
import numpy as np
import torch

from PIL import Image
from skimage import transform
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb


def cast_pred_type(pred):
    result = []
    for tup in pred:
        coord, txt, score = tup
        coord = np.array(coord).tolist()
        score = float(score)
        result.append((coord, txt, score))
    return result


def detect(root_dir):
    reader = easyocr.Reader(['en'])
    image_dir = os.path.join(root_dir, 'img')
    images = glob.glob(os.path.join(image_dir, '*.png')) 
    images += glob.glob(os.path.join(image_dir, '**', '*.png')) 
    # images = images[:3]
    print(len(images))
    assert len(images) > 9000 # adjust depending on number of images in image folder

    out_json = os.path.join(root_dir, 'ocr.json')
    out_anno = {}
    print(f"Find {len(images)} images!")

    for i, image_path in enumerate(images):
        print(F"{i}/{len(images)}")
        img_name = os.path.basename(image_path)
        pred = reader.readtext(image_path)
        out_anno[img_name] = cast_pred_type(pred)

    with open(out_json, 'w') as f:
        json.dump(out_anno, f)


def point_to_box(anno_json):
    with open(anno_json, 'r') as f:
        ocr_anno = json.load(f)
    
    boxed_anno = {}
    for k, v in ocr_anno.items():
        img_ocr_infos = []
        for txt_info in v:
            coord, txt, score = txt_info
            xmin = min([p[0] for p in coord])
            xmax = max([p[0] for p in coord])
            ymin = min([p[1] for p in coord])
            ymax = max([p[1] for p in coord])
            box = [xmin, ymin, xmax, ymax]
            img_ocr_infos.append([box, txt, score])
        boxed_anno[k] = img_ocr_infos
    
    out_path = anno_json.replace('.json', '.box.json')
    with open(out_path, 'w') as f:
        json.dump(boxed_anno, f)


def _mask_white_txt(args):
    img_name, img_boxes, img_dir, out_dir = args
    img_path = os.path.join(img_dir, img_name)
    out_path = os.path.join(out_dir, img_name)
    
    if os.path.exists(out_path):
        return
    
    print(out_path)
    img_boxes = [box_info[0] for box_info in img_boxes]
    if len(img_boxes) > 0:
        boxes = np.asarray(img_boxes, dtype=np.int32)
        # print(boxes)
        boxes = np.concatenate([boxes[:, ::-1][:, 2:], boxes[:,::-1][:, :2]], axis=1)
        # print(boxes)
        # x,y,x,y -> y,x,y,x
        img = np.array(Image.open(img_path).convert('RGB'))
        # res = inpaint_model.inpaint_multi_boxes(img, boxes)
        masked_img, mask = multi_boxes_mask(img, boxes)

        Image.fromarray(masked_img).save(out_path)
        out_path = os.path.join(out_dir, img_name.replace('.png', '.mask.png'))
        Image.fromarray(mask).save(out_path)
    else:
        img = np.asarray(Image.open(img_path).convert('RGB'))
        shutil.copy(img_path, out_path)

        mask = np.zeros_like(img)
        out_path = os.path.join(out_dir, img_name.replace('.png', '.mask.png'))
        Image.fromarray(mask).save(out_path)

def generate_mask(ocr_box_anno, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(ocr_box_anno, 'r') as f:
        boxes_anno = json.load(f)

    # for i, (img_name, img_boxes) in enumerate(boxes_anno.items()):
    #     pass
    
    with Pool(16) as pool:
        args = [
            (img_name, img_boxes, img_dir, out_dir)
            for img_name, img_boxes in boxes_anno.items()
        ]
        pool.map(_mask_white_txt, args)


if __name__ == "__main__":
    """
    detect -[ocr.json]-> point_to_box -[ocr.box.json]->  generate_mask
    """

    fire.Fire({
        "detect": detect,
        "point_to_box": point_to_box,
        "generate_mask": generate_mask,
    })
