import torch, torchvision
import mmdet
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import os
import random
import json
import random
import argparse


def split_coco_dataset(annotation_file, output_dir, train_ratio=0.8):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    random.shuffle(images)
    
    split_idx = int(train_ratio * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    train_img_ids = {img['id'] for img in train_images}
    val_img_ids = {img['id'] for img in val_images}
    
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_img_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_img_ids]
    
    train_data = {k: coco_data[k] for k in coco_data if k != 'images' and k != 'annotations'}
    val_data = {k: coco_data[k] for k in coco_data if k != 'images' and k != 'annotations'}
    train_data['images'] = train_images
    train_data['annotations'] = train_annotations
    val_data['images'] = val_images
    val_data['annotations'] = val_annotations

    with open(os.path.join(output_dir, 'train_annotations.json'), 'w') as f:
        json.dump(train_data, f)
    
    with open(os.path.join(output_dir, 'val_annotations.json'), 'w') as f:
        json.dump(val_data, f)
    
    print(f"Train and validation sets created and saved to {output_dir}.")

def verify_coco_split(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    
    print(f"File: {annotation_file}")
    print(f"Number of images: {num_images}")
    print(f"Number of annotations: {num_annotations}")
    print("-" * 30)

def main(annotation_file, output_dir, train_ratio=0.8):
    
    split_coco_dataset(
        annotation_file=annotation_file,
        output_dir=output_dir,
        train_ratio=0.8
    )
    
    train_json = os.path.join(output_dir, 'train_annotations.json')
    val_json = os.path.join(output_dir, 'val_annotations.json')

    verify_coco_split(train_json)
    verify_coco_split(val_json)
    verify_coco_split(annotation_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split and verify a COCO dataset into training and validation sets.')
    
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to the COCO annotation JSON file.')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save the split JSON files.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of the dataset to be used for training (default is 0.8).')

    args = parser.parse_args()

    main(args.annotation_file, args.output_dir, args.train_ratio)
