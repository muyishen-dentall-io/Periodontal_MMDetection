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
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from matplotlib.patches import Polygon
import cv2
import random
import argparse


def visualize_image_with_annotations(coco, image_dir, image_id, output_path, show_bbox=True):
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    coco.showAnns(anns)

    category_names = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    if show_bbox:
        for ann in anns:
            bbox = ann['bbox']
            x, y, width, height = bbox
            category_id = ann['category_id']
            category_name = category_names[category_id]

            rect = plt.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='black', facecolor='none')
            plt.gca().add_patch(rect)

            plt.text(x, y - 5, category_name, color='red', fontsize=10, backgroundcolor='white')

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
def main(data_dir, output_path, show_bbox=True):
    annotation_file = os.path.join(data_dir, 'annotations.json')

    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    random_image_id = random.choice(image_ids)
    visualize_image_with_annotations(coco, data_dir, random_image_id, output_path, show_bbox=show_bbox)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize COCO annotations and save the image.')
    
    parser.add_argument('--data_dir', type=str, default='/home/muyishen2040/periodontal_mmdet/dataset/taipei_medical/tmu_coco')
    parser.add_argument('--output_path', type=str, default='./annotated_image.png', 
                        help='Path to save the annotated image. Default: ./annotated_image.png')
    parser.add_argument('--show_bbox', type=bool, default=True, 
                        help='Flag to show bounding boxes. Default: True')

    args = parser.parse_args()

    main(args.data_dir, args.output_path, args.show_bbox)