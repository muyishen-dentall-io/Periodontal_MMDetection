import os
import random
import mmcv
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules

# Initialize MMDetection modules
register_all_modules()

# config_file = 'tmu_mask_rcnn_r50_fpn_1x.py'  # Path to your config file
# checkpoint_file = 'work_dirs/tmu_mask_rcnn/epoch_12.pth'  # Path to your trained model
config_file = 'tmu_solov2_r50_fpn_1x_coco.py'  # Path to your config file
checkpoint_file = 'work_dirs/tmu_solov2/epoch_12.pth'  # Path to your trained model

# Initialize the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

# Randomly select an image from your dataset
dataset_dir = '/home/muyishen2040/taipei_medical/tmu_coco/JPEGImages'  # Path to your image directory
random_image = random.choice(os.listdir(dataset_dir))
image_path = os.path.join(dataset_dir, random_image)

# Load the image
image = mmcv.imread(image_path)

# Run inference
result = inference_detector(model, image)

# Initialize the visualizer (this should be done once)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta  # Meta info from the model

# Visualize the result and optionally save to a file
output_file = f'outputs/result_{random_image}'
visualizer.add_datasample(
    'result',
    image,
    data_sample=result,
    draw_gt=False,  # Do not draw ground truth (set to True if you want to draw it)
    wait_time=0,  # No wait time when displaying the result
    out_file=output_file  # Path to save the output image
)

print(f"Inference result saved to {output_file}")
