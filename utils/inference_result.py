import os
import random
import mmcv
import torch
import argparse
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules

def run_inference(config_file, checkpoint_file, dataset_dir, output_dir, draw_gt=False):
    
    register_all_modules()

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_detector(config_file, checkpoint_file, device=device)

    
    random_image = random.choice(os.listdir(dataset_dir))  
    image_path = os.path.join(dataset_dir, random_image)

    
    image = mmcv.imread(image_path)

    
    result = inference_detector(model, image)

    
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta  

    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'result_{random_image}')

    
    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt=draw_gt,  
        wait_time=0,  
        out_file=output_file  
    )

    print(f"Inference result saved to {output_file}")

def main():
    
    parser = argparse.ArgumentParser(description='Run inference on a random image using a trained model.')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the model config file.')
    parser.add_argument('--checkpoint_file', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing the images.')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save the output images.')
    parser.add_argument('--draw_gt', action='store_true', help='Option to draw ground truth annotations.')

    args = parser.parse_args()

    
    run_inference(args.config_file, args.checkpoint_file, args.dataset_dir, args.output_dir, args.draw_gt)

if __name__ == '__main__':
    main()
