import torch
import torchvision
import argparse
import pandas as pd
import os
import torch.nn.utils
from sam2.sam2_video_predictor import SAM2VideoPredictor

from utils.cuda_set import *
from utils.unet import *
from utils.video import *
from utils.dataset import *
from utils.get_points import *

unet_path = 'trained_models/unet.pth'
save_dir = '../dataset/VideoPradictor'
sam2_base_model = 'facebook/sam2-hiera-large'
sam2_tuned_model_path = '/home/behan/Capstone_DeepEcho/sandbox/aj/Fine_Tuned_model_all/fine_tuned_sam2_5000.torch'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(nargs='+', dest='video_path', help='Path to the video file')
    parser.add_argument('--output', '-o', nargs='*', default=save_dir)

    video_path_list = parser.parse_args().video_path
    output_path = parser.parse_args().output
    
    return video_path_list, output_path

def main(video_path_list, output_path):
    device = cuda_set()
        
    unet = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    unet.load_state_dict(torch.load(unet_path))
    unet.eval()
    print("U-Net load successfully.")
    
    for video_path in video_path_list:
        name = video_path.split('/')[-1].split('.')[0]
        print('---------------------------------')
        print(f'start segmentation for {name}...')
        print('---------------------------------')
        
        imgs, fps = convert_to_imgs(video_path)
        save_imgs(name, output_path, imgs)
        
        transform = torchvision.transforms.Compose([Resize((256, 256))])
        dataset = Dataset(output_path, name, transform=transform)
        
        with torch.no_grad():
            img = dataset.__getitem__(0).to(device)
            mask = torch.argmax(unet(img), dim=1).cpu()
            img = img.cpu()[0].permute(1,2,0).squeeze()+0.5

        predictor = SAM2VideoPredictor.from_pretrained(sam2_base_model, device=device)
        predictor.load_state_dict(torch.load(sam2_tuned_model_path))
        
        inference_state = predictor.init_state(video_path=f'{output_path}/{name}')
        predictor.reset_state(inference_state)
        
        points, labels = get_points(mask[0])
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=0,
            points=points,
            labels=labels
        )
        
        video_segment = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segment[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
        save_seg_video(imgs, video_segment, fps, f'{output_path}/{name}.mp4')
        remove_imgs(name, output_path)
        print('---------------------------------')
        print(f'{name} segmentation done.')
        print('---------------------------------')
        os.system(f'mpv --no-config --loop=yes --vo=tct {output_path}/{name}.mp4')
        
        
if __name__ == "__main__":
    video_path_list, output_path = get_arguments()
    main(video_path_list, output_path)