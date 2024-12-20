import os
from .cuda_set import *
from .dataset import *
from .get_points import *
from .unet import *
from .video import *

import torch
import torchvision
import argparse
import pandas as pd
import os
import torch.nn.utils
from sam2.sam2_video_predictor import SAM2VideoPredictor


unet_path = 'trained_models/unet.pth'
sam2_base_model = 'facebook/sam2-hiera-large'
sam2_tuned_model_path = '/home/behan/Capstone_DeepEcho/sandbox/aj/Fine_Tuned_model_FIX/fine_tuned_sam2_9500.torch'

def segment_video(video_path, output_path):
    device = cuda_set()
        
    unet = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    unet.load_state_dict(torch.load(unet_path))
    unet.eval()
    print("U-Net load successfully.")
    
    name = video_path.split('/')[-1].split('.')[0]
    print('---------------------------------')
    print(f'start segmentation for {name}...')
    print('---------------------------------')
    
    imgs, fps, width, height = convert_to_imgs(video_path)
    save_imgs(name, output_path, imgs)
    
    transform = torchvision.transforms.Compose([Resize((256, 256))])
    dataset = Dataset(output_path, name, transform=transform)
    
    with torch.no_grad():
        img = dataset.__getitem__(0).to(device)
        mask = torch.argmax(unet(img), dim=1).cpu()
        mask = mask.permute(1,2,0).squeeze()+0.5
        mask = cv2.resize(mask.numpy(), (width, height), interpolation=cv2.INTER_CUBIC)
        mask = (mask > mask.max()*0.8).astype(int)

        img = img.cpu()[0].permute(1,2,0).squeeze()+0.5
        img = cv2.resize(img.numpy(), (width, height), interpolation=cv2.INTER_CUBIC)

    predictor = SAM2VideoPredictor.from_pretrained(sam2_base_model, device=device)
    predictor.load_state_dict(torch.load(sam2_tuned_model_path))
    
    inference_state = predictor.init_state(video_path=f'{output_path}/{name}')
    predictor.reset_state(inference_state)
    
    points, labels = get_points(mask)
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
        
    # save_seg_video(imgs, video_segment, fps, f'{output_path}/{name}.mp4')
    save_seg_video(imgs, video_segment, fps, width, height, f'{output_path}/{name}.mp4')

    remove_imgs(name, output_path)

    print('---------------------------------')
    print(f'{output_path}/{name}.mp4 saved.')
    print(f'{name} segmentation done.')
    print('---------------------------------')
    # os.system(f'mpv --no-config --loop=yes --vo=tct {output_path}/{name}.mp4')
    return (imgs, video_segment)