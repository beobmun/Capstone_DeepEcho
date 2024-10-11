import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps

def convert_to_imgs(video_path):
    cap, fps = load_video(video_path)
    if cap is None:
        return None
    imgs = []
    if cap.isOpened():
        ret, frame = cap.read()
        while ret:
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
            imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            ret, frame = cap.read()
    cap.release()
    # cv2.destroyAllWindows()
    return np.array(imgs), fps

def save_imgs(video_name, save_dir, imgs):
    os.makedirs(f"{save_dir}/{video_name}", exist_ok=True)
    for i, img in enumerate(imgs):
        cv2.imwrite(f"{save_dir}/{video_name}/{i}.jpg", img)
        
def remove_imgs(video_name, save_dir):
    os.system(f"rm -rf {save_dir}/{video_name}")
    
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
def save_seg_video(imgs, video_segment, fps, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    def update(out_frame_idx):
        ax.clear()
        ax.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(imgs[out_frame_idx], cmap='gray')
        if out_frame_idx in video_segment:
            for out_obj_id, out_mask in video_segment[out_frame_idx].items():
                show_mask(out_mask, ax, obj_id=out_obj_id)
    ani = animation.FuncAnimation(fig, update, frames=range(0, len(imgs)))
    ani.save(save_path, writer='ffmpeg', fps=fps)
    plt.close()