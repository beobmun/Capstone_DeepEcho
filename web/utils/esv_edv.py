import numpy as np
import os
import cv2
import plotly
import plotly.graph_objs as go
import json
from flask import url_for
from .video import convert_to_imgs

def calc_areas(imgs, video_segment):
    areas = []
    masks = []
    
    for out_frame_idx in range(len(imgs)):
        if out_frame_idx in video_segment:
            for out_obj_id, out_mask in video_segment[out_frame_idx].items():
                mask = out_mask.astype(np.uint8).squeeze()
                mask = mask > 0
                area = np.sum(mask)
                masks.append(mask)
                areas.append(area)
    return np.array(areas), np.array(masks)

def find_esv_edv(areas, conv_base): # conv_base = fps/4
    conved_areas = np.convolve(areas, np.ones(conv_base)/conv_base, mode='valid')
    mean_area = np.mean(conved_areas)
    sv_idx = []
    dv_idx = []
    for i, area in enumerate(conved_areas):
        if area < mean_area:
            sv_idx.append(i)
        else:
            dv_idx.append(i)
    
    s = 0
    esv_list = []
    for i in range(len(sv_idx)):
        if i == len(sv_idx)-1:
            temp = sv_idx[s:]
            esv_list.append([temp[0], temp[-1]])
            break
        current_idx = sv_idx[i]
        next_idx = sv_idx[i+1]
        if next_idx - current_idx == 1:
            continue
        temp = sv_idx[s:i+1]
        s = i+1
        esv_list.append([temp[0], temp[-1]])
    esv_points = []
    for r in esv_list:
        try:
            esv_points.append(r[0] + np.argmin(conved_areas[r[0]:r[1]]))
        except Exception as e:
            print(e)
            
    s = 0
    edv_list = []
    for i in range(len(dv_idx)):
        if i == len(dv_idx)-1:
            temp = dv_idx[s:]
            edv_list.append([temp[0], temp[-1]])
            break
        current_idx = dv_idx[i]
        next_idx = dv_idx[i+1]
        if next_idx - current_idx == 1:
            continue
        temp = dv_idx[s:i+1]
        s = i+1
        edv_list.append([temp[0], temp[-1]])
    edv_points = []
    for r in edv_list:
        try:
            edv_points.append(r[0] + np.argmax(conved_areas[r[0]:r[1]]))
        except Exception as e:
            print(e)
            
    esv_points, edv_points = np.array(esv_points), np.array(edv_points)
    esv_points = np.sort(esv_points)
    edv_points = np.sort(edv_points)
    
    if esv_points[0] < 5:
        esv_points = esv_points[1:]
    if esv_points[-1] > len(conved_areas) - 5:
        esv_points = esv_points[:-1]
    
    if edv_points[0] < 5:
        edv_points = edv_points[1:]
    if edv_points[-1] > len(conved_areas) - 5:
        edv_points = edv_points[:-1]
    
    esv_points += conv_base//2
    edv_points += conv_base//2
    
    # frames = np.concatenate([esv_points, edv_points])
    # frames = np.sort(frames)
    
    return conved_areas, esv_points, edv_points

def save_frames(frames, video_path, save_path):
    imgs, _, _, _ = convert_to_imgs(video_path, gray=False)
    os.makedirs(f'{save_path}', exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(f'{save_path}/{i}.jpg', imgs[f])
        
def create_graph(imgs, video_segment, video_path, save_path, conv_base):
    areas, _ = calc_areas(imgs, video_segment)
    conved_areas, esv_points, edv_points = find_esv_edv(areas, conv_base)
    save_frames(esv_points, video_path, f"static/{save_path}/esv")
    save_frames(edv_points, video_path, f"static/{save_path}/edv")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(conved_areas)), 
        y=conved_areas, 
        mode='lines', 
        name='area',
    ))
    
    esv_x = esv_points - conv_base//2
    edv_x = edv_points - conv_base//2
    esv_y = conved_areas[esv_x]
    edv_y = conved_areas[edv_x]
    
    hover_images = []
    esv_text = []
    edv_text = []
    
    name = save_path.split('/')[-1]
    for i in range(len(esv_x)):
        img_path = f'{save_path}/esv/{i}.jpg'
        img_url = url_for("static", filename=img_path)
        hover_images.append({'src': img_url, 'point': f'ESV_{i}'})
        esv_text.append(f'ESV_{i}')
    
    for i in range(len(edv_x)):
        img_path = f'{save_path}/edv/{i}.jpg'
        img_url = url_for("static", filename=img_path)
        hover_images.append({'src': img_url, 'point': f'EDV_{i}'})
        edv_text.append(f'EDV_{i}')
    
    fig.add_trace(go.Scatter(
        x=edv_x,
        y=edv_y,
        mode='markers',
        text=edv_text,
        marker=dict(color='blue', size=8),
        name='EDV'
    ))
    
    fig.add_trace(go.Scatter(
        x=esv_x,
        y=esv_y,
        mode='markers',
        text=esv_text,
        marker=dict(color='red', size=8),
        name='ESV'
    ))
    
    fig.update_layout(
        title_text='Left Ventricular Area Variation',
        title_x=0.5,
        title_y=0.9,
        autosize=True,
        width=1200,
        height=400,
    )
    fig.update_xaxes(title_text='Frame')
        
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON, hover_images