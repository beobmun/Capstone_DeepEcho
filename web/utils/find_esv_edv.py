from itertools import groupby
from operator import itemgetter
import numpy as np

def calc_mask_area(imgs, video_segment):
    areas = []
    masks = []
    for out_frame_idx in range(len(imgs)):
        if out_frame_idx in video_segment:
            for out_obj_id, out_mask in video_segment[out_frame_idx].items():
                mask = out_mask.astype(np.uint8).squeeze()
                mask = mask > 0
                masks.append(mask)
                area = np.sum(mask)
                areas.append(area)

    areas = np.array(areas)
    masks = np.array(masks)
    return masks, areas

def find_esv_edv(areas):
    gradient = np.gradient(areas)
    indices = np.where((gradient > -5) & (gradient < 5))[0]
    idx_groups = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(indices), lambda x: x[0] - x[1])]
    min_indices = [group[np.argmin(np.abs(gradient[group]))] for group in idx_groups]
    esv_points = [idx for idx in min_indices if areas[idx] < np.mean(areas)]
    edv_points = [idx for idx in min_indices if areas[idx] >= np.mean(areas)]
    esv_points = np.array(esv_points)
    edv_points = np.array(edv_points)
    return esv_points, edv_points

def find_esv_edv_1(areas):
    mean_area = np.mean(areas)
    sv_idx = []
    dv_idx = []
    for i, area in enumerate(areas):
        if area < mean_area:
            sv_idx.append(i)
        else:
            dv_idx.append(i)

    s = 0
    esv_list = []
    for i in range(0, len(sv_idx)):
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
            esv_points.append(r[0] + np.argmin(areas[r[0]:r[1]]))
        except Exception as e:
            print(f"Error occurred: {e}")
        
    s = 0
    edv_list = []
    for i in range(0, len(dv_idx)):
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
            edv_points.append(r[0] + np.argmax(areas[r[0]:r[1]]))
        except Exception as e:
            print(f"Error occurred: {e}")
        
    esv_points, edv_points = np.array(esv_points), np.array(edv_points)
    if esv_points[0] < 5:
        esv_points = esv_points[1:]
    if esv_points[-1] > len(areas) - 5:
        esv_points = esv_points[:-1]
        
    if edv_points[0] < 5:
        edv_points = edv_points[1:]
    if edv_points[-1] > len(areas) - 5:
        edv_points = edv_points[:-1]
        
    return esv_points, edv_points