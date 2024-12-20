import numpy as np

def get_rect_coords(mask):
    top, bottom, left, right = 0, 0, 0, 0
    threshold = mask.max()*0.5
    for r in range(mask.shape[0]):
        if mask[r].max() > threshold:
            top = r
            break
    for r in range(mask.shape[0]-1, -1, -1):
        if mask[r].max() > threshold:
            bottom = r
            break
    for c in range(mask.shape[1]):
        if mask[:, c].max() > threshold:
            left = c
            break
    for c in range(mask.shape[1]-1, -1, -1):
        if mask[:, c].max() > threshold:
            right = c
            break
    hori_1 = top + ((bottom - top) // 3)
    hori_2 = top + 2 * ((bottom - top) // 3)
    vert = left + ((right - left) // 2)
    
    return top, bottom, left, right, hori_1, hori_2, vert

def rect_mask(top, bottom, left, right, mask):
    rect = np.zeros((bottom - top, right - left))
    for r in range(top, bottom):
        for c in range(left, right):
            rect[r-top, c-left] = mask[r, c]
    return rect

def get_pos(mask, base_rc):
    threshold = mask.max()*0.9
    coords = np.argwhere(mask > threshold)
    if len(coords) > 25:
        yx = np.array(coords[np.random.randint(len(coords))])
        x, y = yx[1] + base_rc[1], yx[0] + base_rc[0]
        return [x, y]
    return None
 

def get_neg(mask, base_rc):
    threshold = mask.max()*0.1
    coords = np.argwhere(mask < threshold)
    if len(coords) > 25:
        yx = np.array(coords[np.random.randint(len(coords))])
        x, y = yx[1] + base_rc[1], yx[0] + base_rc[0]
        return [x, y]
    return None
            
def get_points(mask):
    top, bottom, left, right, hori_1, hori_2, vert = get_rect_coords(mask)
    
    rectangles = [
        rect_mask(top, hori_1, left, vert, mask),
        rect_mask(top, hori_1, vert, right, mask),
        rect_mask(hori_1, hori_2, left, vert, mask),
        rect_mask(hori_1, hori_2, vert, right, mask),
        rect_mask(hori_2, bottom, left, vert, mask),
        rect_mask(hori_2, bottom, vert, right, mask)
    ]
        
    pos_points = [
        get_pos(rectangles[0], [top, left]),
        get_pos(rectangles[1], [top, vert]),
        get_pos(rectangles[2], [hori_1, left]),
        get_pos(rectangles[3], [hori_1, vert]),
        get_pos(rectangles[4], [hori_2, left]),
        get_pos(rectangles[5], [hori_2, vert])
    ]
    
    neg_points = [
        get_neg(rectangles[0], [top, left]),
        get_neg(rectangles[1], [top, vert]),
        get_neg(rectangles[2], [hori_1, left]),
        get_neg(rectangles[3], [hori_1, vert]),
        get_neg(rectangles[4], [hori_2, left]),
        get_neg(rectangles[5], [hori_2, vert])
    ]
    points = []
    labels = []
    for p in pos_points:
        if p is not None:
            # points.append([p])
            # labels.append([1])
            points.append(p)
            labels.append(1)
    for p in neg_points:
        if p is not None:
            # points.append([p])
            # labels.append([0])
            points.append(p)
            labels.append(0)
    return np.array(points), np.array(labels)

def draw_rect(mask):
    rectangle = np.zeros_like(mask)
    top, bottom, left, right, hori_1, hori_2, vert = get_rect_coords(mask)
    for c in range(left, right+1):
        rectangle[top, c] = 255
        rectangle[hori_1, c] = 255
        rectangle[hori_2, c] = 255
        rectangle[bottom, c] = 255
    for r in range(top, bottom+1):
        rectangle[r, left] = 255
        rectangle[r, vert] = 255
        rectangle[r, right] = 255
    return rectangle

def draw_points(mask):
    points, labels = get_points(mask)
    r, g, b = np.zeros_like(mask), np.zeros_like(mask), np.zeros_like(mask)
    for (p, l) in zip(points, labels):
        # p = p[0]
        # l = l[0]
        if l == 1:
            b[p[1]][p[0]] = 255
        else:
            r[p[1]][p[0]] = 255
    return np.stack([r, g, b], axis=-1)