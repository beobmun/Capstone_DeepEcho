import numpy as np

def continuous(row):
    for i in range(len(row)):
        r = row[i]
        if r == 0:
            return i
    return len(row)

def calc_volume_v1(mask):
    volume = 0
    for r in mask:
        i = 0
        while i < len(r):
            if r[i] == 0:
                i += 1
                continue
            j = continuous(r[i:])
            v = ((j/2)**2)*np.pi
            volume += v
            i += j
    return volume

def calc_volume_v2(mask):
    volume = np.zeros_like(mask)
    volume = volume.astype(np.float32)
    for r_i, r in enumerate(mask):
        i = 0
        while i < len(r):
            if r[i] == 0:
                i += 1
                continue
            diameter = continuous(r[i:])
            radius = diameter//2
            for j, a in enumerate(range(radius, 0, -1)):
                theta = np.arccos(a/radius)
                v = a * np.tan(theta)
                volume[r_i][i+j] = v
                volume[r_i][i+diameter-j-2] = v
            # for j, a in enumerate(range(1, radius)):
            #     theta = np.arccos(a/radius)
            #     v = a * np.tan(theta)
            #     volume[r_i][i+radius+j] = v
            i += diameter    
    return volume

def calc_ef(esv, edv):
    return (edv - esv)/edv * 100