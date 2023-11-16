"""Common utilities for data pre-processing, e.g. matching moving object across frames."""
import numpy as np

def compute_overlap(mask1, mask2):
    if False:
        i = 10
        return i + 15
    return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

def align(seg_img1, seg_img2, seg_img3, threshold_same=0.3):
    if False:
        for i in range(10):
            print('nop')
    res_img1 = np.zeros_like(seg_img1)
    res_img2 = np.zeros_like(seg_img2)
    res_img3 = np.zeros_like(seg_img3)
    remaining_objects2 = list(np.unique(seg_img2.flatten()))
    remaining_objects3 = list(np.unique(seg_img3.flatten()))
    for seg_id in np.unique(seg_img1):
        max_overlap2 = float('-inf')
        max_segid2 = -1
        for seg_id2 in remaining_objects2:
            overlap = compute_overlap(seg_img1 == seg_id, seg_img2 == seg_id2)
            if overlap > max_overlap2:
                max_overlap2 = overlap
                max_segid2 = seg_id2
        if max_overlap2 > threshold_same:
            max_overlap3 = float('-inf')
            max_segid3 = -1
            for seg_id3 in remaining_objects3:
                overlap = compute_overlap(seg_img2 == max_segid2, seg_img3 == seg_id3)
                if overlap > max_overlap3:
                    max_overlap3 = overlap
                    max_segid3 = seg_id3
            if max_overlap3 > threshold_same:
                res_img1[seg_img1 == seg_id] = seg_id
                res_img2[seg_img2 == max_segid2] = seg_id
                res_img3[seg_img3 == max_segid3] = seg_id
                remaining_objects2.remove(max_segid2)
                remaining_objects3.remove(max_segid3)
    return (res_img1, res_img2, res_img3)