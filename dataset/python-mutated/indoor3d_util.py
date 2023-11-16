import glob
from os import path as osp
import numpy as np
BASE_DIR = osp.dirname(osp.abspath(__file__))
class_names = [x.rstrip() for x in open(osp.join(BASE_DIR, 'meta_data/class_names.txt'))]
class2label = {one_class: i for (i, one_class) in enumerate(class_names)}

def export(anno_path, out_filename):
    if False:
        for i in range(10):
            print('nop')
    'Convert original dataset files to points, instance mask and semantic\n    mask files. We aggregated all the points from each instance in the room.\n\n    Args:\n        anno_path (str): path to annotations. e.g. Area_1/office_2/Annotations/\n        out_filename (str): path to save collected points and labels\n        file_format (str): txt or numpy, determines what file format to save.\n\n    Note:\n        the points are shifted before save, the most negative point is now\n            at origin.\n    '
    points_list = []
    ins_idx = 1
    for f in glob.glob(osp.join(anno_path, '*.txt')):
        one_class = osp.basename(f).split('_')[0]
        if one_class not in class_names:
            one_class = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * class2label[one_class]
        ins_labels = np.ones((points.shape[0], 1)) * ins_idx
        ins_idx += 1
        points_list.append(np.concatenate([points, labels, ins_labels], 1))
    data_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min
    np.save(f'{out_filename}_point.npy', data_label[:, :6].astype(np.float32))
    np.save(f'{out_filename}_sem_label.npy', data_label[:, 6].astype(np.int))
    np.save(f'{out_filename}_ins_label.npy', data_label[:, 7].astype(np.int))