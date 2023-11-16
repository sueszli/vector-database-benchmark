"""Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts
"""
import csv
import os
import numpy as np
from plyfile import PlyData

def represents_int(s):
    if False:
        i = 10
        return i + 15
    'Judge whether string s represents an int.\n\n    Args:\n        s(str): The input string to be judged.\n\n    Returns:\n        bool: Whether s represents int or not.\n    '
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    if False:
        for i in range(10):
            print('nop')
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for (k, v) in mapping.items()}
    return mapping

def read_mesh_vertices(filename):
    if False:
        for i in range(10):
            print('nop')
    'Read XYZ for each vertex.\n\n    Args:\n        filename(str): The name of the mesh vertices file.\n\n    Returns:\n        ndarray: Vertices.\n    '
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
    return vertices

def read_mesh_vertices_rgb(filename):
    if False:
        return 10
    'Read XYZ and RGB for each vertex.\n\n    Args:\n        filename(str): The name of the mesh vertices file.\n\n    Returns:\n        Vertices. Note that RGB values are in 0-255.\n    '
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices