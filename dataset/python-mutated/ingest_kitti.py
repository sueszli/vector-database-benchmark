from __future__ import print_function
import os
import glob
import json
import numpy as np
from PIL import Image
from zipfile import ZipFile
import math
from neon.util.persist import ensure_dirs_exist
from tqdm import tqdm

def convert_annot_to_json(path, im_path, out_path, difficult):
    if False:
        i = 10
        return i + 15
    "\n    Converts the KITTI annotations to json file.\n\n    Uses the below reference for the KITTI dataset:\n\n    OO representation of label format used in Kitti dataset.\n\n    Description of fields from Kitti dataset dev kit: (link)[]\n    The label files contain the following information, which can be read and\n    written using the matlab tools (readLabels.m, writeLabels.m) provided within\n    this devkit. All values (numerical or strings) are separated via spaces,\n    each row corresponds to one object. The 15 columns represent:\n    #Values    Name      Description\n    ----------------------------------------------------------------------------\n       1    type         Describes the type of object: 'Car', 'Van', 'Truck',\n                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',\n                         'Misc' or 'DontCare'\n       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where\n                         truncated refers to the object leaving image boundaries\n       1    occluded     Integer (0,1,2,3) indicating occlusion state:\n                         0 = fully visible, 1 = partly occluded\n                         2 = largely occluded, 3 = unknown\n       1    alpha        Observation angle of object, ranging [-pi..pi]\n       4    bbox         2D bounding box of object in the image (0-based index):\n                         contains left, top, right, bottom pixel coordinates\n       3    dimensions   3D object dimensions: height, width, length (in meters)\n       3    location     3D object location x,y,z in camera coordinates (in meters)\n       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]\n       1    score        Only for results: Float, indicating confidence in\n                         detection, needed for p/r curves, higher is better.\n\n    Arguments:\n        path (string): path to KITTI annotation file\n        im_path (string): path to image\n        out_path (string): path to save the json file\n        difficult (bool): include difficult objects\n    "
    with open(path) as f:
        labels = f.readlines()
    annot = {'object': []}
    im = np.array(Image.open(im_path))
    (h, w, c) = im.shape
    annot['size'] = {'depth': c, 'height': h, 'width': w}
    for label in labels:
        vals = label.split()
        type = vals[0]
        truncated = float(vals[1])
        occluded = int(vals[2])
        bbox = tuple([float(x) for x in vals[4:8]])
        bbox_int = tuple([int(math.floor(x)) for x in bbox])
        if type == 'DontCare':
            assert truncated == -1
            assert occluded == -1
        else:
            assert occluded in (0, 1, 2, 3)
        diff = truncated > 0.5 or occluded == 2
        obj = {'bndbox': {'xmin': bbox_int[0], 'ymin': bbox_int[1], 'xmax': bbox_int[2], 'ymax': bbox_int[3]}, 'difficult': difficult, 'name': type, 'truncated': truncated > 0.5, 'occluded': occluded}
        if not diff or difficult:
            annot['object'].append(obj)
    with open(out_path, 'w') as f:
        json.dump(annot, f, indent=4)

def ingest_kitti(input_dir, out_dir, train_percent=90, overwrite=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ingests the KITTI dataset. Peforms the following ops:\n    0. Unzips the files into output directory.\n    1. Convert annotations to json format\n    2. Split the training data into train and validation sets\n    3. Write manifest file\n    4. Write configuration file\n\n    Arguments:\n        input_dir (string): path to folder with KITTI zip files.\n        out_dir (string): path to unzip KITTI data\n        train_percent (float): percent of data to use for training.\n        overwrite (bool): overwrite existing files\n    '
    data_dir = ensure_dirs_exist(os.path.join(out_dir, 'kitti'))
    train_manifest = os.path.join(data_dir, 'train.csv')
    val_manifest = os.path.join(data_dir, 'val.csv')
    if not overwrite and os.path.exists(train_manifest) and os.path.exists(val_manifest):
        print('Found existing manfiest files, skipping ingest,\n              Use --overwrite to rerun ingest anyway.')
        return (train_manifest, val_manifest)
    zipfiles = [os.path.join(input_dir, zipfile) for zipfile in ['data_object_image_2.zip', 'data_object_label_2.zip']]
    for file in zipfiles:
        with ZipFile(file, 'r') as zf:
            print('Extracting {} to {}'.format(file, data_dir))
            zf.extractall(data_dir)
    img_path = os.path.join(data_dir, 'training', 'image_2')
    annot_path = os.path.join(data_dir, 'training', 'label_2')
    images = [os.path.splitext(os.path.basename(im))[0] for im in glob.glob(os.path.join(img_path, '*.png'))]
    print('Found {} images'.format(len(images)))
    assert len(images) > 0, 'Did not found any images. Check your input_dir.'
    annot_save_dir = ensure_dirs_exist(os.path.join(data_dir, 'training', 'label_2-json/'))
    annot_save_dir_difficult = ensure_dirs_exist(os.path.join(data_dir, 'training', 'label_2-json-difficult/'))
    print('Writing annotations to: {} and {}'.format(annot_save_dir, annot_save_dir_difficult))
    for im in tqdm(images):
        path = os.path.join(annot_path, im + '.txt')
        im_path = os.path.join(img_path, im + '.png')
        assert os.path.exists(im_path)
        out_path = os.path.join(annot_save_dir, im + '.json')
        convert_annot_to_json(path, im_path, out_path, difficult=False)
        out_path = os.path.join(annot_save_dir_difficult, im + '.json')
        convert_annot_to_json(path, im_path, out_path, difficult=True)
    np.random.seed(0)
    np.random.shuffle(images)
    train_count = len(images) * train_percent // 100
    train = images[:train_count]
    val = images[train_count:]
    create_manifest(train_manifest, train, annot_save_dir, img_path, data_dir)
    create_manifest(val_manifest, val, annot_save_dir_difficult, img_path, data_dir)
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'kitti.cfg')
    with open(config_path, 'w') as f:
        f.write('manifest = [train:{}, val:{}]\n'.format(train_manifest, val_manifest))
        f.write('manifest_root = {}\n'.format(data_dir))
        f.write('epochs = 14\n')
        f.write('height = 375\n')
        f.write('width = 1242\n')
        f.write('batch_size = 1\n')
    print('Wrote config file to: {}'.format(config_path))

def create_manifest(manifest_path, index_list, annot_dir, image_dir, root_dir):
    if False:
        return 10
    records = [('@FILE', 'FILE')]
    for tag in index_list:
        image = os.path.join(image_dir, tag + '.png')
        annot = os.path.join(annot_dir, tag + '.json')
        assert os.path.exists(image), 'Path {} not found'.format(image)
        assert os.path.exists(annot), 'Path {} not found'.format(annot)
        records.append((os.path.relpath(image, root_dir), os.path.relpath(annot, root_dir)))
    print('Writing manifest file to: {}'.format(manifest_path))
    np.savetxt(manifest_path, records, fmt='%s\t%s')
if __name__ == '__main__':
    from configargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='path to dir with KITTI zip files.')
    parser.add_argument('--output_dir', required=True, help='path to unzip data.')
    parser.add_argument('--overwrite', action='store_true', help='overwrite files')
    parser.add_argument('--training_pct', default=90, help='fraction of data used for training.')
    args = parser.parse_args()
    ingest_kitti(args.input_dir, args.output_dir, args.training_pct, overwrite=args.overwrite)