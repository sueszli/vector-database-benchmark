import hashlib
import numpy as np
import os
import pytest
import requests
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
from utils_cv.detection.data import coco_labels, coco2voc, Urls, extract_keypoints_from_labelbox_json, extract_masks_from_labelbox_json

def test_urls():
    if False:
        for i in range(10):
            print('nop')
    all_urls = Urls.all()
    for url in all_urls:
        with requests.get(url):
            pass

def test_coco_labels():
    if False:
        for i in range(10):
            print('nop')
    COCO_LABELS_FIRST_FIVE = ('__background__', 'person', 'bicycle', 'car', 'motorcycle')
    labels = coco_labels()
    for i in range(5):
        assert labels[i] == COCO_LABELS_FIRST_FIVE[i]
    assert len(labels) == 91

def test_coco2voc(coco_sample_path):
    if False:
        return 10
    output_dir = 'coco2voc_output'
    coco2voc(anno_path=coco_sample_path, output_dir=output_dir, download_images=False)
    filenames = os.listdir(os.path.join(output_dir, 'annotations'))
    assert len(filenames) == 3

@pytest.fixture(scope='session')
def labelbox_export_data(tmp_session):
    if False:
        while True:
            i = 10
    tmp_session = Path(tmp_session)
    data_dir = tmp_session / 'labelbox_test_data'
    im_dir = data_dir / 'images'
    anno_dir = data_dir / 'annotations'
    im_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)
    keypoint_json_path = tmp_session / 'labelbox_keypoint.json'
    mask_json_path = tmp_session / 'labelbox_mask.json'
    for i in range(2):
        im = Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8))
        im.save(im_dir / f'{i}.jpg')
        anno_xml = '<annotation>\n    <folder>images</folder>\n    <size>\n        <width>500</width>\n        <height>500</height>\n        <depth>3</depth>\n    </size>\n    <object>\n        <name>milk_bottle</name>\n        <bndbox>\n            <xmin>100</xmin>\n            <ymin>100</ymin>\n            <xmax>199</xmax>\n            <ymax>199</ymax>\n        </bndbox>\n    </object>\n    <object>\n        <name>carton</name>\n        <bndbox>\n            <xmin>300</xmin>\n            <ymin>300</ymin>\n            <xmax>399</xmax>\n            <ymax>399</ymax>\n        </bndbox>\n    </object>\n</annotation>\n'
        with open(anno_dir / f'{i}.xml', 'w') as f:
            f.write(anno_xml)
    keypoint_json = '[{\n     "Label": {\n         "milk_bottle_p1": [{"geometry": {"x": 320,"y": 320}}],\n         "milk_bottle_p2": [{"geometry": {"x": 350,"y": 350}}],\n         "milk_bottle_p3": [{"geometry": {"x": 390,"y": 390}}],\n         "carton_p1": [{"geometry": {"x": 130,"y": 130}}],\n         "carton_p2": [{"geometry": {"x": 190,"y": 190}}]\n     },\n     "External ID": "1.jpg"}\n]\n'
    keypoint_truth_dict = {'folder': 'images', 'size': {'width': '500', 'height': '500', 'depth': '3'}, 'object': {'milk_bottle': {'bndbox': {'xmin': '100', 'ymin': '100', 'xmax': '199', 'ymax': '199'}, 'keypoints': {'p1': {'x': '320', 'y': '320'}, 'p2': {'x': '350', 'y': '350'}, 'p3': {'x': '390', 'y': '390'}}}, 'carton': {'bndbox': {'xmin': '300', 'ymin': '300', 'xmax': '399', 'ymax': '399'}, 'keypoints': {'p1': {'x': '130', 'y': '130'}, 'p2': {'x': '190', 'y': '190'}}}}}
    with open(keypoint_json_path, 'w') as f:
        f.write(keypoint_json)
    mask_json = '[{\n     "Label": {\n         "objects": [\n             {\n                 "value": "carton",\n                 "instanceURI": "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/labelbox_test_dummy_carton_mask.png"\n             },\n             {\n                 "value": "milk_bottle",\n                 "instanceURI": "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/labelbox_test_dummy_milk_bottle_mask.png"\n             }\n         ]\n     },\n     "External ID": "1.jpg"}\n]\n'
    with open(mask_json_path, 'w') as f:
        f.write(mask_json)
    return (data_dir, mask_json_path, keypoint_json_path, keypoint_truth_dict)

def test_extract_keypoints_from_labelbox_json(labelbox_export_data, tmp_session):
    if False:
        i = 10
        return i + 15
    (data_dir, _, keypoint_json_path, keypoint_truth_dict) = labelbox_export_data
    keypoint_data_dir = Path(tmp_session) / 'labelbox_test_keypoint_data'
    keypoint_data_dir.mkdir(parents=True, exist_ok=True)
    extract_keypoints_from_labelbox_json(keypoint_json_path, data_dir, keypoint_data_dir)
    subdir_exts = [('annotations', 'xml'), ('images', 'jpg')]
    assert len([str(x) for x in keypoint_data_dir.iterdir()]) == 2
    for (name, ext) in subdir_exts:
        subdir = keypoint_data_dir / name
        file_paths = [x for x in subdir.iterdir()]
        assert len(file_paths) == 1
        assert subdir / f'0.{ext}' not in file_paths
        assert subdir / f'1.{ext}' in file_paths

    def md5sum(path):
        if False:
            print('Hello World!')
        with open(path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        return md5
    im_path = 'images/1.jpg'
    assert md5sum(data_dir / im_path) == md5sum(keypoint_data_dir / im_path)
    tree = ET.parse(keypoint_data_dir / 'annotations' / '1.xml')
    root = tree.getroot()
    assert len(root.findall('folder')) == 1
    assert root.find('folder').text == keypoint_truth_dict['folder']
    assert len(root.findall('size')) == 1
    size_node = root.find('size')
    size_truth = keypoint_truth_dict['size']
    assert len(size_node.findall('width')) == 1
    assert size_node.find('width').text == size_truth['width']
    assert size_node.find('height').text == size_truth['height']
    assert size_node.find('depth').text == size_truth['depth']
    obj_nodes = root.findall('object')
    obj_truths = keypoint_truth_dict['object']
    assert len(obj_nodes) == len(obj_truths)
    for obj_node in obj_nodes:
        obj_name = obj_node.find('name').text
        bndbox_node = obj_node.find('bndbox')
        bndbox_truth = obj_truths[obj_name]['bndbox']
        for coord in bndbox_truth:
            assert bndbox_node.find(coord).text == bndbox_truth[coord]
        kp_node = obj_node.find('keypoints')
        kp_truth = obj_truths[obj_name]['keypoints']
        for kp_name in kp_truth:
            p_node = kp_node.find(kp_name)
            p_truth = kp_truth[kp_name]
            assert p_node.find('x').text == p_truth['x']
            assert p_node.find('y').text == p_truth['y']

def test_extract_masks_from_labelbox_json(labelbox_export_data, tmp_session):
    if False:
        return 10
    (data_dir, mask_json_path, _, _) = labelbox_export_data
    mask_data_dir = Path(tmp_session) / 'labelbox_test_mask_data'
    mask_data_dir.mkdir(parents=True, exist_ok=True)
    extract_masks_from_labelbox_json(mask_json_path, data_dir, mask_data_dir)
    assert len([str(x) for x in mask_data_dir.iterdir()]) == 3
    for (name, ext) in [('annotations', 'xml'), ('images', 'jpg'), ('segmentation-masks', 'png')]:
        subdir = mask_data_dir / name
        file_paths = [x for x in subdir.iterdir()]
        assert len(file_paths) == 1
        assert subdir / f'0.{ext}' not in file_paths
        assert subdir / f'1.{ext}' in file_paths

    def md5sum(path):
        if False:
            return 10
        with open(path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        return md5
    for name in ['images/1.jpg', 'annotations/1.xml']:
        assert md5sum(data_dir / name) == md5sum(mask_data_dir / name)
    mask = np.array(Image.open(mask_data_dir / 'segmentation-masks' / '1.png'))
    assert mask.shape == (500, 500)
    assert np.all(mask[100:200, 100:200] == 1)
    assert np.all(mask[300:400, 300:400] == 2)