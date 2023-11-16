"""Preprocesses COCO minival data for Object Detection evaluation using mean Average Precision.

The 2014 validation images & annotations can be downloaded from:
http://cocodataset.org/#download
The minival image ID allowlist, a subset of the 2014 validation set, can be
found here:
https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt.

This script takes in the original images folder, instances JSON file and
image ID allowlist and produces the following in the specified output folder:
A subfolder for allowlisted images (images/), and a file (ground_truth.pbtxt)
containing an instance of tflite::evaluation::ObjectDetectionGroundTruth.
"""
import argparse
import ast
import collections
import os
import shutil
import sys
from absl import logging
from tensorflow.lite.tools.evaluation.proto import evaluation_stages_pb2

def _get_ground_truth_detections(instances_file, allowlist_file=None, num_images=None):
    if False:
        i = 10
        return i + 15
    "Processes the annotations JSON file and returns ground truth data corresponding to allowlisted image IDs.\n\n  Args:\n    instances_file: COCO instances JSON file, usually named as\n      instances_val20xx.json.\n    allowlist_file: File containing COCO minival image IDs to allowlist for\n      evaluation, one per line.\n    num_images: Number of allowlisted images to pre-process. First num_images\n      are chosen based on sorted list of filenames. If None, all allowlisted\n      files are preprocessed.\n\n  Returns:\n    A dict mapping image id (int) to a per-image dict that contains:\n      'filename', 'image' & 'height' mapped to filename & image dimensions\n      respectively\n      AND\n      'detections' to a list of detection dicts, with each mapping:\n        'category_id' to COCO category id (starting with 1) &\n        'bbox' to a list of dimension-normalized [top, left, bottom, right]\n        bounding-box values.\n  "
    with open(instances_file, 'r') as annotation_dump:
        data_dict = ast.literal_eval(annotation_dump.readline())
    image_data = collections.OrderedDict()
    if allowlist_file is not None:
        with open(allowlist_file, 'r') as allowlist:
            image_id_allowlist = set([int(x) for x in allowlist.readlines()])
    else:
        image_id_allowlist = [image['id'] for image in data_dict['images']]
    for image_dict in data_dict['images']:
        image_id = image_dict['id']
        if image_id not in image_id_allowlist:
            continue
        image_data_dict = {}
        image_data_dict['id'] = image_dict['id']
        image_data_dict['file_name'] = image_dict['file_name']
        image_data_dict['height'] = image_dict['height']
        image_data_dict['width'] = image_dict['width']
        image_data_dict['detections'] = []
        image_data[image_id] = image_data_dict
    shared_image_ids = set()
    for annotation_dict in data_dict['annotations']:
        image_id = annotation_dict['image_id']
        if image_id in image_data:
            shared_image_ids.add(image_id)
    output_image_ids = sorted(shared_image_ids)
    if num_images:
        if num_images <= 0:
            logging.warning('--num_images is %d, hence outputing all annotated images.', num_images)
        elif num_images > len(shared_image_ids):
            logging.warning('--num_images (%d) is larger than the number of annotated images.', num_images)
        else:
            output_image_ids = output_image_ids[:num_images]
    for image_id in list(image_data):
        if image_id not in output_image_ids:
            del image_data[image_id]
    for annotation_dict in data_dict['annotations']:
        image_id = annotation_dict['image_id']
        if image_id not in output_image_ids:
            continue
        image_data_dict = image_data[image_id]
        bbox = annotation_dict['bbox']
        top = bbox[1]
        left = bbox[0]
        bottom = top + bbox[3]
        right = left + bbox[2]
        if top > image_data_dict['height'] or left > image_data_dict['width'] or bottom > image_data_dict['height'] or (right > image_data_dict['width']):
            continue
        object_d = {}
        object_d['bbox'] = [top / image_data_dict['height'], left / image_data_dict['width'], bottom / image_data_dict['height'], right / image_data_dict['width']]
        object_d['category_id'] = annotation_dict['category_id']
        image_data_dict['detections'].append(object_d)
    return image_data

def _dump_data(ground_truth_detections, images_folder_path, output_folder_path):
    if False:
        i = 10
        return i + 15
    'Dumps images & data from ground-truth objects into output_folder_path.\n\n  The following are created in output_folder_path:\n    images/: sub-folder for allowlisted validation images.\n    ground_truth.pb: A binary proto file containing all ground-truth\n    object-sets.\n\n  Args:\n    ground_truth_detections: A dict mapping image id to ground truth data.\n      Output of _get_ground_truth_detections.\n    images_folder_path: Validation images folder\n    output_folder_path: folder to output files to.\n  '
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    output_images_folder = os.path.join(output_folder_path, 'images')
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    output_proto_file = os.path.join(output_folder_path, 'ground_truth.pb')
    ground_truth_data = evaluation_stages_pb2.ObjectDetectionGroundTruth()
    for image_dict in ground_truth_detections.values():
        detection_result = ground_truth_data.detection_results.add()
        detection_result.image_id = image_dict['id']
        detection_result.image_name = image_dict['file_name']
        for detection_dict in image_dict['detections']:
            object_instance = detection_result.objects.add()
            object_instance.bounding_box.normalized_top = detection_dict['bbox'][0]
            object_instance.bounding_box.normalized_left = detection_dict['bbox'][1]
            object_instance.bounding_box.normalized_bottom = detection_dict['bbox'][2]
            object_instance.bounding_box.normalized_right = detection_dict['bbox'][3]
            object_instance.class_id = detection_dict['category_id']
        shutil.copy2(os.path.join(images_folder_path, image_dict['file_name']), output_images_folder)
    with open(output_proto_file, 'wb') as proto_file:
        proto_file.write(ground_truth_data.SerializeToString())

def _parse_args():
    if False:
        i = 10
        return i + 15
    'Creates a parser that parse the command line arguments.\n\n  Returns:\n    A namespace parsed from command line arguments.\n  '
    parser = argparse.ArgumentParser(description='preprocess_coco_minival: Preprocess COCO minival dataset')
    parser.add_argument('--images_folder', type=str, help='Full path of the validation images folder.', required=True)
    parser.add_argument('--instances_file', type=str, help='Full path of the input JSON file, like instances_val20xx.json.', required=True)
    parser.add_argument('--allowlist_file', type=str, help='File with COCO image ids to preprocess, one on each line.', required=False)
    parser.add_argument('--num_images', type=int, help='Number of allowlisted images to preprocess into the output folder.', required=False)
    parser.add_argument('--output_folder', type=str, help='Full path to output images & text proto files into.', required=True)
    return parser.parse_known_args(args=sys.argv[1:])[0]
if __name__ == '__main__':
    args = _parse_args()
    ground_truths = _get_ground_truth_detections(args.instances_file, args.allowlist_file, args.num_images)
    _dump_data(ground_truths, args.images_folder, args.output_folder)