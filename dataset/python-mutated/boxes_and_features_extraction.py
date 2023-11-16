"""Library to extract/save boxes and DELF features."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import math
import os
import time
import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow as tf
from google.protobuf import text_format
from delf import delf_config_pb2
from delf import box_io
from delf import feature_io
from delf import detector
from delf import extractor
_BOX_EXTENSION = '.boxes'
_DELF_EXTENSION = '.delf'
_STATUS_CHECK_ITERATIONS = 100
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _PilLoader(path):
    if False:
        print('Hello World!')
    'Helper function to read image with PIL.\n\n  Args:\n    path: Path to image to be loaded.\n\n  Returns:\n    PIL image in RGB format.\n  '
    with tf.gfile.GFile(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def _WriteMappingBasenameToIds(index_names_ids_and_boxes, output_path):
    if False:
        i = 10
        return i + 15
    'Helper function to write CSV mapping from DELF file name to IDs.\n\n  Args:\n    index_names_ids_and_boxes: List containing 3-element lists with name, image\n      ID and box ID.\n    output_path: Output CSV path.\n  '
    with tf.gfile.GFile(output_path, 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=['name', 'index_image_id', 'box_id'])
        csv_writer.writeheader()
        for name_imid_boxid in index_names_ids_and_boxes:
            csv_writer.writerow({'name': name_imid_boxid[0], 'index_image_id': name_imid_boxid[1], 'box_id': name_imid_boxid[2]})

def ExtractBoxesAndFeaturesToFiles(image_names, image_paths, delf_config_path, detector_model_dir, detector_thresh, output_features_dir, output_boxes_dir, output_mapping):
    if False:
        i = 10
        return i + 15
    "Extracts boxes and features, saving them to files.\n\n  Boxes are saved to <image_name>.boxes files. DELF features are extracted for\n  the entire image and saved into <image_name>.delf files. In addition, DELF\n  features are extracted for each high-confidence bounding box in the image, and\n  saved into files named <image_name>_0.delf, <image_name>_1.delf, etc.\n\n  It checks if descriptors/boxes already exist, and skips computation for those.\n\n  Args:\n    image_names: List of image names. These are used to compose output file\n      names for boxes and features.\n    image_paths: List of image paths. image_paths[i] is the path for the image\n      named by image_names[i]. `image_names` and `image_paths` must have the\n      same number of elements.\n    delf_config_path: Path to DelfConfig proto text file.\n    detector_model_dir: Directory where detector SavedModel is located.\n    detector_thresh: Threshold used to decide if an image's detected box\n      undergoes feature extraction.\n    output_features_dir: Directory where DELF features will be written to.\n    output_boxes_dir: Directory where detected boxes will be written to.\n    output_mapping: CSV file which maps each .delf file name to the image ID and\n      detected box ID.\n\n  Raises:\n    ValueError: If len(image_names) and len(image_paths) are different.\n  "
    num_images = len(image_names)
    if len(image_paths) != num_images:
        raise ValueError('image_names and image_paths have different number of items')
    config = delf_config_pb2.DelfConfig()
    with tf.gfile.GFile(delf_config_path, 'r') as f:
        text_format.Merge(f.read(), config)
    if not tf.gfile.Exists(output_features_dir):
        tf.gfile.MakeDirs(output_features_dir)
    if not tf.gfile.Exists(output_boxes_dir):
        tf.gfile.MakeDirs(output_boxes_dir)
    if not tf.gfile.Exists(os.path.dirname(output_mapping)):
        tf.gfile.MakeDirs(os.path.dirname(output_mapping))
    names_ids_and_boxes = []
    with tf.Graph().as_default():
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            detector_fn = detector.MakeDetector(sess, detector_model_dir, import_scope='detector')
            delf_extractor_fn = extractor.MakeExtractor(sess, config, import_scope='extractor_delf')
            start = time.clock()
            for i in range(num_images):
                if i == 0:
                    print('Starting to extract features/boxes...')
                elif i % _STATUS_CHECK_ITERATIONS == 0:
                    elapsed = time.clock() - start
                    print('Processing image %d out of %d, last %d images took %f seconds' % (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
                    start = time.clock()
                image_name = image_names[i]
                output_feature_filename_whole_image = os.path.join(output_features_dir, image_name + _DELF_EXTENSION)
                output_box_filename = os.path.join(output_boxes_dir, image_name + _BOX_EXTENSION)
                pil_im = _PilLoader(image_paths[i])
                (width, height) = pil_im.size
                if tf.gfile.Exists(output_box_filename):
                    print('Skipping box computation for %s' % image_name)
                    (boxes_out, scores_out, class_indices_out) = box_io.ReadFromFile(output_box_filename)
                else:
                    (boxes_out, scores_out, class_indices_out) = detector_fn(np.expand_dims(pil_im, 0))
                    boxes_out = boxes_out[0]
                    scores_out = scores_out[0]
                    class_indices_out = class_indices_out[0]
                    box_io.WriteToFile(output_box_filename, boxes_out, scores_out, class_indices_out)
                num_delf_files = 1
                selected_boxes = []
                for (box_ind, box) in enumerate(boxes_out):
                    if scores_out[box_ind] >= detector_thresh:
                        selected_boxes.append(box)
                num_delf_files += len(selected_boxes)
                for delf_file_ind in range(num_delf_files):
                    if delf_file_ind == 0:
                        box_name = image_name
                        output_feature_filename = output_feature_filename_whole_image
                    else:
                        box_name = image_name + '_' + str(delf_file_ind - 1)
                        output_feature_filename = os.path.join(output_features_dir, box_name + _DELF_EXTENSION)
                    names_ids_and_boxes.append([box_name, i, delf_file_ind - 1])
                    if tf.gfile.Exists(output_feature_filename):
                        print('Skipping DELF computation for %s' % box_name)
                        continue
                    if delf_file_ind >= 1:
                        bbox_for_cropping = selected_boxes[delf_file_ind - 1]
                        bbox_for_cropping_pil_convention = [int(math.floor(bbox_for_cropping[1] * width)), int(math.floor(bbox_for_cropping[0] * height)), int(math.ceil(bbox_for_cropping[3] * width)), int(math.ceil(bbox_for_cropping[2] * height))]
                        pil_cropped_im = pil_im.crop(bbox_for_cropping_pil_convention)
                        im = np.array(pil_cropped_im)
                    else:
                        im = np.array(pil_im)
                    (locations_out, descriptors_out, feature_scales_out, attention_out) = delf_extractor_fn(im)
                    feature_io.WriteToFile(output_feature_filename, locations_out, feature_scales_out, descriptors_out, attention_out)
    _WriteMappingBasenameToIds(names_ids_and_boxes, output_mapping)