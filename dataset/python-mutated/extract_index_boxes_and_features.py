"""Extracts DELF and boxes from the Revisited Oxford/Paris index datasets.

Boxes are saved to <image_name>.boxes files. DELF features are extracted for the
entire image and saved into <image_name>.delf files. In addition, DELF features
are extracted for each high-confidence bounding box in the image, and saved into
files named <image_name>_0.delf, <image_name>_1.delf, etc.

The program checks if descriptors/boxes already exist, and skips computation for
those.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
from tensorflow.python.platform import app
from delf.python.detect_to_retrieve import boxes_and_features_extraction
from delf.python.detect_to_retrieve import dataset
cmd_args = None
_IMAGE_EXTENSION = '.jpg'

def main(argv):
    if False:
        return 10
    if len(argv) > 1:
        raise RuntimeError('Too many command-line arguments.')
    print('Reading list of index images from dataset file...')
    (_, index_list, _) = dataset.ReadDatasetFile(cmd_args.dataset_file_path)
    num_images = len(index_list)
    print('done! Found %d images' % num_images)
    image_paths = [os.path.join(cmd_args.images_dir, index_image_name + _IMAGE_EXTENSION) for index_image_name in index_list]
    boxes_and_features_extraction.ExtractBoxesAndFeaturesToFiles(image_names=index_list, image_paths=image_paths, delf_config_path=cmd_args.delf_config_path, detector_model_dir=cmd_args.detector_model_dir, detector_thresh=cmd_args.detector_thresh, output_features_dir=cmd_args.output_features_dir, output_boxes_dir=cmd_args.output_boxes_dir, output_mapping=cmd_args.output_index_mapping)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--delf_config_path', type=str, default='/tmp/delf_config_example.pbtxt', help='\n      Path to DelfConfig proto text file with configuration to be used for DELF\n      extraction.\n      ')
    parser.add_argument('--detector_model_dir', type=str, default='/tmp/detector_model', help='\n      Directory where detector SavedModel is located.\n      ')
    parser.add_argument('--detector_thresh', type=float, default=0.1, help="\n      Threshold used to decide if an image's detected box undergoes feature\n      extraction. For all detected boxes with detection score larger than this,\n      a .delf file is saved containing the box features. Note that this\n      threshold is used only to select which boxes are used in feature\n      extraction; all detected boxes are actually saved in the .boxes file, even\n      those with score lower than detector_thresh.\n      ")
    parser.add_argument('--dataset_file_path', type=str, default='/tmp/gnd_roxford5k.mat', help='\n      Dataset file for Revisited Oxford or Paris dataset, in .mat format.\n      ')
    parser.add_argument('--images_dir', type=str, default='/tmp/images', help='\n      Directory where dataset images are located, all in .jpg format.\n      ')
    parser.add_argument('--output_boxes_dir', type=str, default='/tmp/boxes', help="\n      Directory where detected boxes will be written to. Each image's boxes\n      will be written to a file with same name, and extension replaced by\n      .boxes.\n      ")
    parser.add_argument('--output_features_dir', type=str, default='/tmp/features', help="\n      Directory where DELF features will be written to. Each image's features\n      will be written to a file with same name, and extension replaced by .delf,\n      eg: <image_name>.delf. In addition, DELF features are extracted for each\n      high-confidence bounding box in the image, and saved into files named\n      <image_name>_0.delf, <image_name>_1.delf, etc.\n      ")
    parser.add_argument('--output_index_mapping', type=str, default='/tmp/index_mapping.csv', help="\n      CSV file which maps each .delf file name to the index image ID and\n      detected box ID. The format is 'name,index_image_id,box_id', including a\n      header. The 'name' refers to the .delf file name without extension.\n\n      For example, a few lines may be like:\n        'radcliffe_camera_000158,2,-1'\n        'radcliffe_camera_000158_0,2,0'\n        'radcliffe_camera_000158_1,2,1'\n        'radcliffe_camera_000158_2,2,2'\n      ")
    (cmd_args, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)