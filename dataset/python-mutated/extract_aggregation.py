"""Extracts aggregation for images from Revisited Oxford/Paris datasets.

The program checks if the aggregated representation for an image already exists,
and skips computation for those.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from tensorflow.python.platform import app
from delf.python.detect_to_retrieve import aggregation_extraction
from delf.python.detect_to_retrieve import dataset
cmd_args = None

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    if len(argv) > 1:
        raise RuntimeError('Too many command-line arguments.')
    print('Reading list of images from dataset file...')
    (query_list, index_list, _) = dataset.ReadDatasetFile(cmd_args.dataset_file_path)
    if cmd_args.use_query_images:
        image_list = query_list
    else:
        image_list = index_list
    num_images = len(image_list)
    print('done! Found %d images' % num_images)
    aggregation_extraction.ExtractAggregatedRepresentationsToFiles(image_names=image_list, features_dir=cmd_args.features_dir, aggregation_config_path=cmd_args.aggregation_config_path, mapping_path=cmd_args.index_mapping_path, output_aggregation_dir=cmd_args.output_aggregation_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--aggregation_config_path', type=str, default='/tmp/aggregation_config.pbtxt', help='\n      Path to AggregationConfig proto text file with configuration to be used\n      for extraction.\n      ')
    parser.add_argument('--dataset_file_path', type=str, default='/tmp/gnd_roxford5k.mat', help='\n      Dataset file for Revisited Oxford or Paris dataset, in .mat format.\n      ')
    parser.add_argument('--use_query_images', type=lambda x: str(x).lower() == 'true', default=False, help='\n      If True, processes the query images of the dataset. If False, processes\n      the database (ie, index) images.\n      ')
    parser.add_argument('--features_dir', type=str, default='/tmp/features', help='\n      Directory where image features are located, all in .delf format.\n      ')
    parser.add_argument('--index_mapping_path', type=str, default='', help='\n      Optional CSV file which maps each .delf file name to the index image ID\n      and detected box ID. If regional aggregation is performed, this should be\n      set. Otherwise, this is ignored.\n      Usually this file is obtained as an output from the\n      `extract_index_boxes_and_features.py` script.\n      ')
    parser.add_argument('--output_aggregation_dir', type=str, default='/tmp/aggregation', help="\n      Directory where aggregation output will be written to. Each image's\n      features will be written to a file with same name, and extension replaced\n      by one of\n      ['.vlad', '.asmk', '.asmk_star', '.rvlad', '.rasmk', '.rasmk_star'].\n      ")
    (cmd_args, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)