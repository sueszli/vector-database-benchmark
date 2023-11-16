"""Performs image retrieval on Revisited Oxford/Paris datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np
from scipy import spatial
from skimage import measure
from skimage import transform
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import aggregation_config_pb2
from delf import datum_io
from delf import feature_aggregation_similarity
from delf import feature_io
from delf.python.detect_to_retrieve import dataset
cmd_args = None
_VLAD = aggregation_config_pb2.AggregationConfig.VLAD
_ASMK = aggregation_config_pb2.AggregationConfig.ASMK
_ASMK_STAR = aggregation_config_pb2.AggregationConfig.ASMK_STAR
_DELF_EXTENSION = '.delf'
_VLAD_EXTENSION_SUFFIX = 'vlad'
_ASMK_EXTENSION_SUFFIX = 'asmk'
_ASMK_STAR_EXTENSION_SUFFIX = 'asmk_star'
_PR_RANKS = (1, 5, 10)
_STATUS_CHECK_LOAD_ITERATIONS = 50
_STATUS_CHECK_GV_ITERATIONS = 10
_METRICS_FILENAME = 'metrics.txt'
_NUM_TO_RERANK = 100
_FEATURE_DISTANCE_THRESHOLD = 0.9
_NUM_RANSAC_TRIALS = 1000
_MIN_RANSAC_SAMPLES = 3
_RANSAC_RESIDUAL_THRESHOLD = 10

def _ReadAggregatedDescriptors(input_dir, image_list, config):
    if False:
        for i in range(10):
            print('nop')
    'Reads aggregated descriptors.\n\n  Args:\n    input_dir: Directory where aggregated descriptors are located.\n    image_list: List of image names for which to load descriptors.\n    config: AggregationConfig used for images.\n\n  Returns:\n    aggregated_descriptors: List containing #images items, each a 1D NumPy\n      array.\n    visual_words: If using VLAD aggregation, returns an empty list. Otherwise,\n      returns a list containing #images items, each a 1D NumPy array.\n  '
    extension = '.'
    if config.use_regional_aggregation:
        extension += 'r'
    if config.aggregation_type == _VLAD:
        extension += _VLAD_EXTENSION_SUFFIX
    elif config.aggregation_type == _ASMK:
        extension += _ASMK_EXTENSION_SUFFIX
    elif config.aggregation_type == _ASMK_STAR:
        extension += _ASMK_STAR_EXTENSION_SUFFIX
    else:
        raise ValueError('Invalid aggregation type: %d' % config.aggregation_type)
    num_images = len(image_list)
    aggregated_descriptors = []
    visual_words = []
    print('Starting to collect descriptors for %d images...' % num_images)
    start = time.clock()
    for i in range(num_images):
        if i > 0 and i % _STATUS_CHECK_LOAD_ITERATIONS == 0:
            elapsed = time.clock() - start
            print('Reading descriptors for image %d out of %d, last %d images took %f seconds' % (i, num_images, _STATUS_CHECK_LOAD_ITERATIONS, elapsed))
            start = time.clock()
        descriptors_filename = image_list[i] + extension
        descriptors_fullpath = os.path.join(input_dir, descriptors_filename)
        if config.aggregation_type == _VLAD:
            aggregated_descriptors.append(datum_io.ReadFromFile(descriptors_fullpath))
        else:
            (d, v) = datum_io.ReadPairFromFile(descriptors_fullpath)
            if config.aggregation_type == _ASMK_STAR:
                d = d.astype('uint8')
            aggregated_descriptors.append(d)
            visual_words.append(v)
    return (aggregated_descriptors, visual_words)

def _MatchFeatures(query_locations, query_descriptors, index_image_locations, index_image_descriptors):
    if False:
        for i in range(10):
            print('nop')
    'Matches local features using geometric verification.\n\n  First, finds putative local feature matches by matching `query_descriptors`\n  against a KD-tree from the `index_image_descriptors`. Then, attempts to fit an\n  affine transformation between the putative feature corresponces using their\n  locations.\n\n  Args:\n    query_locations: Locations of local features for query image. NumPy array of\n      shape [#query_features, 2].\n    query_descriptors: Descriptors of local features for query image. NumPy\n      array of shape [#query_features, depth].\n    index_image_locations: Locations of local features for index image. NumPy\n      array of shape [#index_image_features, 2].\n    index_image_descriptors: Descriptors of local features for index image.\n      NumPy array of shape [#index_image_features, depth].\n\n  Returns:\n    score: Number of inliers of match. If no match is found, returns 0.\n  '
    num_features_query = query_locations.shape[0]
    num_features_index_image = index_image_locations.shape[0]
    if not num_features_query or not num_features_index_image:
        return 0
    index_image_tree = spatial.cKDTree(index_image_descriptors)
    (_, indices) = index_image_tree.query(query_descriptors, distance_upper_bound=_FEATURE_DISTANCE_THRESHOLD)
    query_locations_to_use = np.array([query_locations[i,] for i in range(num_features_query) if indices[i] != num_features_index_image])
    index_image_locations_to_use = np.array([index_image_locations[indices[i],] for i in range(num_features_query) if indices[i] != num_features_index_image])
    if not query_locations_to_use.shape[0]:
        return 0
    (_, inliers) = measure.ransac((index_image_locations_to_use, query_locations_to_use), transform.AffineTransform, min_samples=_MIN_RANSAC_SAMPLES, residual_threshold=_RANSAC_RESIDUAL_THRESHOLD, max_trials=_NUM_RANSAC_TRIALS)
    if inliers is None:
        inliers = []
    return sum(inliers)

def _RerankByGeometricVerification(input_ranks, initial_scores, query_name, index_names, query_features_dir, index_features_dir, junk_ids):
    if False:
        while True:
            i = 10
    'Re-ranks retrieval results using geometric verification.\n\n  Args:\n    input_ranks: 1D NumPy array with indices of top-ranked index images, sorted\n      from the most to the least similar.\n    initial_scores: 1D NumPy array with initial similarity scores between query\n      and index images. Entry i corresponds to score for image i.\n    query_name: Name for query image (string).\n    index_names: List of names for index images (strings).\n    query_features_dir: Directory where query local feature file is located\n      (string).\n    index_features_dir: Directory where index local feature files are located\n      (string).\n    junk_ids: Set with indices of junk images which should not be considered\n      during re-ranking.\n\n  Returns:\n    output_ranks: 1D NumPy array with index image indices, sorted from the most\n      to the least similar according to the geometric verification and initial\n      scores.\n\n  Raises:\n    ValueError: If `input_ranks`, `initial_scores` and `index_names` do not have\n      the same number of entries.\n  '
    num_index_images = len(index_names)
    if len(input_ranks) != num_index_images:
        raise ValueError('input_ranks and index_names have different number of elements: %d vs %d' % (len(input_ranks), len(index_names)))
    if len(initial_scores) != num_index_images:
        raise ValueError('initial_scores and index_names have different number of elements: %d vs %d' % (len(initial_scores), len(index_names)))
    input_ranks_for_gv = []
    for ind in input_ranks:
        if ind not in junk_ids:
            input_ranks_for_gv.append(ind)
    num_to_rerank = min(_NUM_TO_RERANK, len(input_ranks_for_gv))
    query_features_path = os.path.join(query_features_dir, query_name + _DELF_EXTENSION)
    (query_locations, _, query_descriptors, _, _) = feature_io.ReadFromFile(query_features_path)
    inliers_and_initial_scores = []
    for i in range(num_index_images):
        inliers_and_initial_scores.append([0, initial_scores[i]])
    print('Starting to re-rank')
    for i in range(num_to_rerank):
        if i > 0 and i % _STATUS_CHECK_GV_ITERATIONS == 0:
            print('Re-ranking: i = %d out of %d' % (i, num_to_rerank))
        index_image_id = input_ranks_for_gv[i]
        index_image_features_path = os.path.join(index_features_dir, index_names[index_image_id] + _DELF_EXTENSION)
        (index_image_locations, _, index_image_descriptors, _, _) = feature_io.ReadFromFile(index_image_features_path)
        inliers_and_initial_scores[index_image_id][0] = _MatchFeatures(query_locations, query_descriptors, index_image_locations, index_image_descriptors)

    def _InliersInitialScoresSorting(k):
        if False:
            i = 10
            return i + 15
        'Helper function to sort list based on two entries.\n\n    Args:\n      k: Index into `inliers_and_initial_scores`.\n\n    Returns:\n      Tuple containing inlier score and initial score.\n    '
        return (inliers_and_initial_scores[k][0], inliers_and_initial_scores[k][1])
    output_ranks = sorted(range(num_index_images), key=_InliersInitialScoresSorting, reverse=True)
    return output_ranks

def _SaveMetricsFile(mean_average_precision, mean_precisions, mean_recalls, pr_ranks, output_path):
    if False:
        for i in range(10):
            print('nop')
    'Saves aggregated retrieval metrics to text file.\n\n  Args:\n    mean_average_precision: Dict mapping each dataset protocol to a float.\n    mean_precisions: Dict mapping each dataset protocol to a NumPy array of\n      floats with shape [len(pr_ranks)].\n    mean_recalls: Dict mapping each dataset protocol to a NumPy array of floats\n      with shape [len(pr_ranks)].\n    pr_ranks: List of integers.\n    output_path: Full file path.\n  '
    with tf.gfile.GFile(output_path, 'w') as f:
        for k in sorted(mean_average_precision.keys()):
            f.write('{}\n  mAP={}\n  mP@k{} {}\n  mR@k{} {}\n'.format(k, np.around(mean_average_precision[k] * 100, decimals=2), np.array(pr_ranks), np.around(mean_precisions[k] * 100, decimals=2), np.array(pr_ranks), np.around(mean_recalls[k] * 100, decimals=2)))

def main(argv):
    if False:
        i = 10
        return i + 15
    if len(argv) > 1:
        raise RuntimeError('Too many command-line arguments.')
    print('Parsing dataset...')
    (query_list, index_list, ground_truth) = dataset.ReadDatasetFile(cmd_args.dataset_file_path)
    num_query_images = len(query_list)
    num_index_images = len(index_list)
    (_, medium_ground_truth, hard_ground_truth) = dataset.ParseEasyMediumHardGroundTruth(ground_truth)
    print('done! Found %d queries and %d index images' % (num_query_images, num_index_images))
    query_config = aggregation_config_pb2.AggregationConfig()
    with tf.gfile.GFile(cmd_args.query_aggregation_config_path, 'r') as f:
        text_format.Merge(f.read(), query_config)
    index_config = aggregation_config_pb2.AggregationConfig()
    with tf.gfile.GFile(cmd_args.index_aggregation_config_path, 'r') as f:
        text_format.Merge(f.read(), index_config)
    (query_aggregated_descriptors, query_visual_words) = _ReadAggregatedDescriptors(cmd_args.query_aggregation_dir, query_list, query_config)
    (index_aggregated_descriptors, index_visual_words) = _ReadAggregatedDescriptors(cmd_args.index_aggregation_dir, index_list, index_config)
    similarity_computer = feature_aggregation_similarity.SimilarityAggregatedRepresentation(index_config)
    ranks_before_gv = np.zeros([num_query_images, num_index_images], dtype='int32')
    if cmd_args.use_geometric_verification:
        medium_ranks_after_gv = np.zeros([num_query_images, num_index_images], dtype='int32')
        hard_ranks_after_gv = np.zeros([num_query_images, num_index_images], dtype='int32')
    for i in range(num_query_images):
        print('Performing retrieval with query %d (%s)...' % (i, query_list[i]))
        start = time.clock()
        similarities = np.zeros([num_index_images])
        for j in range(num_index_images):
            similarities[j] = similarity_computer.ComputeSimilarity(query_aggregated_descriptors[i], index_aggregated_descriptors[j], query_visual_words[i], index_visual_words[j])
        ranks_before_gv[i] = np.argsort(-similarities)
        if cmd_args.use_geometric_verification:
            medium_ranks_after_gv[i] = _RerankByGeometricVerification(ranks_before_gv[i], similarities, query_list[i], index_list, cmd_args.query_features_dir, cmd_args.index_features_dir, set(medium_ground_truth[i]['junk']))
            hard_ranks_after_gv[i] = _RerankByGeometricVerification(ranks_before_gv[i], similarities, query_list[i], index_list, cmd_args.query_features_dir, cmd_args.index_features_dir, set(hard_ground_truth[i]['junk']))
        elapsed = time.clock() - start
        print('done! Retrieval for query %d took %f seconds' % (i, elapsed))
    if not tf.gfile.Exists(cmd_args.output_dir):
        tf.gfile.MakeDirs(cmd_args.output_dir)
    medium_metrics = dataset.ComputeMetrics(ranks_before_gv, medium_ground_truth, _PR_RANKS)
    hard_metrics = dataset.ComputeMetrics(ranks_before_gv, hard_ground_truth, _PR_RANKS)
    if cmd_args.use_geometric_verification:
        medium_metrics_after_gv = dataset.ComputeMetrics(medium_ranks_after_gv, medium_ground_truth, _PR_RANKS)
        hard_metrics_after_gv = dataset.ComputeMetrics(hard_ranks_after_gv, hard_ground_truth, _PR_RANKS)
    mean_average_precision_dict = {'medium': medium_metrics[0], 'hard': hard_metrics[0]}
    mean_precisions_dict = {'medium': medium_metrics[1], 'hard': hard_metrics[1]}
    mean_recalls_dict = {'medium': medium_metrics[2], 'hard': hard_metrics[2]}
    if cmd_args.use_geometric_verification:
        mean_average_precision_dict.update({'medium_after_gv': medium_metrics_after_gv[0], 'hard_after_gv': hard_metrics_after_gv[0]})
        mean_precisions_dict.update({'medium_after_gv': medium_metrics_after_gv[1], 'hard_after_gv': hard_metrics_after_gv[1]})
        mean_recalls_dict.update({'medium_after_gv': medium_metrics_after_gv[2], 'hard_after_gv': hard_metrics_after_gv[2]})
    _SaveMetricsFile(mean_average_precision_dict, mean_precisions_dict, mean_recalls_dict, _PR_RANKS, os.path.join(cmd_args.output_dir, _METRICS_FILENAME))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--index_aggregation_config_path', type=str, default='/tmp/index_aggregation_config.pbtxt', help='\n      Path to index AggregationConfig proto text file. This is used to load the\n      aggregated descriptors from the index, and to define the parameters used\n      in computing similarity for aggregated descriptors.\n      ')
    parser.add_argument('--query_aggregation_config_path', type=str, default='/tmp/query_aggregation_config.pbtxt', help='\n      Path to query AggregationConfig proto text file. This is only used to load\n      the aggregated descriptors for the queries.\n      ')
    parser.add_argument('--dataset_file_path', type=str, default='/tmp/gnd_roxford5k.mat', help='\n      Dataset file for Revisited Oxford or Paris dataset, in .mat format.\n      ')
    parser.add_argument('--index_aggregation_dir', type=str, default='/tmp/index_aggregation', help='\n      Directory where index aggregated descriptors are located.\n      ')
    parser.add_argument('--query_aggregation_dir', type=str, default='/tmp/query_aggregation', help='\n      Directory where query aggregated descriptors are located.\n      ')
    parser.add_argument('--use_geometric_verification', type=lambda x: str(x).lower() == 'true', default=False, help='\n      If True, performs re-ranking using local feature-based geometric\n      verification.\n      ')
    parser.add_argument('--index_features_dir', type=str, default='/tmp/index_features', help='\n      Only used if `use_geometric_verification` is True.\n      Directory where index local image features are located, all in .delf\n      format.\n      ')
    parser.add_argument('--query_features_dir', type=str, default='/tmp/query_features', help='\n      Only used if `use_geometric_verification` is True.\n      Directory where query local image features are located, all in .delf\n      format.\n      ')
    parser.add_argument('--output_dir', type=str, default='/tmp/retrieval', help='\n      Directory where retrieval output will be written to. A file containing\n      metrics for this run is saved therein, with file name "metrics.txt".\n      ')
    (cmd_args, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)