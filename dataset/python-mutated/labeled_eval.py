"""Generates test Recall@K statistics on labeled classification problems."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from six.moves import xrange
import data_providers
from estimators.get_estimator import get_estimator
from utils import util
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string('config_paths', '', '\n    Path to a YAML configuration files defining FLAG values. Multiple files\n    can be separated by the `#` symbol. Files are merged recursively. Setting\n    a key in these files is equivalent to setting the FLAG value with\n    the same name.\n    ')
tf.flags.DEFINE_string('model_params', '{}', 'YAML configuration string for the model parameters.')
tf.app.flags.DEFINE_string('mode', 'validation', 'Which dataset to evaluate: `validation` | `test`.')
tf.app.flags.DEFINE_string('master', 'local', 'BNS name of the TensorFlow master to use')
tf.app.flags.DEFINE_string('checkpoint_iter', '', 'Evaluate this specific checkpoint.')
tf.app.flags.DEFINE_string('checkpointdir', '/tmp/tcn', 'Path to model checkpoints.')
tf.app.flags.DEFINE_string('outdir', '/tmp/tcn', 'Path to write summaries to.')
FLAGS = tf.app.flags.FLAGS

def nearest_cross_sequence_neighbors(data, tasks, n_neighbors=1):
    if False:
        i = 10
        return i + 15
    'Computes the n_neighbors nearest neighbors for every row in data.\n\n  Args:\n    data: A np.float32 array of shape [num_data, embedding size] holding\n      an embedded validation / test dataset.\n    tasks: A list of strings of size [num_data] holding the task or sequence\n      name that each row belongs to.\n    n_neighbors: The number of knn indices to return for each row.\n  Returns:\n    indices: an np.int32 array of size [num_data, n_neighbors] holding the\n      n_neighbors nearest indices for every row in data. These are\n      restricted to be from different named sequences (as defined in `tasks`).\n  '
    num_data = data.shape[0]
    tasks = np.array(tasks)
    tasks = np.reshape(tasks, (num_data, 1))
    assert len(tasks.shape) == 2
    not_adjacent = tasks != tasks.T
    pdist = pairwise_distances(data, metric='sqeuclidean')
    indices = np.zeros((num_data, n_neighbors), dtype=np.int32)
    for idx in range(num_data):
        distances = [(pdist[idx][i], i) for i in xrange(num_data) if not_adjacent[idx][i]]
        (_, nearest_indices) = zip(*sorted(distances, key=lambda x: x[0])[:n_neighbors])
        indices[idx] = nearest_indices
    return indices

def compute_cross_sequence_recall_at_k(retrieved_labels, labels, k_list):
    if False:
        for i in range(10):
            print('nop')
    'Compute recall@k for a given list of k values.\n\n  Recall is one if an example of the same class is retrieved among the\n    top k nearest neighbors given a query example and zero otherwise.\n    Counting the recall for all examples and averaging the counts returns\n    recall@k score.\n\n  Args:\n    retrieved_labels: 2-D Numpy array of KNN labels for every embedding.\n    labels: 1-D Numpy array of shape [number of data].\n    k_list: List of k values to evaluate recall@k.\n\n  Returns:\n    recall_list: List of recall@k values.\n  '
    kvalue_to_recall = dict(zip(k_list, np.zeros(len(k_list))))
    for k in k_list:
        matches = defaultdict(float)
        counts = defaultdict(float)
        for (i, label_value) in enumerate(labels):
            if label_value in retrieved_labels[i][:k]:
                matches[label_value] += 1.0
            counts[label_value] += 1.0
        kvalue_to_recall[k] = np.mean([matches[l] / counts[l] for l in matches])
    return [kvalue_to_recall[i] for i in k_list]

def compute_cross_sequence_recalls_at_k(embeddings, labels, label_attr_keys, tasks, k_list, summary_writer, training_step):
    if False:
        return 10
    "Computes and reports the recall@k for each classification problem.\n\n  This takes an embedding matrix and an array of multiclass labels\n  with size [num_data, number of classification problems], then\n  computes the average recall@k for each classification problem\n  as well as the average across problems.\n\n  Args:\n    embeddings: A np.float32 array of size [num_data, embedding_size]\n      representing the embedded validation or test dataset.\n    labels: A np.int32 array of size [num_data, num_classification_problems]\n      holding multiclass labels for each embedding for each problem.\n    label_attr_keys: List of strings, holds the names of the classification\n      problems.\n    tasks: A list of strings describing the video sequence each row\n      belongs to. This is used to restrict the recall@k computation\n      to cross-sequence examples.\n    k_list: A list of ints, the k values to evaluate recall@k.\n    summary_writer: A tf.summary.FileWriter.\n    training_step: Int, the current training step we're evaluating.\n  "
    num_data = float(embeddings.shape[0])
    assert labels.shape[0] == num_data
    indices = nearest_cross_sequence_neighbors(embeddings, tasks, n_neighbors=max(k_list))
    retrieved_labels = labels[indices]
    recall_lists = []
    for (idx, label_attr) in enumerate(label_attr_keys):
        problem_labels = labels[:, idx]
        problem_retrieved = retrieved_labels[:, :, idx]
        recall_list = compute_cross_sequence_recall_at_k(retrieved_labels=problem_retrieved, labels=problem_labels, k_list=k_list)
        recall_lists.append(recall_list)
        for (k, recall) in zip(k_list, recall_list):
            recall_error = 1 - recall
            summ = tf.Summary(value=[tf.Summary.Value(tag='validation/classification/%s error@top%d' % (label_attr, k), simple_value=recall_error)])
            print('%s recall@K=%d' % (label_attr, k), recall_error)
            summary_writer.add_summary(summ, int(training_step))
    recall_lists = np.array(recall_lists)
    for i in range(recall_lists.shape[1]):
        average_recall = np.mean(recall_lists[:, i])
        recall_error = 1 - average_recall
        summ = tf.Summary(value=[tf.Summary.Value(tag='validation/classification/average error@top%d' % k_list[i], simple_value=recall_error)])
        print('Average recall@K=%d' % k_list[i], recall_error)
        summary_writer.add_summary(summ, int(training_step))

def evaluate_once(estimator, input_fn_by_view, batch_size, checkpoint_path, label_attr_keys, embedding_size, num_views, k_list):
    if False:
        return 10
    "Compute the recall@k for a given checkpoint path.\n\n  Args:\n    estimator: an `Estimator` object to evaluate.\n    input_fn_by_view: An input_fn to an `Estimator's` predict method. Takes\n      a view index and returns a dict holding ops for getting raw images for\n      the view.\n    batch_size: Int, size of the labeled eval batch.\n    checkpoint_path: String, path to the specific checkpoint being evaluated.\n    label_attr_keys: A list of Strings, holding each attribute name.\n    embedding_size: Int, the size of the embedding.\n    num_views: Int, number of views in the dataset.\n    k_list: List of ints, list of K values to compute recall at K for.\n  "
    feat_matrix = np.zeros((0, embedding_size))
    label_vect = np.zeros((0, len(label_attr_keys)))
    tasks = []
    eval_tensor_keys = ['embeddings', 'tasks', 'classification_labels']
    for view_index in range(num_views):
        predictions = estimator.inference(input_fn_by_view(view_index), checkpoint_path, batch_size, predict_keys=eval_tensor_keys)
        for (i, p) in enumerate(predictions):
            if i % 100 == 0:
                tf.logging.info('Embedded %d images for view %d' % (i, view_index))
            label = p['classification_labels']
            task = p['tasks']
            embedding = p['embeddings']
            feat_matrix = np.append(feat_matrix, [embedding], axis=0)
            label_vect = np.append(label_vect, [label], axis=0)
            tasks.append(task)
    ckpt_step = int(checkpoint_path.split('-')[-1])
    summary_dir = os.path.join(FLAGS.outdir, 'labeled_eval_summaries')
    summary_writer = tf.summary.FileWriter(summary_dir)
    compute_cross_sequence_recalls_at_k(feat_matrix, label_vect, label_attr_keys, tasks, k_list, summary_writer, ckpt_step)

def get_labeled_tables(config):
    if False:
        return 10
    'Gets either labeled test or validation tables, based on flags.'
    mode = FLAGS.mode
    if mode == 'validation':
        labeled_tables = util.GetFilesRecursively(config.data.labeled.validation)
    elif mode == 'test':
        labeled_tables = util.GetFilesRecursively(config.data.labeled.test)
    else:
        raise ValueError('Unknown dataset: %s' % mode)
    return labeled_tables

def main(_):
    if False:
        i = 10
        return i + 15
    'Runs main labeled eval loop.'
    config = util.ParseConfigsToLuaTable(FLAGS.config_paths, FLAGS.model_params)
    checkpointdir = FLAGS.checkpointdir
    estimator = get_estimator(config, checkpointdir)
    image_attr_keys = config.data.labeled.image_attr_keys
    label_attr_keys = config.data.labeled.label_attr_keys
    embedding_size = config.embedding_size
    num_views = config.data.num_views
    k_list = config.val.recall_at_k_list
    batch_size = config.data.batch_size
    labeled_tables = get_labeled_tables(config)

    def input_fn_by_view(view_index):
        if False:
            return 10
        'Returns an input_fn for use with a tf.Estimator by view.'

        def input_fn():
            if False:
                return 10
            (preprocessed_images, labels, tasks) = data_providers.labeled_data_provider(labeled_tables, estimator.preprocess_data, view_index, image_attr_keys, label_attr_keys, batch_size=batch_size)
            return ({'batch_preprocessed': preprocessed_images, 'tasks': tasks, 'classification_labels': labels}, None)
        return input_fn
    if FLAGS.checkpoint_iter:
        checkpoint_path = os.path.join('%s/model.ckpt-%s' % (checkpointdir, FLAGS.checkpoint_iter))
        evaluate_once(estimator, input_fn_by_view, batch_size, checkpoint_path, label_attr_keys, embedding_size, num_views, k_list)
    else:
        for checkpoint_path in tf.contrib.training.checkpoints_iterator(checkpointdir):
            evaluate_once(estimator, input_fn_by_view, batch_size, checkpoint_path, label_attr_keys, embedding_size, num_views, k_list)
if __name__ == '__main__':
    tf.app.run()