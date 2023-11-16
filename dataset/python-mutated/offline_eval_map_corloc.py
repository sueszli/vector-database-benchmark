"""Evaluation executable for detection data.

This executable evaluates precomputed detections produced by a detection
model and writes the evaluation results into csv file metrics.csv, stored
in the directory, specified by --eval_dir.

The evaluation metrics set is supplied in object_detection.protos.EvalConfig
in metrics_set field.
Currently two set of metrics are supported:
- pascal_voc_metrics: standard PASCAL VOC 2007 metric
- open_images_detection_metrics: Open Image V2 metric
All other field of object_detection.protos.EvalConfig are ignored.

Example usage:
    ./compute_metrics \\
        --eval_dir=path/to/eval_dir \\
        --eval_config_path=path/to/evaluation/configuration/file \\
        --input_config_path=path/to/input/configuration/file
"""
import csv
import os
import re
import tensorflow as tf
from object_detection.core import standard_fields
from object_detection.legacy import evaluator
from object_detection.metrics import tf_example_parser
from object_detection.utils import config_util
from object_detection.utils import label_map_util
flags = tf.app.flags
tf.logging.set_verbosity(tf.logging.INFO)
flags.DEFINE_string('eval_dir', None, 'Directory to write eval summaries to.')
flags.DEFINE_string('eval_config_path', None, 'Path to an eval_pb2.EvalConfig config file.')
flags.DEFINE_string('input_config_path', None, 'Path to an eval_pb2.InputConfig config file.')
FLAGS = flags.FLAGS

def _generate_sharded_filenames(filename):
    if False:
        return 10
    m = re.search('@(\\d{1,})', filename)
    if m:
        num_shards = int(m.group(1))
        return [re.sub('@(\\d{1,})', '-%.5d-of-%.5d' % (i, num_shards), filename) for i in range(num_shards)]
    else:
        return [filename]

def _generate_filenames(filenames):
    if False:
        print('Hello World!')
    result = []
    for filename in filenames:
        result += _generate_sharded_filenames(filename)
    return result

def read_data_and_evaluate(input_config, eval_config):
    if False:
        while True:
            i = 10
    'Reads pre-computed object detections and groundtruth from tf_record.\n\n  Args:\n    input_config: input config proto of type\n      object_detection.protos.InputReader.\n    eval_config: evaluation config proto of type\n      object_detection.protos.EvalConfig.\n\n  Returns:\n    Evaluated detections metrics.\n\n  Raises:\n    ValueError: if input_reader type is not supported or metric type is unknown.\n  '
    if input_config.WhichOneof('input_reader') == 'tf_record_input_reader':
        input_paths = input_config.tf_record_input_reader.input_path
        categories = label_map_util.create_categories_from_labelmap(input_config.label_map_path)
        object_detection_evaluators = evaluator.get_evaluators(eval_config, categories)
        object_detection_evaluator = object_detection_evaluators[0]
        skipped_images = 0
        processed_images = 0
        for input_path in _generate_filenames(input_paths):
            tf.logging.info('Processing file: {0}'.format(input_path))
            record_iterator = tf.python_io.tf_record_iterator(path=input_path)
            data_parser = tf_example_parser.TfExampleDetectionAndGTParser()
            for string_record in record_iterator:
                tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 1000, processed_images)
                processed_images += 1
                example = tf.train.Example()
                example.ParseFromString(string_record)
                decoded_dict = data_parser.parse(example)
                if decoded_dict:
                    object_detection_evaluator.add_single_ground_truth_image_info(decoded_dict[standard_fields.DetectionResultFields.key], decoded_dict)
                    object_detection_evaluator.add_single_detected_image_info(decoded_dict[standard_fields.DetectionResultFields.key], decoded_dict)
                else:
                    skipped_images += 1
                    tf.logging.info('Skipped images: {0}'.format(skipped_images))
        return object_detection_evaluator.evaluate()
    raise ValueError('Unsupported input_reader_config.')

def write_metrics(metrics, output_dir):
    if False:
        return 10
    'Write metrics to the output directory.\n\n  Args:\n    metrics: A dictionary containing metric names and values.\n    output_dir: Directory to write metrics to.\n  '
    tf.logging.info('Writing metrics.')
    with open(os.path.join(output_dir, 'metrics.csv'), 'w') as csvfile:
        metrics_writer = csv.writer(csvfile, delimiter=',')
        for (metric_name, metric_value) in metrics.items():
            metrics_writer.writerow([metric_name, str(metric_value)])

def main(argv):
    if False:
        return 10
    del argv
    required_flags = ['input_config_path', 'eval_config_path', 'eval_dir']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))
    configs = config_util.get_configs_from_multiple_files(eval_input_config_path=FLAGS.input_config_path, eval_config_path=FLAGS.eval_config_path)
    eval_config = configs['eval_config']
    input_config = configs['eval_input_config']
    metrics = read_data_and_evaluate(input_config, eval_config)
    write_metrics(metrics, FLAGS.eval_dir)
if __name__ == '__main__':
    tf.app.run(main)