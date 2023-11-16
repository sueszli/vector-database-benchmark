"""Executes BERT benchmarks and accuracy tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import json
import math
import os
import time
from absl import flags
from absl.testing import flagsaver
import tensorflow as tf
from official.benchmark import bert_benchmark_utils as benchmark_utils
from official.nlp import bert_modeling as modeling
from official.nlp.bert import input_pipeline
from official.nlp.bert import run_classifier
from official.utils.misc import distribution_utils
PRETRAINED_CHECKPOINT_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/bert_model.ckpt'
CLASSIFIER_TRAIN_DATA_PATH = 'gs://tf-perfzero-data/bert/classification/mrpc_train.tf_record'
CLASSIFIER_EVAL_DATA_PATH = 'gs://tf-perfzero-data/bert/classification/mrpc_eval.tf_record'
CLASSIFIER_INPUT_META_DATA_PATH = 'gs://tf-perfzero-data/bert/classification/mrpc_meta_data'
MODEL_CONFIG_FILE_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/bert_config.json'
TMP_DIR = os.getenv('TMPDIR')
FLAGS = flags.FLAGS

class BertClassifyBenchmarkBase(benchmark_utils.BertBenchmarkBase):
    """Base class to hold methods common to test classes in the module."""

    def __init__(self, output_dir=None):
        if False:
            for i in range(10):
                print('nop')
        super(BertClassifyBenchmarkBase, self).__init__(output_dir)
        self.num_epochs = None
        self.num_steps_per_epoch = None

    @flagsaver.flagsaver
    def _run_bert_classifier(self, callbacks=None, use_ds=True):
        if False:
            return 10
        'Starts BERT classification task.'
        with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
            input_meta_data = json.loads(reader.read().decode('utf-8'))
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        epochs = self.num_epochs if self.num_epochs else FLAGS.num_train_epochs
        if self.num_steps_per_epoch:
            steps_per_epoch = self.num_steps_per_epoch
        else:
            train_data_size = input_meta_data['train_data_size']
            steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)
        warmup_steps = int(epochs * steps_per_epoch * 0.1)
        eval_steps = int(math.ceil(input_meta_data['eval_data_size'] / FLAGS.eval_batch_size))
        strategy = distribution_utils.get_distribution_strategy(distribution_strategy='mirrored' if use_ds else 'off', num_gpus=self.num_gpus)
        steps_per_loop = 1
        max_seq_length = input_meta_data['max_seq_length']
        train_input_fn = functools.partial(input_pipeline.create_classifier_dataset, FLAGS.train_data_path, seq_length=max_seq_length, batch_size=FLAGS.train_batch_size)
        eval_input_fn = functools.partial(input_pipeline.create_classifier_dataset, FLAGS.eval_data_path, seq_length=max_seq_length, batch_size=FLAGS.eval_batch_size, is_training=False, drop_remainder=False)
        run_classifier.run_bert_classifier(strategy, bert_config, input_meta_data, FLAGS.model_dir, epochs, steps_per_epoch, steps_per_loop, eval_steps, warmup_steps, FLAGS.learning_rate, FLAGS.init_checkpoint, train_input_fn, eval_input_fn, custom_callbacks=callbacks)

class BertClassifyBenchmarkReal(BertClassifyBenchmarkBase):
    """Short benchmark performance tests for BERT model.

  Tests BERT classification performance in different GPU configurations.
  The naming convention of below test cases follow
  `benchmark_(number of gpus)_gpu_(dataset type)` format.
  """

    def __init__(self, output_dir=TMP_DIR, **kwargs):
        if False:
            return 10
        super(BertClassifyBenchmarkReal, self).__init__(output_dir=output_dir)
        self.train_data_path = CLASSIFIER_TRAIN_DATA_PATH
        self.eval_data_path = CLASSIFIER_EVAL_DATA_PATH
        self.bert_config_file = MODEL_CONFIG_FILE_PATH
        self.input_meta_data_path = CLASSIFIER_INPUT_META_DATA_PATH
        self.num_steps_per_epoch = 110
        self.num_epochs = 1

    def _run_and_report_benchmark(self, training_summary_path, min_accuracy=0, max_accuracy=1, use_ds=True):
        if False:
            i = 10
            return i + 15
        'Starts BERT performance benchmark test.'
        start_time_sec = time.time()
        self._run_bert_classifier(callbacks=[self.timer_callback], use_ds=use_ds)
        wall_time_sec = time.time() - start_time_sec
        with tf.io.gfile.GFile(training_summary_path, 'rb') as reader:
            summary = json.loads(reader.read().decode('utf-8'))
        summary.pop('eval_metrics', None)
        super(BertClassifyBenchmarkReal, self)._report_benchmark(stats=summary, wall_time_sec=wall_time_sec, min_accuracy=min_accuracy, max_accuracy=max_accuracy)

    def benchmark_1_gpu_mrpc(self):
        if False:
            for i in range(10):
                print('nop')
        'Test BERT model performance with 1 GPU.'
        self._setup()
        self.num_gpus = 1
        FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_mrpc')
        FLAGS.train_data_path = self.train_data_path
        FLAGS.eval_data_path = self.eval_data_path
        FLAGS.input_meta_data_path = self.input_meta_data_path
        FLAGS.bert_config_file = self.bert_config_file
        FLAGS.train_batch_size = 4
        FLAGS.eval_batch_size = 4
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path)

    def benchmark_1_gpu_mrpc_xla(self):
        if False:
            for i in range(10):
                print('nop')
        'Test BERT model performance with 1 GPU.'
        self._setup()
        self.num_gpus = 1
        FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_mrpc_xla')
        FLAGS.train_data_path = self.train_data_path
        FLAGS.eval_data_path = self.eval_data_path
        FLAGS.input_meta_data_path = self.input_meta_data_path
        FLAGS.bert_config_file = self.bert_config_file
        FLAGS.train_batch_size = 4
        FLAGS.eval_batch_size = 4
        FLAGS.enable_xla = True
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path)

    def benchmark_1_gpu_mrpc_no_dist_strat(self):
        if False:
            i = 10
            return i + 15
        'Test BERT model performance with 1 GPU, no distribution strategy.'
        self._setup()
        self.num_gpus = 1
        FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_mrpc_no_dist_strat')
        FLAGS.train_data_path = self.train_data_path
        FLAGS.eval_data_path = self.eval_data_path
        FLAGS.input_meta_data_path = self.input_meta_data_path
        FLAGS.bert_config_file = self.bert_config_file
        FLAGS.train_batch_size = 4
        FLAGS.eval_batch_size = 4
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path, use_ds=False)

    def benchmark_2_gpu_mrpc(self):
        if False:
            while True:
                i = 10
        'Test BERT model performance with 2 GPUs.'
        self._setup()
        self.num_gpus = 2
        FLAGS.model_dir = self._get_model_dir('benchmark_2_gpu_mrpc')
        FLAGS.train_data_path = self.train_data_path
        FLAGS.eval_data_path = self.eval_data_path
        FLAGS.input_meta_data_path = self.input_meta_data_path
        FLAGS.bert_config_file = self.bert_config_file
        FLAGS.train_batch_size = 8
        FLAGS.eval_batch_size = 8
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path)

    def benchmark_4_gpu_mrpc(self):
        if False:
            while True:
                i = 10
        'Test BERT model performance with 4 GPUs.'
        self._setup()
        self.num_gpus = 4
        FLAGS.model_dir = self._get_model_dir('benchmark_4_gpu_mrpc')
        FLAGS.train_data_path = self.train_data_path
        FLAGS.eval_data_path = self.eval_data_path
        FLAGS.input_meta_data_path = self.input_meta_data_path
        FLAGS.bert_config_file = self.bert_config_file
        FLAGS.train_batch_size = 16
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path)

    def benchmark_8_gpu_mrpc(self):
        if False:
            i = 10
            return i + 15
        'Test BERT model performance with 8 GPUs.'
        self._setup()
        FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_mrpc')
        FLAGS.train_data_path = self.train_data_path
        FLAGS.eval_data_path = self.eval_data_path
        FLAGS.input_meta_data_path = self.input_meta_data_path
        FLAGS.bert_config_file = self.bert_config_file
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path)

    def benchmark_1_gpu_amp_mrpc_no_dist_strat(self):
        if False:
            print('Hello World!')
        'Performance for 1 GPU no DS with automatic mixed precision.'
        self._setup()
        self.num_gpus = 1
        FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_amp_mrpc_no_dist_strat')
        FLAGS.train_data_path = self.train_data_path
        FLAGS.eval_data_path = self.eval_data_path
        FLAGS.input_meta_data_path = self.input_meta_data_path
        FLAGS.bert_config_file = self.bert_config_file
        FLAGS.train_batch_size = 4
        FLAGS.eval_batch_size = 4
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path, use_ds=False)

    def benchmark_8_gpu_amp_mrpc(self):
        if False:
            for i in range(10):
                print('nop')
        'Test BERT model performance with 8 GPUs with automatic mixed precision.\n    '
        self._setup()
        self.num_gpus = 8
        FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_amp_mrpc')
        FLAGS.train_data_path = self.train_data_path
        FLAGS.eval_data_path = self.eval_data_path
        FLAGS.input_meta_data_path = self.input_meta_data_path
        FLAGS.bert_config_file = self.bert_config_file
        FLAGS.train_batch_size = 32
        FLAGS.eval_batch_size = 32
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path, use_ds=False)

class BertClassifyAccuracy(BertClassifyBenchmarkBase):
    """Short accuracy test for BERT model.

  Tests BERT classification task model accuracy. The naming
  convention of below test cases follow
  `benchmark_(number of gpus)_gpu_(dataset type)` format.
  """

    def __init__(self, output_dir=TMP_DIR, **kwargs):
        if False:
            return 10
        self.train_data_path = CLASSIFIER_TRAIN_DATA_PATH
        self.eval_data_path = CLASSIFIER_EVAL_DATA_PATH
        self.bert_config_file = MODEL_CONFIG_FILE_PATH
        self.input_meta_data_path = CLASSIFIER_INPUT_META_DATA_PATH
        self.pretrained_checkpoint_path = PRETRAINED_CHECKPOINT_PATH
        super(BertClassifyAccuracy, self).__init__(output_dir=output_dir)

    def _run_and_report_benchmark(self, training_summary_path, min_accuracy=0.84, max_accuracy=0.88):
        if False:
            for i in range(10):
                print('nop')
        'Starts BERT accuracy benchmark test.'
        start_time_sec = time.time()
        self._run_bert_classifier(callbacks=[self.timer_callback])
        wall_time_sec = time.time() - start_time_sec
        with tf.io.gfile.GFile(training_summary_path, 'rb') as reader:
            summary = json.loads(reader.read().decode('utf-8'))
        super(BertClassifyAccuracy, self)._report_benchmark(stats=summary, wall_time_sec=wall_time_sec, min_accuracy=min_accuracy, max_accuracy=max_accuracy)

    def _setup(self):
        if False:
            while True:
                i = 10
        super(BertClassifyAccuracy, self)._setup()
        FLAGS.train_data_path = self.train_data_path
        FLAGS.eval_data_path = self.eval_data_path
        FLAGS.input_meta_data_path = self.input_meta_data_path
        FLAGS.bert_config_file = self.bert_config_file
        FLAGS.init_checkpoint = self.pretrained_checkpoint_path

    def benchmark_8_gpu_mrpc(self):
        if False:
            while True:
                i = 10
        'Run BERT model accuracy test with 8 GPUs.\n\n    Due to comparatively small cardinality of  MRPC dataset, training\n    accuracy metric has high variance between trainings. As so, we\n    set the wide range of allowed accuracy (84% to 88%).\n    '
        self._setup()
        FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_mrpc')
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path)

    def benchmark_8_gpu_mrpc_xla(self):
        if False:
            while True:
                i = 10
        'Run BERT model accuracy test with 8 GPUs with XLA.'
        self._setup()
        FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_mrpc_xla')
        FLAGS.enable_xla = True
        summary_path = os.path.join(FLAGS.model_dir, 'summaries/training_summary.txt')
        self._run_and_report_benchmark(summary_path)
if __name__ == '__main__':
    tf.test.main()