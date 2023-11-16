"""Run masked LM/next sentence masked_lm pre-training for BERT in tf2.0."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from official.modeling import model_training_utils
from official.nlp import bert_modeling as modeling
from official.nlp import bert_models
from official.nlp import optimization
from official.nlp.bert import common_flags
from official.nlp.bert import input_pipeline
from official.nlp.bert import model_saving_utils
from official.utils.misc import distribution_utils
from official.utils.misc import tpu_lib
flags.DEFINE_string('input_files', None, 'File path to retrieve training data for pre-training.')
flags.DEFINE_integer('max_seq_length', 128, 'The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20, 'Maximum predictions per sequence_output.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_integer('num_steps_per_epoch', 1000, 'Total number of training steps to run per epoch.')
flags.DEFINE_float('warmup_steps', 10000, 'Warmup steps for Adam weight decay optimizer.')
common_flags.define_common_bert_flags()
FLAGS = flags.FLAGS

def get_pretrain_dataset_fn(input_file_pattern, seq_length, max_predictions_per_seq, global_batch_size):
    if False:
        for i in range(10):
            print('nop')
    'Returns input dataset from input file string.'

    def _dataset_fn(ctx=None):
        if False:
            i = 10
            return i + 15
        'Returns tf.data.Dataset for distributed BERT pretraining.'
        input_patterns = input_file_pattern.split(',')
        batch_size = ctx.get_per_replica_batch_size(global_batch_size)
        train_dataset = input_pipeline.create_pretrain_dataset(input_patterns, seq_length, max_predictions_per_seq, batch_size, is_training=True, input_pipeline_context=ctx)
        return train_dataset
    return _dataset_fn

def get_loss_fn(loss_factor=1.0):
    if False:
        print('Hello World!')
    'Returns loss function for BERT pretraining.'

    def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
        if False:
            i = 10
            return i + 15
        return tf.keras.backend.mean(losses) * loss_factor
    return _bert_pretrain_loss_fn

def run_customized_training(strategy, bert_config, max_seq_length, max_predictions_per_seq, model_dir, steps_per_epoch, steps_per_loop, epochs, initial_lr, warmup_steps, input_files, train_batch_size):
    if False:
        return 10
    'Run BERT pretrain model training using low-level API.'
    train_input_fn = get_pretrain_dataset_fn(input_files, max_seq_length, max_predictions_per_seq, train_batch_size)

    def _get_pretrain_model():
        if False:
            print('Hello World!')
        'Gets a pretraining model.'
        (pretrain_model, core_model) = bert_models.pretrain_model(bert_config, max_seq_length, max_predictions_per_seq)
        pretrain_model.optimizer = optimization.create_optimizer(initial_lr, steps_per_epoch * epochs, warmup_steps)
        if FLAGS.fp16_implementation == 'graph_rewrite':
            pretrain_model.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(pretrain_model.optimizer)
        return (pretrain_model, core_model)
    trained_model = model_training_utils.run_customized_training_loop(strategy=strategy, model_fn=_get_pretrain_model, loss_fn=get_loss_fn(loss_factor=1.0 / strategy.num_replicas_in_sync if FLAGS.scale_loss else 1.0), model_dir=model_dir, train_input_fn=train_input_fn, steps_per_epoch=steps_per_epoch, steps_per_loop=steps_per_loop, epochs=epochs, sub_model_export_name='pretrained/bert_model')
    return trained_model

def run_bert_pretrain(strategy):
    if False:
        i = 10
        return i + 15
    'Runs BERT pre-training.'
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if not strategy:
        raise ValueError('Distribution strategy is not specified.')
    logging.info('Training using customized training loop TF 2.0 with distrubutedstrategy.')
    return run_customized_training(strategy, bert_config, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, FLAGS.model_dir, FLAGS.num_steps_per_epoch, FLAGS.steps_per_loop, FLAGS.num_train_epochs, FLAGS.learning_rate, FLAGS.warmup_steps, FLAGS.input_files, FLAGS.train_batch_size)

def main(_):
    if False:
        return 10
    assert tf.version.VERSION.startswith('2.')
    if not FLAGS.model_dir:
        FLAGS.model_dir = '/tmp/bert20/'
    strategy = distribution_utils.get_distribution_strategy(distribution_strategy=FLAGS.distribution_strategy, num_gpus=FLAGS.num_gpus, tpu_address=FLAGS.tpu)
    if strategy:
        print('***** Number of cores used : ', strategy.num_replicas_in_sync)
    run_bert_pretrain(strategy)
if __name__ == '__main__':
    app.run(main)