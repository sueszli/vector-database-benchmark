"""Run BERT on SQuAD 1.1 and SQuAD 2.0 in tf2.0."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
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
from official.nlp.bert import squad_lib
from official.nlp.bert import tokenization
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import tpu_lib
flags.DEFINE_enum('mode', 'train_and_predict', ['train_and_predict', 'train', 'predict', 'export_only'], 'One of {"train_and_predict", "train", "predict", "export_only"}. `train_and_predict`: both train and predict to a json file. `train`: only trains the model. `predict`: predict answers from the squad json file. `export_only`: will take the latest checkpoint inside model_dir and export a `SavedModel`.')
flags.DEFINE_string('train_data_path', '', 'Training data path with train tfrecords.')
flags.DEFINE_string('input_meta_data_path', None, 'Path to file that contains meta data about input to be used for training and evaluation.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_string('predict_file', None, 'Prediction data path with train tfrecords.')
flags.DEFINE_string('vocab_file', None, 'The vocabulary file that the BERT model was trained on.')
flags.DEFINE_bool('do_lower_case', True, 'Whether to lower case the input text. Should be True for uncased models and False for cased models.')
flags.DEFINE_bool('verbose_logging', False, 'If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.')
flags.DEFINE_integer('predict_batch_size', 8, 'Total batch size for prediction.')
flags.DEFINE_integer('n_best_size', 20, 'The total number of n-best predictions to generate in the nbest_predictions.json output file.')
flags.DEFINE_integer('max_answer_length', 30, 'The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.')
flags.DEFINE_bool('use_keras_bert_for_squad', True, 'Deprecated and will be removed soon.')
common_flags.define_common_bert_flags()
FLAGS = flags.FLAGS

def squad_loss_fn(start_positions, end_positions, start_logits, end_logits, loss_factor=1.0):
    if False:
        while True:
            i = 10
    'Returns sparse categorical crossentropy for start/end logits.'
    start_loss = tf.keras.backend.sparse_categorical_crossentropy(start_positions, start_logits, from_logits=True)
    end_loss = tf.keras.backend.sparse_categorical_crossentropy(end_positions, end_logits, from_logits=True)
    total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
    total_loss *= loss_factor
    return total_loss

def get_loss_fn(loss_factor=1.0):
    if False:
        i = 10
        return i + 15
    'Gets a loss function for squad task.'

    def _loss_fn(labels, model_outputs):
        if False:
            print('Hello World!')
        start_positions = labels['start_positions']
        end_positions = labels['end_positions']
        (start_logits, end_logits) = model_outputs
        return squad_loss_fn(start_positions, end_positions, start_logits, end_logits, loss_factor=loss_factor)
    return _loss_fn

def get_raw_results(predictions):
    if False:
        i = 10
        return i + 15
    'Converts multi-replica predictions to RawResult.'
    for (unique_ids, start_logits, end_logits) in zip(predictions['unique_ids'], predictions['start_logits'], predictions['end_logits']):
        for values in zip(unique_ids.numpy(), start_logits.numpy(), end_logits.numpy()):
            yield squad_lib.RawResult(unique_id=values[0], start_logits=values[1].tolist(), end_logits=values[2].tolist())

def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size, is_training):
    if False:
        return 10
    'Gets a closure to create a dataset..'

    def _dataset_fn(ctx=None):
        if False:
            while True:
                i = 10
        'Returns tf.data.Dataset for distributed BERT pretraining.'
        batch_size = ctx.get_per_replica_batch_size(global_batch_size) if ctx else global_batch_size
        dataset = input_pipeline.create_squad_dataset(input_file_pattern, max_seq_length, batch_size, is_training=is_training, input_pipeline_context=ctx)
        return dataset
    return _dataset_fn

def predict_squad_customized(strategy, input_meta_data, bert_config, predict_tfrecord_path, num_steps):
    if False:
        return 10
    'Make predictions using a Bert-based squad model.'
    predict_dataset_fn = get_dataset_fn(predict_tfrecord_path, input_meta_data['max_seq_length'], FLAGS.predict_batch_size, is_training=False)
    predict_iterator = iter(strategy.experimental_distribute_datasets_from_function(predict_dataset_fn))
    with strategy.scope():
        tf.keras.mixed_precision.experimental.set_policy('float32')
        (squad_model, _) = bert_models.squad_model(bert_config, input_meta_data['max_seq_length'], float_type=tf.float32)
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    logging.info('Restoring checkpoints from %s', checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=squad_model)
    checkpoint.restore(checkpoint_path).expect_partial()

    @tf.function
    def predict_step(iterator):
        if False:
            print('Hello World!')
        'Predicts on distributed devices.'

        def _replicated_step(inputs):
            if False:
                while True:
                    i = 10
            'Replicated prediction calculation.'
            (x, _) = inputs
            unique_ids = x.pop('unique_ids')
            (start_logits, end_logits) = squad_model(x, training=False)
            return dict(unique_ids=unique_ids, start_logits=start_logits, end_logits=end_logits)
        outputs = strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))
        return tf.nest.map_structure(strategy.experimental_local_results, outputs)
    all_results = []
    for _ in range(num_steps):
        predictions = predict_step(predict_iterator)
        for result in get_raw_results(predictions):
            all_results.append(result)
        if len(all_results) % 100 == 0:
            logging.info('Made predictions for %d records.', len(all_results))
    return all_results

def train_squad(strategy, input_meta_data, custom_callbacks=None, run_eagerly=False):
    if False:
        while True:
            i = 10
    'Run bert squad training.'
    if strategy:
        logging.info('Training using customized training loop with distribution strategy.')
    keras_utils.set_config_v2(FLAGS.enable_xla)
    use_float16 = common_flags.use_float16()
    if use_float16:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    epochs = FLAGS.num_train_epochs
    num_train_examples = input_meta_data['train_data_size']
    max_seq_length = input_meta_data['max_seq_length']
    steps_per_epoch = int(num_train_examples / FLAGS.train_batch_size)
    warmup_steps = int(epochs * num_train_examples * 0.1 / FLAGS.train_batch_size)
    train_input_fn = get_dataset_fn(FLAGS.train_data_path, max_seq_length, FLAGS.train_batch_size, is_training=True)

    def _get_squad_model():
        if False:
            return 10
        'Get Squad model and optimizer.'
        (squad_model, core_model) = bert_models.squad_model(bert_config, max_seq_length, float_type=tf.float16 if use_float16 else tf.float32, hub_module_url=FLAGS.hub_module_url)
        squad_model.optimizer = optimization.create_optimizer(FLAGS.learning_rate, steps_per_epoch * epochs, warmup_steps)
        if use_float16:
            squad_model.optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(squad_model.optimizer, loss_scale=common_flags.get_loss_scale())
        if FLAGS.fp16_implementation == 'graph_rewrite':
            squad_model.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(squad_model.optimizer)
        return (squad_model, core_model)
    loss_fn = get_loss_fn(loss_factor=1.0 / strategy.num_replicas_in_sync if FLAGS.scale_loss else 1.0)
    model_training_utils.run_customized_training_loop(strategy=strategy, model_fn=_get_squad_model, loss_fn=loss_fn, model_dir=FLAGS.model_dir, steps_per_epoch=steps_per_epoch, steps_per_loop=FLAGS.steps_per_loop, epochs=epochs, train_input_fn=train_input_fn, init_checkpoint=FLAGS.init_checkpoint, run_eagerly=run_eagerly, custom_callbacks=custom_callbacks)

def predict_squad(strategy, input_meta_data):
    if False:
        print('Hello World!')
    'Makes predictions for a squad dataset.'
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    doc_stride = input_meta_data['doc_stride']
    max_query_length = input_meta_data['max_query_length']
    version_2_with_negative = input_meta_data.get('version_2_with_negative', False)
    eval_examples = squad_lib.read_squad_examples(input_file=FLAGS.predict_file, is_training=False, version_2_with_negative=version_2_with_negative)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    eval_writer = squad_lib.FeatureWriter(filename=os.path.join(FLAGS.model_dir, 'eval.tf_record'), is_training=False)
    eval_features = []

    def _append_feature(feature, is_padding):
        if False:
            while True:
                i = 10
        if not is_padding:
            eval_features.append(feature)
        eval_writer.process_feature(feature)
    dataset_size = squad_lib.convert_examples_to_features(examples=eval_examples, tokenizer=tokenizer, max_seq_length=input_meta_data['max_seq_length'], doc_stride=doc_stride, max_query_length=max_query_length, is_training=False, output_fn=_append_feature, batch_size=FLAGS.predict_batch_size)
    eval_writer.close()
    logging.info('***** Running predictions *****')
    logging.info('  Num orig examples = %d', len(eval_examples))
    logging.info('  Num split examples = %d', len(eval_features))
    logging.info('  Batch size = %d', FLAGS.predict_batch_size)
    num_steps = int(dataset_size / FLAGS.predict_batch_size)
    all_results = predict_squad_customized(strategy, input_meta_data, bert_config, eval_writer.filename, num_steps)
    output_prediction_file = os.path.join(FLAGS.model_dir, 'predictions.json')
    output_nbest_file = os.path.join(FLAGS.model_dir, 'nbest_predictions.json')
    output_null_log_odds_file = os.path.join(FLAGS.model_dir, 'null_odds.json')
    squad_lib.write_predictions(eval_examples, eval_features, all_results, FLAGS.n_best_size, FLAGS.max_answer_length, FLAGS.do_lower_case, output_prediction_file, output_nbest_file, output_null_log_odds_file, verbose=FLAGS.verbose_logging)

def export_squad(model_export_path, input_meta_data):
    if False:
        for i in range(10):
            print('nop')
    'Exports a trained model as a `SavedModel` for inference.\n\n  Args:\n    model_export_path: a string specifying the path to the SavedModel directory.\n    input_meta_data: dictionary containing meta data about input and model.\n\n  Raises:\n    Export path is not specified, got an empty string or None.\n  '
    if not model_export_path:
        raise ValueError('Export path is not specified: %s' % model_export_path)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    (squad_model, _) = bert_models.squad_model(bert_config, input_meta_data['max_seq_length'], float_type=tf.float32)
    model_saving_utils.export_bert_model(model_export_path, model=squad_model, checkpoint_dir=FLAGS.model_dir)

def main(_):
    if False:
        i = 10
        return i + 15
    assert tf.version.VERSION.startswith('2.')
    if not FLAGS.use_keras_bert_for_squad:
        raise ValueError('Old tf2 BERT is no longer supported. Please use keras BERT.')
    with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
        input_meta_data = json.loads(reader.read().decode('utf-8'))
    if FLAGS.mode == 'export_only':
        export_squad(FLAGS.model_export_path, input_meta_data)
        return
    strategy = distribution_utils.get_distribution_strategy(distribution_strategy=FLAGS.distribution_strategy, num_gpus=FLAGS.num_gpus, tpu_address=FLAGS.tpu)
    if FLAGS.mode in ('train', 'train_and_predict'):
        train_squad(strategy, input_meta_data)
    if FLAGS.mode in ('predict', 'train_and_predict'):
        predict_squad(strategy, input_meta_data)
if __name__ == '__main__':
    flags.mark_flag_as_required('bert_config_file')
    flags.mark_flag_as_required('model_dir')
    app.run(main)