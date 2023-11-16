"""A program to train a tensorflow neural net segmenter from a conll file."""
import base64
import os
import os.path
import random
import time
from absl import flags
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from google.protobuf import text_format
from syntaxnet.ops import gen_parser_ops
from syntaxnet import task_spec_pb2
from syntaxnet import sentence_pb2
from dragnn.protos import spec_pb2
from dragnn.python.sentence_io import ConllSentenceReader
from dragnn.python import evaluation
from dragnn.python import graph_builder
from dragnn.python import lexicon
from dragnn.python import spec_builder
from dragnn.python import trainer_lib
FLAGS = flags.FLAGS
flags.DEFINE_string('tf_master', '', 'TensorFlow execution engine to connect to.')
flags.DEFINE_string('resource_path', '', 'Path to constructed resources.')
flags.DEFINE_string('tensorboard_dir', '', 'Directory for TensorBoard logs output.')
flags.DEFINE_string('checkpoint_filename', '', 'Filename to save the best checkpoint to.')
flags.DEFINE_string('training_corpus_path', '', 'Path to training data.')
flags.DEFINE_string('dev_corpus_path', '', 'Path to development set data.')
flags.DEFINE_bool('compute_lexicon', False, '')
flags.DEFINE_bool('projectivize_training_set', True, '')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train for.')
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('report_every', 500, 'Report cost and training accuracy every this many steps.')
flags.DEFINE_string('hyperparams', 'decay_steps:32000 dropout_rate:0.8 gradient_clip_norm:1 learning_method:"momentum" learning_rate:0.1 seed:1 momentum:0.95 use_moving_average:true', 'Hyperparameters of the model to train, either in ProtoBuftext format or base64-encoded ProtoBuf text format.')

def main(unused_argv):
    if False:
        i = 10
        return i + 15
    logging.set_verbosity(logging.INFO)
    if not gfile.IsDirectory(FLAGS.resource_path):
        gfile.MakeDirs(FLAGS.resource_path)
    if FLAGS.compute_lexicon:
        logging.info('Computing lexicon...')
        lexicon.build_lexicon(FLAGS.resource_path, FLAGS.training_corpus_path)
    lookahead = spec_builder.ComponentSpecBuilder('lookahead')
    lookahead.set_network_unit(name='wrapped_units.LayerNormBasicLSTMNetwork', hidden_layer_sizes='256')
    lookahead.set_transition_system(name='shift-only', left_to_right='false')
    lookahead.add_fixed_feature(name='char', fml='input(-1).char input.char input(1).char', embedding_dim=32)
    lookahead.add_fixed_feature(name='char-bigram', fml='input.char-bigram', embedding_dim=32)
    lookahead.fill_from_resources(FLAGS.resource_path, FLAGS.tf_master)
    segmenter = spec_builder.ComponentSpecBuilder('segmenter')
    segmenter.set_network_unit(name='wrapped_units.LayerNormBasicLSTMNetwork', hidden_layer_sizes='128')
    segmenter.set_transition_system(name='binary-segment-transitions')
    segmenter.add_token_link(source=lookahead, fml='input.focus stack.focus', embedding_dim=64)
    segmenter.fill_from_resources(FLAGS.resource_path, FLAGS.tf_master)
    master_spec = spec_pb2.MasterSpec()
    master_spec.component.extend([lookahead.spec, segmenter.spec])
    logging.info('Constructed master spec: %s', str(master_spec))
    with gfile.GFile(FLAGS.resource_path + '/master_spec', 'w') as f:
        f.write(str(master_spec).encode('utf-8'))
    hyperparam_config = spec_pb2.GridPoint()
    try:
        text_format.Parse(FLAGS.hyperparams, hyperparam_config)
    except text_format.ParseError:
        text_format.Parse(base64.b64decode(FLAGS.hyperparams), hyperparam_config)
    graph = tf.Graph()
    with graph.as_default():
        builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
        component_targets = spec_builder.default_targets_from_spec(master_spec)
        trainers = [builder.add_training_from_config(target) for target in component_targets]
        assert len(trainers) == 1
        annotator = builder.add_annotation()
        builder.add_saver()
    training_set = ConllSentenceReader(FLAGS.training_corpus_path, projectivize=False).corpus()
    dev_set = ConllSentenceReader(FLAGS.dev_corpus_path, projectivize=False).corpus()
    with tf.Session(graph=tf.Graph()) as tmp_session:
        char_training_set_op = gen_parser_ops.segmenter_training_data_constructor(training_set)
        char_dev_set_op = gen_parser_ops.char_token_generator(dev_set)
        char_training_set = tmp_session.run(char_training_set_op)
        char_dev_set = tmp_session.run(char_dev_set_op)
    logging.info('Training on %d sentences.', len(training_set))
    logging.info('Tuning on %d sentences.', len(dev_set))
    pretrain_steps = [0]
    train_steps = [FLAGS.num_epochs * len(training_set)]
    tf.logging.info('Creating TensorFlow checkpoint dir...')
    gfile.MakeDirs(os.path.dirname(FLAGS.checkpoint_filename))
    summary_writer = trainer_lib.get_summary_writer(FLAGS.tensorboard_dir)
    with tf.Session(FLAGS.tf_master, graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        trainer_lib.run_training(sess, trainers, annotator, evaluation.segmentation_summaries, pretrain_steps, train_steps, char_training_set, char_dev_set, dev_set, FLAGS.batch_size, summary_writer, FLAGS.report_every, builder.saver, FLAGS.checkpoint_filename)
if __name__ == '__main__':
    tf.app.run()