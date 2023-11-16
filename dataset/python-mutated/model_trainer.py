"""Trainer for generic DRAGNN models.

This trainer uses a "model directory" for both input and output.  When invoked,
the model directory should contain the following inputs:

  <model_dir>/config.txt: A stringified dict that defines high-level
    configuration parameters.  Unset parameters default to False.
  <model_dir>/master.pbtxt: A text-format MasterSpec proto that defines
    the DRAGNN network to train.
  <model_dir>/hyperparameters.pbtxt: A text-format GridPoint proto that
    defines training hyper-parameters.
  <model_dir>/targets.pbtxt: (Optional) A text-format TrainingGridSpec whose
    "target" field defines the training targets.  If missing, then default
    training targets are used instead.

On success, the model directory will contain the following outputs:

  <model_dir>/checkpoints/best: The best checkpoint seen during training, as
    measured by accuracy on the eval corpus.
  <model_dir>/tensorboard: TensorBoard log directory.

Outside of the files and subdirectories named above, the model directory should
contain any other necessary files (e.g., pretrained embeddings).  See the model
builders in dragnn/examples.
"""
import ast
import collections
import os
import os.path
from absl import app
from absl import flags
import tensorflow as tf
from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import evaluation
from dragnn.python import graph_builder
from dragnn.python import sentence_io
from dragnn.python import spec_builder
from dragnn.python import trainer_lib
from syntaxnet.ops import gen_parser_ops
from syntaxnet.util import check
FLAGS = flags.FLAGS
flags.DEFINE_string('tf_master', '', 'TensorFlow execution engine to connect to.')
flags.DEFINE_string('model_dir', None, 'Path to a prepared model directory.')
flags.DEFINE_string('pretrain_steps', None, 'Comma-delimited list of pre-training steps per training target.')
flags.DEFINE_string('pretrain_epochs', None, 'Comma-delimited list of pre-training epochs per training target.')
flags.DEFINE_string('train_steps', None, 'Comma-delimited list of training steps per training target.')
flags.DEFINE_string('train_epochs', None, 'Comma-delimited list of training epochs per training target.')
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('report_every', 200, 'Report cost and training accuracy every this many steps.')

def _read_text_proto(path, proto_type):
    if False:
        print('Hello World!')
    'Reads a text-format instance of |proto_type| from the |path|.'
    proto = proto_type()
    with tf.gfile.FastGFile(path) as proto_file:
        text_format.Parse(proto_file.read(), proto)
    return proto

def _convert_to_char_corpus(corpus):
    if False:
        return 10
    'Converts the word-based |corpus| into a char-based corpus.'
    with tf.Session(graph=tf.Graph()) as tmp_session:
        conversion_op = gen_parser_ops.segmenter_training_data_constructor(corpus)
        return tmp_session.run(conversion_op)

def _get_steps(steps_flag, epochs_flag, corpus_length):
    if False:
        return 10
    'Converts the |steps_flag| or |epochs_flag| into a list of step counts.'
    if steps_flag:
        return map(int, steps_flag.split(','))
    return [corpus_length * int(epochs) for epochs in epochs_flag.split(',')]

def main(unused_argv):
    if False:
        print('Hello World!')
    tf.logging.set_verbosity(tf.logging.INFO)
    check.NotNone(FLAGS.model_dir, '--model_dir is required')
    check.Ne(FLAGS.pretrain_steps is None, FLAGS.pretrain_epochs is None, 'Exactly one of --pretrain_steps or --pretrain_epochs is required')
    check.Ne(FLAGS.train_steps is None, FLAGS.train_epochs is None, 'Exactly one of --train_steps or --train_epochs is required')
    config_path = os.path.join(FLAGS.model_dir, 'config.txt')
    master_path = os.path.join(FLAGS.model_dir, 'master.pbtxt')
    hyperparameters_path = os.path.join(FLAGS.model_dir, 'hyperparameters.pbtxt')
    targets_path = os.path.join(FLAGS.model_dir, 'targets.pbtxt')
    checkpoint_path = os.path.join(FLAGS.model_dir, 'checkpoints/best')
    tensorboard_dir = os.path.join(FLAGS.model_dir, 'tensorboard')
    with tf.gfile.FastGFile(config_path) as config_file:
        config = collections.defaultdict(bool, ast.literal_eval(config_file.read()))
    train_corpus_path = config['train_corpus_path']
    tune_corpus_path = config['tune_corpus_path']
    projectivize_train_corpus = config['projectivize_train_corpus']
    master = _read_text_proto(master_path, spec_pb2.MasterSpec)
    hyperparameters = _read_text_proto(hyperparameters_path, spec_pb2.GridPoint)
    targets = spec_builder.default_targets_from_spec(master)
    if tf.gfile.Exists(targets_path):
        targets = _read_text_proto(targets_path, spec_pb2.TrainingGridSpec).target
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(hyperparameters.seed)
        builder = graph_builder.MasterBuilder(master, hyperparameters)
        trainers = [builder.add_training_from_config(target) for target in targets]
        annotator = builder.add_annotation()
        builder.add_saver()
    train_corpus = sentence_io.ConllSentenceReader(train_corpus_path, projectivize=projectivize_train_corpus).corpus()
    tune_corpus = sentence_io.ConllSentenceReader(tune_corpus_path, projectivize=False).corpus()
    gold_tune_corpus = tune_corpus
    if config['convert_to_char_corpora']:
        train_corpus = _convert_to_char_corpus(train_corpus)
        tune_corpus = _convert_to_char_corpus(tune_corpus)
    pretrain_steps = _get_steps(FLAGS.pretrain_steps, FLAGS.pretrain_epochs, len(train_corpus))
    train_steps = _get_steps(FLAGS.train_steps, FLAGS.train_epochs, len(train_corpus))
    check.Eq(len(targets), len(pretrain_steps), 'Length mismatch between training targets and --pretrain_steps')
    check.Eq(len(targets), len(train_steps), 'Length mismatch between training targets and --train_steps')
    tf.logging.info('Training on %d sentences.', len(train_corpus))
    tf.logging.info('Tuning on %d sentences.', len(tune_corpus))
    tf.logging.info('Creating TensorFlow checkpoint dir...')
    summary_writer = trainer_lib.get_summary_writer(tensorboard_dir)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if tf.gfile.IsDirectory(checkpoint_dir):
        tf.gfile.DeleteRecursively(checkpoint_dir)
    elif tf.gfile.Exists(checkpoint_dir):
        tf.gfile.Remove(checkpoint_dir)
    tf.gfile.MakeDirs(checkpoint_dir)
    with tf.Session(FLAGS.tf_master, graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        trainer_lib.run_training(sess, trainers, annotator, evaluation.parser_summaries, pretrain_steps, train_steps, train_corpus, tune_corpus, gold_tune_corpus, FLAGS.batch_size, summary_writer, FLAGS.report_every, builder.saver, checkpoint_path)
    tf.logging.info('Best checkpoint written to:\n%s', checkpoint_path)
if __name__ == '__main__':
    app.run(main)