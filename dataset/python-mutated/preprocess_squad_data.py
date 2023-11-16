"""Script to pre-process SQUAD data into tfrecords."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import sentencepiece as spm
from official.nlp.xlnet import squad_utils
flags.DEFINE_integer('num_proc', default=1, help='Number of preprocessing processes.')
flags.DEFINE_integer('proc_id', default=0, help='Process id for preprocessing.')
flags.DEFINE_string('output_dir', default='', help='Output dir for TF records.')
flags.DEFINE_string('spiece_model_file', default='', help='Sentence Piece model path.')
flags.DEFINE_string('train_file', default='', help='Path of train file.')
flags.DEFINE_string('predict_file', default='', help='Path of prediction file.')
flags.DEFINE_integer('max_seq_length', default=512, help='Max sequence length')
flags.DEFINE_integer('max_query_length', default=64, help='Max query length')
flags.DEFINE_integer('doc_stride', default=128, help='Doc stride')
flags.DEFINE_bool('uncased', default=False, help='Use uncased data.')
flags.DEFINE_bool('create_train_data', default=True, help='Whether to create training data.')
flags.DEFINE_bool('create_eval_data', default=False, help='Whether to create eval data.')
FLAGS = flags.FLAGS

def preprocess():
    if False:
        for i in range(10):
            print('nop')
    'Preprocesses SQUAD data.'
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(FLAGS.spiece_model_file)
    spm_basename = os.path.basename(FLAGS.spiece_model_file)
    if FLAGS.create_train_data:
        train_rec_file = os.path.join(FLAGS.output_dir, '{}.{}.slen-{}.qlen-{}.train.tf_record'.format(spm_basename, FLAGS.proc_id, FLAGS.max_seq_length, FLAGS.max_query_length))
        logging.info('Read examples from %s', FLAGS.train_file)
        train_examples = squad_utils.read_squad_examples(FLAGS.train_file, is_training=True)
        train_examples = train_examples[FLAGS.proc_id::FLAGS.num_proc]
        random.shuffle(train_examples)
        write_to_logging = 'Write to ' + train_rec_file
        logging.info(write_to_logging)
        train_writer = squad_utils.FeatureWriter(filename=train_rec_file, is_training=True)
        squad_utils.convert_examples_to_features(examples=train_examples, sp_model=sp_model, max_seq_length=FLAGS.max_seq_length, doc_stride=FLAGS.doc_stride, max_query_length=FLAGS.max_query_length, is_training=True, output_fn=train_writer.process_feature, uncased=FLAGS.uncased)
        train_writer.close()
    if FLAGS.create_eval_data:
        eval_examples = squad_utils.read_squad_examples(FLAGS.predict_file, is_training=False)
        squad_utils.create_eval_data(spm_basename, sp_model, eval_examples, FLAGS.max_seq_length, FLAGS.max_query_length, FLAGS.doc_stride, FLAGS.uncased, FLAGS.output_dir)

def main(_):
    if False:
        while True:
            i = 10
    logging.set_verbosity(logging.INFO)
    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.mkdir(FLAGS.output_dir)
    preprocess()
if __name__ == '__main__':
    app.run(main)