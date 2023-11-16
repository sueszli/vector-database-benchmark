"""Module for decoding."""
import os
import time
import beam_search
import data
from six.moves import xrange
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_decode_steps', 1000000, 'Number of decoding steps.')
tf.app.flags.DEFINE_integer('decode_batches_per_ckpt', 8000, 'Number of batches to decode before restoring next checkpoint')
DECODE_LOOP_DELAY_SECS = 60
DECODE_IO_FLUSH_INTERVAL = 100

class DecodeIO(object):
    """Writes the decoded and references to RKV files for Rouge score.

    See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
  """

    def __init__(self, outdir):
        if False:
            print('Hello World!')
        self._cnt = 0
        self._outdir = outdir
        if not os.path.exists(self._outdir):
            os.mkdir(self._outdir)
        self._ref_file = None
        self._decode_file = None

    def Write(self, reference, decode):
        if False:
            print('Hello World!')
        'Writes the reference and decoded outputs to RKV files.\n\n    Args:\n      reference: The human (correct) result.\n      decode: The machine-generated result\n    '
        self._ref_file.write('output=%s\n' % reference)
        self._decode_file.write('output=%s\n' % decode)
        self._cnt += 1
        if self._cnt % DECODE_IO_FLUSH_INTERVAL == 0:
            self._ref_file.flush()
            self._decode_file.flush()

    def ResetFiles(self):
        if False:
            return 10
        'Resets the output files. Must be called once before Write().'
        if self._ref_file:
            self._ref_file.close()
        if self._decode_file:
            self._decode_file.close()
        timestamp = int(time.time())
        self._ref_file = open(os.path.join(self._outdir, 'ref%d' % timestamp), 'w')
        self._decode_file = open(os.path.join(self._outdir, 'decode%d' % timestamp), 'w')

class BSDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batch_reader, hps, vocab):
        if False:
            i = 10
            return i + 15
        'Beam search decoding.\n\n    Args:\n      model: The seq2seq attentional model.\n      batch_reader: The batch data reader.\n      hps: Hyperparamters.\n      vocab: Vocabulary\n    '
        self._model = model
        self._model.build_graph()
        self._batch_reader = batch_reader
        self._hps = hps
        self._vocab = vocab
        self._saver = tf.train.Saver()
        self._decode_io = DecodeIO(FLAGS.decode_dir)

    def DecodeLoop(self):
        if False:
            print('Hello World!')
        'Decoding loop for long running process.'
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        step = 0
        while step < FLAGS.max_decode_steps:
            time.sleep(DECODE_LOOP_DELAY_SECS)
            if not self._Decode(self._saver, sess):
                continue
            step += 1

    def _Decode(self, saver, sess):
        if False:
            while True:
                i = 10
        'Restore a checkpoint and decode it.\n\n    Args:\n      saver: Tensorflow checkpoint saver.\n      sess: Tensorflow session.\n    Returns:\n      If success, returns true, otherwise, false.\n    '
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
            return False
        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        saver.restore(sess, ckpt_path)
        self._decode_io.ResetFiles()
        for _ in xrange(FLAGS.decode_batches_per_ckpt):
            (article_batch, _, _, article_lens, _, _, origin_articles, origin_abstracts) = self._batch_reader.NextBatch()
            for i in xrange(self._hps.batch_size):
                bs = beam_search.BeamSearch(self._model, self._hps.batch_size, self._vocab.WordToId(data.SENTENCE_START), self._vocab.WordToId(data.SENTENCE_END), self._hps.dec_timesteps)
                article_batch_cp = article_batch.copy()
                article_batch_cp[:] = article_batch[i:i + 1]
                article_lens_cp = article_lens.copy()
                article_lens_cp[:] = article_lens[i:i + 1]
                best_beam = bs.BeamSearch(sess, article_batch_cp, article_lens_cp)[0]
                decode_output = [int(t) for t in best_beam.tokens[1:]]
                self._DecodeBatch(origin_articles[i], origin_abstracts[i], decode_output)
        return True

    def _DecodeBatch(self, article, abstract, output_ids):
        if False:
            return 10
        'Convert id to words and writing results.\n\n    Args:\n      article: The original article string.\n      abstract: The human (correct) abstract string.\n      output_ids: The abstract word ids output by machine.\n    '
        decoded_output = ' '.join(data.Ids2Words(output_ids, self._vocab))
        end_p = decoded_output.find(data.SENTENCE_END, 0)
        if end_p != -1:
            decoded_output = decoded_output[:end_p]
        tf.logging.info('article:  %s', article)
        tf.logging.info('abstract: %s', abstract)
        tf.logging.info('decoded:  %s', decoded_output)
        self._decode_io.Write(abstract, decoded_output.strip())