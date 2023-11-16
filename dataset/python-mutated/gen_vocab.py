"""Generates vocabulary and term frequency files for datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import iteritems
from collections import defaultdict
import tensorflow as tf
from data import data_utils
from data import document_generators
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', '', 'Path to save vocab.txt and vocab_freq.txt.')
flags.DEFINE_boolean('use_unlabeled', True, 'Whether to use the unlabeled sentiment dataset in the vocabulary.')
flags.DEFINE_boolean('include_validation', False, 'Whether to include the validation set in the vocabulary.')
flags.DEFINE_integer('doc_count_threshold', 1, 'The minimum number of documents a word or bigram should occur in to keep it in the vocabulary.')
MAX_VOCAB_SIZE = 100 * 1000

def fill_vocab_from_doc(doc, vocab_freqs, doc_counts):
    if False:
        return 10
    'Fills vocabulary and doc counts with tokens from doc.\n\n  Args:\n    doc: Document to read tokens from.\n    vocab_freqs: dict<token, frequency count>\n    doc_counts: dict<token, document count>\n\n  Returns:\n    None\n  '
    doc_seen = set()
    for token in document_generators.tokens(doc):
        if doc.add_tokens or token in vocab_freqs:
            vocab_freqs[token] += 1
        if token not in doc_seen:
            doc_counts[token] += 1
            doc_seen.add(token)

def main(_):
    if False:
        i = 10
        return i + 15
    tf.logging.set_verbosity(tf.logging.INFO)
    vocab_freqs = defaultdict(int)
    doc_counts = defaultdict(int)
    for doc in document_generators.documents(dataset='train', include_unlabeled=FLAGS.use_unlabeled, include_validation=FLAGS.include_validation):
        fill_vocab_from_doc(doc, vocab_freqs, doc_counts)
    vocab_freqs = dict(((term, freq) for (term, freq) in iteritems(vocab_freqs) if doc_counts[term] > FLAGS.doc_count_threshold))
    ordered_vocab_freqs = data_utils.sort_vocab_by_frequency(vocab_freqs)
    ordered_vocab_freqs = ordered_vocab_freqs[:MAX_VOCAB_SIZE]
    ordered_vocab_freqs.append((data_utils.EOS_TOKEN, 1))
    tf.gfile.MakeDirs(FLAGS.output_dir)
    data_utils.write_vocab_and_frequency(ordered_vocab_freqs, FLAGS.output_dir)
if __name__ == '__main__':
    tf.app.run()