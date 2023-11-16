"""Common stuff used with LexNET."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from sklearn import metrics
import tensorflow as tf
POSTAGS = ['PAD', 'VERB', 'CONJ', 'NOUN', 'PUNCT', 'ADP', 'ADJ', 'DET', 'ADV', 'PART', 'NUM', 'X', 'INTJ', 'SYM']
POSTAG_TO_ID = {tag: tid for (tid, tag) in enumerate(POSTAGS)}
DEPLABELS = ['PAD', 'UNK', 'ROOT', 'abbrev', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'cc', 'ccomp', 'complm', 'conj', 'cop', 'csubj', 'csubjpass', 'dep', 'det', 'dobj', 'expl', 'infmod', 'iobj', 'mark', 'mwe', 'nc', 'neg', 'nn', 'npadvmod', 'nsubj', 'nsubjpass', 'num', 'number', 'p', 'parataxis', 'partmod', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prepc', 'prt', 'ps', 'purpcl', 'quantmod', 'rcmod', 'ref', 'rel', 'suffix', 'title', 'tmod', 'xcomp', 'xsubj']
DEPLABEL_TO_ID = {label: lid for (lid, label) in enumerate(DEPLABELS)}
DIRS = '_^V<>'
DIR_TO_ID = {dir: did for (did, dir) in enumerate(DIRS)}

def load_word_embeddings(word_embeddings_dir, word_embeddings_file):
    if False:
        print('Hello World!')
    'Loads pretrained word embeddings from a binary file and returns the matrix.\n\n  Args:\n    word_embeddings_dir: The directory for the word embeddings.\n    word_embeddings_file: The pretrained word embeddings text file.\n\n  Returns:\n    The word embeddings matrix\n  '
    embedding_file = os.path.join(word_embeddings_dir, word_embeddings_file)
    vocab_file = os.path.join(word_embeddings_dir, os.path.dirname(word_embeddings_file), 'vocab.txt')
    with open(vocab_file) as f_in:
        vocab = [line.strip() for line in f_in]
    vocab_size = len(vocab)
    print('Embedding file "%s" has %d tokens' % (embedding_file, vocab_size))
    with open(embedding_file) as f_in:
        embeddings = np.load(f_in)
    dim = embeddings.shape[1]
    special_embeddings = np.random.normal(0, 0.1, (4, dim))
    embeddings = np.vstack((special_embeddings, embeddings))
    embeddings = embeddings.astype(np.float32)
    return embeddings

def full_evaluation(model, session, instances, labels, set_name, classes):
    if False:
        while True:
            i = 10
    "Prints a full evaluation on the current set.\n\n  Performance (recall, precision and F1), classification report (per\n  class performance), and confusion matrix).\n\n  Args:\n    model: The currently trained path-based model.\n    session: The current TensorFlow session.\n    instances: The current set instances.\n    labels: The current set labels.\n    set_name: The current set name (train/validation/test).\n    classes: The class label names.\n\n  Returns:\n    The model's prediction for the given instances.\n  "
    pred = model.predict(session, instances)
    (precision, recall, f1, _) = metrics.precision_recall_fscore_support(labels, pred, average='weighted')
    print('%s set: Precision: %.3f, Recall: %.3f, F1: %.3f' % (set_name, precision, recall, f1))
    print('%s classification report:' % set_name)
    print(metrics.classification_report(labels, pred, target_names=classes))
    print('%s confusion matrix:' % set_name)
    cm = metrics.confusion_matrix(labels, pred, labels=range(len(classes)))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print_cm(cm, labels=classes)
    return pred

def print_cm(cm, labels):
    if False:
        return 10
    'Pretty print for confusion matrices.\n\n  From: https://gist.github.com/zachguo/10296432.\n\n  Args:\n    cm: The confusion matrix.\n    labels: The class names.\n  '
    columnwidth = 10
    empty_cell = ' ' * columnwidth
    short_labels = [label[:12].rjust(10, ' ') for label in labels]
    header = empty_cell + ' '
    header += ''.join([' %{0}s '.format(columnwidth) % label for label in short_labels])
    print(header)
    for (i, label1) in enumerate(short_labels):
        row = '%{0}s '.format(columnwidth) % label1[:10]
        for j in range(len(short_labels)):
            value = int(cm[i, j]) if not np.isnan(cm[i, j]) else 0
            cell = ' %{0}d '.format(10) % value
            row += cell + ' '
        print(row)

def load_all_labels(records):
    if False:
        i = 10
        return i + 15
    'Reads TensorFlow examples from a RecordReader and returns only the labels.\n\n  Args:\n    records: a record list with TensorFlow examples.\n\n  Returns:\n    The labels\n  '
    curr_features = tf.parse_example(records, {'rel_id': tf.FixedLenFeature([1], dtype=tf.int64)})
    labels = tf.squeeze(curr_features['rel_id'], [-1])
    return labels

def load_all_pairs(records):
    if False:
        print('Hello World!')
    'Reads TensorFlow examples from a RecordReader and returns the word pairs.\n\n  Args:\n    records: a record list with TensorFlow examples.\n\n  Returns:\n    The word pairs\n  '
    curr_features = tf.parse_example(records, {'pair': tf.FixedLenFeature([1], dtype=tf.string)})
    word_pairs = curr_features['pair']
    return word_pairs

def write_predictions(pairs, labels, predictions, classes, predictions_file):
    if False:
        for i in range(10):
            print('nop')
    'Write the predictions to a file.\n\n  Args:\n    pairs: the word pairs (list of tuple of two strings).\n    labels: the gold-standard labels for these pairs (array of rel ID).\n    predictions: the predicted labels for these pairs (array of rel ID).\n    classes: a list of relation names.\n    predictions_file: where to save the predictions.\n  '
    with open(predictions_file, 'w') as f_out:
        for (pair, label, pred) in zip(pairs, labels, predictions):
            (w1, w2) = pair
            f_out.write('\t'.join([w1, w2, classes[label], classes[pred]]) + '\n')