"""Input readers and document/token generators for datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import csv
import os
import random
import tensorflow as tf
from data import data_utils
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '', 'Which dataset to generate data for')
flags.DEFINE_boolean('output_unigrams', True, 'Whether to output unigrams.')
flags.DEFINE_boolean('output_bigrams', False, 'Whether to output bigrams.')
flags.DEFINE_boolean('output_char', False, 'Whether to output characters.')
flags.DEFINE_boolean('lowercase', True, 'Whether to lowercase document terms.')
flags.DEFINE_string('imdb_input_dir', '', 'The input directory containing the IMDB sentiment dataset.')
flags.DEFINE_integer('imdb_validation_pos_start_id', 10621, 'File id of the first file in the pos sentiment validation set.')
flags.DEFINE_integer('imdb_validation_neg_start_id', 10625, 'File id of the first file in the neg sentiment validation set.')
flags.DEFINE_string('dbpedia_input_dir', '', 'Path to DBpedia directory containing train.csv and test.csv.')
flags.DEFINE_string('rcv1_input_dir', '', 'Path to rcv1 directory containing train.csv, unlab.csv, and test.csv.')
flags.DEFINE_string('rt_input_dir', '', 'The Rotten Tomatoes dataset input directory.')
flags.DEFINE_string('amazon_unlabeled_input_file', '', 'The unlabeled Amazon Reviews dataset input file. If set, the input file is used to augment RT and IMDB vocab.')
Document = namedtuple('Document', 'content is_validation is_test label add_tokens')

def documents(dataset='train', include_unlabeled=False, include_validation=False):
    if False:
        i = 10
        return i + 15
    "Generates Documents based on FLAGS.dataset.\n\n  Args:\n    dataset: str, identifies folder within IMDB data directory, test or train.\n    include_unlabeled: bool, whether to include the unsup directory. Only valid\n      when dataset=train.\n    include_validation: bool, whether to include validation data.\n\n  Yields:\n    Document\n\n  Raises:\n    ValueError: if include_unlabeled is true but dataset is not 'train'\n  "
    if include_unlabeled and dataset != 'train':
        raise ValueError('If include_unlabeled=True, must use train dataset')
    random.seed(302)
    ds = FLAGS.dataset
    if ds == 'imdb':
        docs_gen = imdb_documents
    elif ds == 'dbpedia':
        docs_gen = dbpedia_documents
    elif ds == 'rcv1':
        docs_gen = rcv1_documents
    elif ds == 'rt':
        docs_gen = rt_documents
    else:
        raise ValueError('Unrecognized dataset %s' % FLAGS.dataset)
    for doc in docs_gen(dataset, include_unlabeled, include_validation):
        yield doc

def tokens(doc):
    if False:
        i = 10
        return i + 15
    'Given a Document, produces character or word tokens.\n\n  Tokens can be either characters, or word-level tokens (unigrams and/or\n  bigrams).\n\n  Args:\n    doc: Document to produce tokens from.\n\n  Yields:\n    token\n\n  Raises:\n    ValueError: if all FLAGS.{output_unigrams, output_bigrams, output_char}\n      are False.\n  '
    if not (FLAGS.output_unigrams or FLAGS.output_bigrams or FLAGS.output_char):
        raise ValueError('At least one of {FLAGS.output_unigrams, FLAGS.output_bigrams, FLAGS.output_char} must be true')
    content = doc.content.strip()
    if FLAGS.lowercase:
        content = content.lower()
    if FLAGS.output_char:
        for char in content:
            yield char
    else:
        tokens_ = data_utils.split_by_punct(content)
        for (i, token) in enumerate(tokens_):
            if FLAGS.output_unigrams:
                yield token
            if FLAGS.output_bigrams:
                previous_token = tokens_[i - 1] if i > 0 else data_utils.EOS_TOKEN
                bigram = '_'.join([previous_token, token])
                yield bigram
                if i + 1 == len(tokens_):
                    bigram = '_'.join([token, data_utils.EOS_TOKEN])
                    yield bigram

def imdb_documents(dataset='train', include_unlabeled=False, include_validation=False):
    if False:
        i = 10
        return i + 15
    'Generates Documents for IMDB dataset.\n\n  Data from http://ai.stanford.edu/~amaas/data/sentiment/\n\n  Args:\n    dataset: str, identifies folder within IMDB data directory, test or train.\n    include_unlabeled: bool, whether to include the unsup directory. Only valid\n      when dataset=train.\n    include_validation: bool, whether to include validation data.\n\n  Yields:\n    Document\n\n  Raises:\n    ValueError: if FLAGS.imdb_input_dir is empty.\n  '
    if not FLAGS.imdb_input_dir:
        raise ValueError('Must provide FLAGS.imdb_input_dir')
    tf.logging.info('Generating IMDB documents...')

    def check_is_validation(filename, class_label):
        if False:
            i = 10
            return i + 15
        if class_label is None:
            return False
        file_idx = int(filename.split('_')[0])
        is_pos_valid = class_label and file_idx >= FLAGS.imdb_validation_pos_start_id
        is_neg_valid = not class_label and file_idx >= FLAGS.imdb_validation_neg_start_id
        return is_pos_valid or is_neg_valid
    dirs = [(dataset + '/pos', True), (dataset + '/neg', False)]
    if include_unlabeled:
        dirs.append(('train/unsup', None))
    for (d, class_label) in dirs:
        for filename in os.listdir(os.path.join(FLAGS.imdb_input_dir, d)):
            is_validation = check_is_validation(filename, class_label)
            if is_validation and (not include_validation):
                continue
            with open(os.path.join(FLAGS.imdb_input_dir, d, filename), encoding='utf-8') as imdb_f:
                content = imdb_f.read()
            yield Document(content=content, is_validation=is_validation, is_test=False, label=class_label, add_tokens=True)
    if FLAGS.amazon_unlabeled_input_file and include_unlabeled:
        with open(FLAGS.amazon_unlabeled_input_file, encoding='utf-8') as rt_f:
            for content in rt_f:
                yield Document(content=content, is_validation=False, is_test=False, label=None, add_tokens=False)

def dbpedia_documents(dataset='train', include_unlabeled=False, include_validation=False):
    if False:
        for i in range(10):
            print('nop')
    'Generates Documents for DBpedia dataset.\n\n  Dataset linked to at https://github.com/zhangxiangxiao/Crepe.\n\n  Args:\n    dataset: str, identifies the csv file within the DBpedia data directory,\n      test or train.\n    include_unlabeled: bool, unused.\n    include_validation: bool, whether to include validation data, which is a\n      randomly selected 10% of the data.\n\n  Yields:\n    Document\n\n  Raises:\n    ValueError: if FLAGS.dbpedia_input_dir is empty.\n  '
    del include_unlabeled
    if not FLAGS.dbpedia_input_dir:
        raise ValueError('Must provide FLAGS.dbpedia_input_dir')
    tf.logging.info('Generating DBpedia documents...')
    with open(os.path.join(FLAGS.dbpedia_input_dir, dataset + '.csv')) as db_f:
        reader = csv.reader(db_f)
        for row in reader:
            is_validation = random.randint(1, 10) == 1
            if is_validation and (not include_validation):
                continue
            content = row[1] + ' ' + row[2]
            yield Document(content=content, is_validation=is_validation, is_test=False, label=int(row[0]) - 1, add_tokens=True)

def rcv1_documents(dataset='train', include_unlabeled=True, include_validation=False):
    if False:
        return 10
    'Generates Documents for Reuters Corpus (rcv1) dataset.\n\n  Dataset described at\n  http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm\n\n  Args:\n    dataset: str, identifies the csv file within the rcv1 data directory.\n    include_unlabeled: bool, whether to include the unlab file. Only valid\n      when dataset=train.\n    include_validation: bool, whether to include validation data, which is a\n      randomly selected 10% of the data.\n\n  Yields:\n    Document\n\n  Raises:\n    ValueError: if FLAGS.rcv1_input_dir is empty.\n  '
    if not FLAGS.rcv1_input_dir:
        raise ValueError('Must provide FLAGS.rcv1_input_dir')
    tf.logging.info('Generating rcv1 documents...')
    datasets = [dataset]
    if include_unlabeled:
        if dataset == 'train':
            datasets.append('unlab')
    for dset in datasets:
        with open(os.path.join(FLAGS.rcv1_input_dir, dset + '.csv')) as db_f:
            reader = csv.reader(db_f)
            for row in reader:
                is_validation = random.randint(1, 10) == 1
                if is_validation and (not include_validation):
                    continue
                content = row[1]
                yield Document(content=content, is_validation=is_validation, is_test=False, label=int(row[0]), add_tokens=True)

def rt_documents(dataset='train', include_unlabeled=True, include_validation=False):
    if False:
        print('Hello World!')
    'Generates Documents for the Rotten Tomatoes dataset.\n\n  Dataset available at http://www.cs.cornell.edu/people/pabo/movie-review-data/\n  In this dataset, amazon reviews are used for the unlabeled data.\n\n  Args:\n    dataset: str, identifies the data subdirectory.\n    include_unlabeled: bool, whether to include the unlabeled data. Only valid\n      when dataset=train.\n    include_validation: bool, whether to include validation data, which is a\n      randomly selected 10% of the data.\n\n  Yields:\n    Document\n\n  Raises:\n    ValueError: if FLAGS.rt_input_dir is empty.\n  '
    if not FLAGS.rt_input_dir:
        raise ValueError('Must provide FLAGS.rt_input_dir')
    tf.logging.info('Generating rt documents...')
    data_files = []
    input_filenames = os.listdir(FLAGS.rt_input_dir)
    for inp_fname in input_filenames:
        if inp_fname.endswith('.pos'):
            data_files.append((os.path.join(FLAGS.rt_input_dir, inp_fname), True))
        elif inp_fname.endswith('.neg'):
            data_files.append((os.path.join(FLAGS.rt_input_dir, inp_fname), False))
    if include_unlabeled and FLAGS.amazon_unlabeled_input_file:
        data_files.append((FLAGS.amazon_unlabeled_input_file, None))
    for (filename, class_label) in data_files:
        with open(filename) as rt_f:
            for content in rt_f:
                if class_label is None:
                    if content.startswith('review/text'):
                        yield Document(content=content, is_validation=False, is_test=False, label=None, add_tokens=False)
                else:
                    random_int = random.randint(1, 10)
                    is_validation = random_int == 1
                    is_test = random_int == 2
                    if is_test and dataset != 'test' or (is_validation and (not include_validation)):
                        continue
                    yield Document(content=content, is_validation=is_validation, is_test=is_test, label=class_label, add_tokens=True)