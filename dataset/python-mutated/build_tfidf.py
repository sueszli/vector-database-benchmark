"""A script to build the tf-idf document matrices for retrieval."""
import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter
from drqa import retriever
from drqa import tokenizers
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None

def init(tokenizer_class, db_class, db_opts):
    if False:
        print('Hello World!')
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

def fetch_text(doc_id):
    if False:
        i = 10
        return i + 15
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)

def tokenize(text):
    if False:
        for i in range(10):
            print('nop')
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)

def count(ngram, hash_size, doc_id):
    if False:
        return 10
    'Fetch the text of a document and compute hashed ngrams counts.'
    global DOC2IDX
    (row, col, data) = ([], [], [])
    tokens = tokenize(retriever.utils.normalize(fetch_text(doc_id)))
    ngrams = tokens.ngrams(n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram)
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return (row, col, data)

def get_count_matrix(args, db, db_opts):
    if False:
        return 10
    'Form a sparse word to document count matrix (inverted index).\n\n    M[i, j] = # times word i appears in document j.\n    '
    global DOC2IDX
    db_class = retriever.get_class(db)
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    DOC2IDX = {doc_id: i for (i, doc_id) in enumerate(doc_ids)}
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(args.num_workers, initializer=init, initargs=(tok_class, db_class, db_opts))
    logger.info('Mapping...')
    (row, col, data) = ([], [], [])
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for (i, batch) in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for (b_row, b_col, b_data) in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()
    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix((data, (row, col)), shape=(args.hash_size, len(doc_ids)))
    count_matrix.sum_duplicates()
    return (count_matrix, (DOC2IDX, doc_ids))

def get_tfidf_matrix(cnts):
    if False:
        return 10
    'Convert the word count matrix into tfidf one.\n\n    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))\n    * tf = term frequency in document\n    * N = number of documents\n    * Nt = number of occurences of term in all documents\n    '
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs

def get_doc_freqs(cnts):
    if False:
        print('Hello World!')
    'Return word --> # of docs it appears in.'
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None, help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None, help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2, help='Use up to N-size n-grams (e.g. 2 = unigrams + bigrams)')
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)), help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple', help="String option specifying tokenizer type to use (e.g. 'corenlp')")
    parser.add_argument('--num-workers', type=int, default=None, help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()
    logging.info('Counting words...')
    (count_matrix, doc_dict) = get_count_matrix(args, 'sqlite', {'db_path': args.db_path})
    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)
    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)
    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += '-tfidf-ngram=%d-hash=%d-tokenizer=%s' % (args.ngram, args.hash_size, args.tokenizer)
    filename = os.path.join(args.out_dir, basename)
    logger.info('Saving to %s.npz' % filename)
    metadata = {'doc_freqs': freqs, 'tokenizer': args.tokenizer, 'hash_size': args.hash_size, 'ngram': args.ngram, 'doc_dict': doc_dict}
    retriever.utils.save_sparse_csr(filename, tfidf, metadata)