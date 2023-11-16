"""A script to read in and store documents in a sqlite database."""
import argparse
import sqlite3
import json
import os
import logging
import importlib.util
from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from drqa.retriever import utils
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
PREPROCESS_FN = None

def init(filename):
    if False:
        i = 10
        return i + 15
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess

def import_module(filename):
    if False:
        while True:
            i = 10
    'Import a module given a full path to the file.'
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def iter_files(path):
    if False:
        i = 10
        return i + 15
    'Walk through all files located under a root path.'
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for (dirpath, _, filenames) in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)

def get_contents(filename):
    if False:
        return 10
    'Parse the contents of a file. Each line is a JSON encoded document.'
    global PREPROCESS_FN
    documents = []
    with open(filename) as f:
        for line in f:
            doc = json.loads(line)
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
            if not doc:
                continue
            documents.append((utils.normalize(doc['id']), doc['text']))
    return documents

def store_contents(data_path, save_path, preprocess, num_workers=None):
    if False:
        while True:
            i = 10
    'Preprocess and store a corpus of documents in sqlite.\n\n    Args:\n        data_path: Root path to directory (or directory of directories) of files\n          containing json encoded documents (must have `id` and `text` fields).\n        save_path: Path to output sqlite db.\n        preprocess: Path to file defining a custom `preprocess` function. Takes\n          in and outputs a structured doc.\n        num_workers: Number of parallel processes to use when reading docs.\n    '
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)
    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute('CREATE TABLE documents (id PRIMARY KEY, text);')
    workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(pairs)
            c.executemany('INSERT INTO documents VALUES (?,?)', pairs)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--preprocess', type=str, default=None, help='File path to a python module that defines a `preprocess` function')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()
    store_contents(args.data_path, args.save_path, args.preprocess, args.num_workers)