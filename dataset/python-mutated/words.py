"""
Word Vectors module
"""
import logging
import os
import pickle
import tempfile
from errno import ENOENT
from multiprocessing import Pool
import numpy as np
try:
    import fasttext
    from pymagnitude import converter, Magnitude
    WORDS = True
except ImportError:
    WORDS = False
from ..pipeline import Tokenizer
from ..version import __pickle__
from .base import Vectors
logger = logging.getLogger(__name__)
VECTORS = None

def create(config, scoring):
    if False:
        return 10
    '\n    Multiprocessing helper method. Creates a global embeddings object to be accessed in a new subprocess.\n\n    Args:\n        config: vector configuration\n        scoring: scoring instance\n    '
    global VECTORS
    VECTORS = WordVectors(config, scoring)

def transform(document):
    if False:
        print('Hello World!')
    '\n    Multiprocessing helper method. Transforms document into an embeddings vector.\n\n    Args:\n        document: (id, data, tags)\n\n    Returns:\n        (id, embedding)\n    '
    return (document[0], VECTORS.transform(document))

class WordVectors(Vectors):
    """
    Builds sentence embeddings/vectors using weighted word embeddings.
    """

    def load(self, path):
        if False:
            i = 10
            return i + 15
        if not path or not os.path.isfile(path):
            raise IOError(ENOENT, 'Vector model file not found', path)
        return Magnitude(path, case_insensitive=True, blocking=not self.initialized)

    def encode(self, data):
        if False:
            i = 10
            return i + 15
        embeddings = []
        for tokens in data:
            if isinstance(tokens, str):
                tokens = Tokenizer.tokenize(tokens)
            weights = self.scoring.weights(tokens) if self.scoring else None
            if weights and [x for x in weights if x > 0]:
                embedding = np.average(self.lookup(tokens), weights=np.array(weights, dtype=np.float32), axis=0)
            else:
                embedding = np.mean(self.lookup(tokens), axis=0)
            embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)

    def index(self, documents, batchsize=1):
        if False:
            while True:
                i = 10
        if 'parallel' in self.config and (not self.config['parallel']):
            return super().index(documents, batchsize)
        (ids, dimensions, batches, stream) = ([], None, 0, None)
        args = (self.config, self.scoring)
        with Pool(os.cpu_count(), initializer=create, initargs=args) as pool:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.npy', delete=False) as output:
                stream = output.name
                embeddings = []
                for (uid, embedding) in pool.imap(transform, documents):
                    if not dimensions:
                        dimensions = embedding.shape[0]
                    ids.append(uid)
                    embeddings.append(embedding)
                    if len(embeddings) == batchsize:
                        pickle.dump(np.array(embeddings, dtype=np.float32), output, protocol=__pickle__)
                        batches += 1
                        embeddings = []
                if embeddings:
                    pickle.dump(np.array(embeddings, dtype=np.float32), output, protocol=__pickle__)
                    batches += 1
        return (ids, dimensions, batches, stream)

    def lookup(self, tokens):
        if False:
            return 10
        '\n        Queries word vectors for given list of input tokens.\n\n        Args:\n            tokens: list of tokens to query\n\n        Returns:\n            word vectors array\n        '
        return self.model.query(tokens)

    @staticmethod
    def isdatabase(path):
        if False:
            i = 10
            return i + 15
        '\n        Checks if this is a SQLite database file which is the file format used for word vectors databases.\n\n        Args:\n            path: path to check\n\n        Returns:\n            True if this is a SQLite database\n        '
        if isinstance(path, str) and os.path.isfile(path) and (os.path.getsize(path) >= 100):
            with open(path, 'rb') as f:
                header = f.read(100)
            return header.startswith(b'SQLite format 3\x00')
        return False

    @staticmethod
    def build(data, size, mincount, path):
        if False:
            return 10
        '\n        Builds fastText vectors from a file.\n\n        Args:\n            data: path to input data file\n            size: number of vector dimensions\n            mincount: minimum number of occurrences required to register a token\n            path: path to output file\n        '
        model = fasttext.train_unsupervised(data, dim=size, minCount=mincount)
        logger.info('Building %d dimension model', size)
        with open(path + '.txt', 'w', encoding='utf-8') as output:
            words = model.get_words()
            output.write(f'{len(words)} {model.get_dimension()}\n')
            for word in words:
                if word != '</s>':
                    vector = model.get_word_vector(word)
                    data = ''
                    for v in vector:
                        data += ' ' + str(v)
                    output.write(word + data + '\n')
        logger.info('Converting vectors to magnitude format')
        converter.convert(path + '.txt', path + '.magnitude', subword=True)