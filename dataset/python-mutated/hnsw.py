"""
HNSW module
"""
import numpy as np
try:
    from hnswlib import Index
    HNSWLIB = True
except ImportError:
    HNSWLIB = False
from .base import ANN

class HNSW(ANN):
    """
    Builds an ANN index using the hnswlib library.
    """

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        if not HNSWLIB:
            raise ImportError('HNSW is not available - install "similarity" extra to enable')

    def load(self, path):
        if False:
            while True:
                i = 10
        self.backend = Index(dim=self.config['dimensions'], space=self.config['metric'])
        self.backend.load_index(path)

    def index(self, embeddings):
        if False:
            return 10
        self.config['metric'] = 'ip'
        efconstruction = self.setting('efconstruction', 200)
        m = self.setting('m', 16)
        seed = self.setting('randomseed', 100)
        self.backend = Index(dim=self.config['dimensions'], space=self.config['metric'])
        self.backend.init_index(max_elements=embeddings.shape[0], ef_construction=efconstruction, M=m, random_seed=seed)
        self.backend.add_items(embeddings, np.arange(embeddings.shape[0], dtype=np.int64))
        self.config['offset'] = embeddings.shape[0]
        self.config['deletes'] = 0
        self.metadata({'efconstruction': efconstruction, 'm': m, 'seed': seed})

    def append(self, embeddings):
        if False:
            for i in range(10):
                print('nop')
        new = embeddings.shape[0]
        self.backend.resize_index(self.config['offset'] + new)
        self.backend.add_items(embeddings, np.arange(self.config['offset'], self.config['offset'] + new, dtype=np.int64))
        self.config['offset'] += new
        self.metadata()

    def delete(self, ids):
        if False:
            print('Hello World!')
        for uid in ids:
            try:
                self.backend.mark_deleted(uid)
                self.config['deletes'] += 1
            except RuntimeError:
                continue

    def search(self, queries, limit):
        if False:
            print('Hello World!')
        ef = self.setting('efsearch')
        if ef:
            self.backend.set_ef(ef)
        (ids, distances) = self.backend.knn_query(queries, k=limit)
        results = []
        for (x, distance) in enumerate(distances):
            scores = [1 - d for d in distance.tolist()]
            results.append(list(zip(ids[x].tolist(), scores)))
        return results

    def count(self):
        if False:
            while True:
                i = 10
        return self.backend.get_current_count() - self.config['deletes']

    def save(self, path):
        if False:
            return 10
        self.backend.save_index(path)