import datetime
import faiss
import numpy as np
from utils import logger
embedding_size = 512

class RetrievalIndex(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        pass
        self.indices = {}
        self.image_hashes = {}

    def build_index_for_user(self, user_id, image_hashes, image_embeddings):
        if False:
            return 10
        logger.info('building index for user {} - got {} photos to process'.format(user_id, len(image_hashes)))
        start = datetime.datetime.now()
        self.indices[user_id] = faiss.IndexFlatIP(embedding_size)
        self.image_hashes[user_id] = []
        for (h, e) in zip(image_hashes, image_embeddings):
            self.image_hashes[user_id].append(h)
            self.indices[user_id].add(np.array([e], dtype=np.float32))
        elapsed = (datetime.datetime.now() - start).total_seconds()
        logger.info('finished building index for user %d - took %.2f seconds' % (user_id, elapsed))

    def search_similar(self, user_id, in_embedding, n=100, thres=27.0):
        if False:
            print('Hello World!')
        start = datetime.datetime.now()
        (dist, res_indices) = self.indices[user_id].search(np.array([in_embedding], dtype=np.float32), n)
        res = []
        for (distance, idx) in sorted(zip(dist[0], res_indices[0]), reverse=True):
            if distance >= thres:
                res.append(self.image_hashes[user_id][idx])
        elapsed = (datetime.datetime.now() - start).total_seconds()
        logger.info('searched for %d images for user %d - took %.2f seconds' % (n, user_id, elapsed))
        return res