import random
from google.cloud import firestore

class Shard:
    """
    A shard is a distributed counter. Each shard can support being incremented
    once per second. Multiple shards are needed within a Counter to allow
    more frequent incrementing.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._count = 0

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        return {'count': self._count}

class Counter:
    """
    A counter stores a collection of shards which are
    summed to return a total count. This allows for more
    frequent incrementing than a single document.
    """

    def __init__(self, num_shards):
        if False:
            return 10
        self._num_shards = num_shards

    def init_counter(self, doc_ref):
        if False:
            print('Hello World!')
        '\n        Create a given number of shards as\n        subcollection of specified document.\n        '
        col_ref = doc_ref.collection('shards')
        for num in range(self._num_shards):
            shard = Shard()
            col_ref.document(str(num)).set(shard.to_dict())

    def increment_counter(self, doc_ref):
        if False:
            while True:
                i = 10
        'Increment a randomly picked shard.'
        doc_id = random.randint(0, self._num_shards - 1)
        shard_ref = doc_ref.collection('shards').document(str(doc_id))
        return shard_ref.update({'count': firestore.Increment(1)})

    def get_count(self, doc_ref):
        if False:
            print('Hello World!')
        'Return a total count across all shards.'
        total = 0
        shards = doc_ref.collection('shards').list_documents()
        for shard in shards:
            total += shard.get().to_dict().get('count', 0)
        return total