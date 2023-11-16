import itertools

class InvalidCallback(ValueError):
    pass

class _Chunker(object):

    def __init__(self, table, chunksize, callback):
        if False:
            while True:
                i = 10
        self.queue = []
        self.table = table
        self.chunksize = chunksize
        if callback and (not callable(callback)):
            raise InvalidCallback
        self.callback = callback

    def flush(self):
        if False:
            i = 10
            return i + 15
        self.queue.clear()

    def _queue_add(self, item):
        if False:
            print('Hello World!')
        self.queue.append(item)
        if len(self.queue) >= self.chunksize:
            self.flush()

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        self.flush()

class ChunkedInsert(_Chunker):
    """Batch up insert operations
    with ChunkedInsert(my_table) as inserter:
        inserter(row)

    Rows will be inserted in groups of `chunksize` (defaulting to 1000). An
    optional callback can be provided that will be called before the insert.
    This callback takes one parameter which is the queue which is about to be
    inserted into the database
    """

    def __init__(self, table, chunksize=1000, callback=None):
        if False:
            return 10
        self.fields = set()
        super().__init__(table, chunksize, callback)

    def insert(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.fields.update(item.keys())
        super()._queue_add(item)

    def flush(self):
        if False:
            i = 10
            return i + 15
        for item in self.queue:
            for field in self.fields:
                item[field] = item.get(field)
        if self.callback is not None:
            self.callback(self.queue)
        self.table.insert_many(self.queue)
        super().flush()

class ChunkedUpdate(_Chunker):
    """Batch up update operations
    with ChunkedUpdate(my_table) as updater:
        updater(row)

    Rows will be updated in groups of `chunksize` (defaulting to 1000). An
    optional callback can be provided that will be called before the update.
    This callback takes one parameter which is the queue which is about to be
    updated into the database
    """

    def __init__(self, table, keys, chunksize=1000, callback=None):
        if False:
            for i in range(10):
                print('nop')
        self.keys = keys
        super().__init__(table, chunksize, callback)

    def update(self, item):
        if False:
            for i in range(10):
                print('nop')
        super()._queue_add(item)

    def flush(self):
        if False:
            i = 10
            return i + 15
        if self.callback is not None:
            self.callback(self.queue)
        self.queue.sort(key=dict.keys)
        for (fields, items) in itertools.groupby(self.queue, key=dict.keys):
            self.table.update_many(list(items), self.keys)
        super().flush()