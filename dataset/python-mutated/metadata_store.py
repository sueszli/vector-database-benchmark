import logging
from datetime import datetime as dt
import bson
import pandas as pd
import pymongo
from .bson_store import BSONStore
from .._util import indent
from ..decorators import mongo_retry
from ..exceptions import NoDataFoundException
logger = logging.getLogger(__name__)
METADATA_STORE_TYPE = 'MetadataStore'

class MetadataStore(BSONStore):
    """
    Metadata Store. This stores metadata with timestamps to allow temporal queries.

    Entries are stored in the following format:
        'symbol': symbol name
        'metadata': metadata to be persisted
        'start_time': when entry becomes effective
        'end_time': (Optional) when entry expires. If not set, it is still in effect

    For each symbol end_time of a entry should match start_time of the next one except for the current entry.
    """

    @classmethod
    def initialize_library(cls, arctic_lib, hashed=True, **kwargs):
        if False:
            return 10
        MetadataStore(arctic_lib)._ensure_index()
        BSONStore.initialize_library(arctic_lib, hashed, **kwargs)

    @mongo_retry
    def _ensure_index(self):
        if False:
            return 10
        self.create_index([('symbol', pymongo.ASCENDING), ('start_time', pymongo.DESCENDING)], unique=True, background=True)

    def __init__(self, arctic_lib):
        if False:
            while True:
                i = 10
        self._arctic_lib = arctic_lib
        self._reset()

    def _reset(self):
        if False:
            while True:
                i = 10
        self._collection = self._arctic_lib.get_top_level_collection().metadata

    def __getstate__(self):
        if False:
            while True:
                i = 10
        return {'arctic_lib': self._arctic_lib}

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        return MetadataStore.__init__(self, state['arctic_lib'])

    def __str__(self):
        if False:
            print('Hello World!')
        return '<%s at %s>\n%s' % (self.__class__.__name__, hex(id(self)), indent(str(self._arctic_lib), 4))

    def __repr__(self):
        if False:
            print('Hello World!')
        return str(self)

    @mongo_retry
    def list_symbols(self, regex=None, as_of=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n         Return the symbols in this library.\n\n         Parameters\n         ----------\n         as_of : `datetime.datetime`\n            filter symbols valid at given time\n         regex : `str`\n             filter symbols by the passed in regular expression\n         kwargs :\n             kwarg keys are used as fields to query for symbols with metadata matching\n             the kwargs query\n\n         Returns\n         -------\n         String list of symbols in the library\n        '
        if not (regex or as_of or kwargs):
            return self.distinct('symbol')
        index_query = {}
        if as_of is not None:
            index_query['start_time'] = {'$lte': as_of}
        if regex or as_of:
            index_query['symbol'] = {'$regex': regex or '^'}
        data_query = {}
        if kwargs:
            for (k, v) in kwargs.items():
                data_query['metadata.' + k] = v
        pipeline = [{'$sort': {'symbol': pymongo.ASCENDING, 'start_time': pymongo.DESCENDING}}]
        if index_query:
            pipeline.append({'$match': index_query})
        pipeline.append({'$group': {'_id': '$symbol', 'metadata': {'$first': '$metadata'}}})
        if data_query:
            pipeline.append({'$match': data_query})
        pipeline.append({'$project': {'_id': 0, 'symbol': '$_id'}})
        return sorted((r['symbol'] for r in self.aggregate(pipeline)))

    @mongo_retry
    def has_symbol(self, symbol):
        if False:
            for i in range(10):
                print('nop')
        return self.find_one({'symbol': symbol}) is not None

    @mongo_retry
    def read_history(self, symbol):
        if False:
            print('Hello World!')
        '\n        Return all metadata saved for `symbol`\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n\n        Returns\n        -------\n        pandas.DateFrame containing timestamps and metadata entries\n        '
        find = self.find({'symbol': symbol}, sort=[('start_time', pymongo.ASCENDING)])
        times = []
        entries = []
        for item in find:
            times.append(item['start_time'])
            entries.append(item['metadata'])
        return pd.DataFrame({symbol: entries}, times)

    @mongo_retry
    def read(self, symbol, as_of=None):
        if False:
            i = 10
            return i + 15
        '\n        Return current metadata saved for `symbol`\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        as_of : `datetime.datetime`\n            return entry valid at given time\n\n        Returns\n        -------\n        metadata\n        '
        if as_of is not None:
            res = self.find_one({'symbol': symbol, 'start_time': {'$lte': as_of}}, sort=[('start_time', pymongo.DESCENDING)])
        else:
            res = self.find_one({'symbol': symbol}, sort=[('start_time', pymongo.DESCENDING)])
        return res['metadata'] if res is not None else None

    def write_history(self, collection):
        if False:
            i = 10
            return i + 15
        "\n        Manually overwrite entire metadata history for symbols in `collection`\n\n        Parameters\n        ----------\n        collection : `list of pandas.DataFrame`\n            with symbol names as headers and timestamps as indices\n            (the same format as output of read_history)\n            Example:\n                [pandas.DataFrame({'symbol': [{}]}, [datetime.datetime.utcnow()])]\n        "
        documents = []
        for dataframe in collection:
            if len(dataframe.columns) != 1:
                raise ValueError('More than one symbol found in a DataFrame')
            symbol = dataframe.columns[0]
            times = dataframe.index
            entries = dataframe[symbol].values
            if self.has_symbol(symbol):
                self.purge(symbol)
            doc = {'symbol': symbol, 'metadata': entries[0], 'start_time': times[0]}
            for (metadata, start_time) in zip(entries[1:], times[1:]):
                if metadata == doc['metadata']:
                    continue
                doc['end_time'] = start_time
                documents.append(doc)
                doc = {'symbol': symbol, 'metadata': metadata, 'start_time': start_time}
            documents.append(doc)
        self.insert_many(documents)

    def append(self, symbol, metadata, start_time=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update metadata entry for `symbol`\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        metadata : `dict`\n            to be persisted\n        start_time : `datetime.datetime`\n            when metadata becomes effective\n            Default: datetime.datetime.utcnow()\n        '
        if start_time is None:
            start_time = dt.utcnow()
        old_metadata = self.find_one({'symbol': symbol}, sort=[('start_time', pymongo.DESCENDING)])
        if old_metadata is not None:
            if old_metadata['start_time'] >= start_time:
                raise ValueError('start_time={} is earlier than the last metadata @{}'.format(start_time, old_metadata['start_time']))
            if old_metadata['metadata'] == metadata:
                return old_metadata
        elif metadata is None:
            return
        self.find_one_and_update({'symbol': symbol}, {'$set': {'end_time': start_time}}, sort=[('start_time', pymongo.DESCENDING)])
        document = {'_id': bson.ObjectId(), 'symbol': symbol, 'metadata': metadata, 'start_time': start_time}
        mongo_retry(self.insert_one)(document)
        logger.debug('Finished writing metadata for %s', symbol)
        return document

    def prepend(self, symbol, metadata, start_time=None):
        if False:
            print('Hello World!')
        '\n        Prepend a metadata entry for `symbol`\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        metadata : `dict`\n            to be persisted\n        start_time : `datetime.datetime`\n            when metadata becomes effective\n            Default: datetime.datetime.min\n        '
        if metadata is None:
            return
        if start_time is None:
            start_time = dt.min
        old_metadata = self.find_one({'symbol': symbol}, sort=[('start_time', pymongo.ASCENDING)])
        if old_metadata is not None:
            if old_metadata['start_time'] <= start_time:
                raise ValueError('start_time={} is later than the first metadata @{}'.format(start_time, old_metadata['start_time']))
            if old_metadata['metadata'] == metadata:
                self.find_one_and_update({'symbol': symbol}, {'$set': {'start_time': start_time}}, sort=[('start_time', pymongo.ASCENDING)])
                old_metadata['start_time'] = start_time
                return old_metadata
            end_time = old_metadata.get('start_time')
        else:
            end_time = None
        document = {'_id': bson.ObjectId(), 'symbol': symbol, 'metadata': metadata, 'start_time': start_time}
        if end_time is not None:
            document['end_time'] = end_time
        mongo_retry(self.insert_one)(document)
        logger.debug('Finished writing metadata for %s', symbol)
        return document

    def pop(self, symbol):
        if False:
            while True:
                i = 10
        '\n        Delete current metadata of `symbol`\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name to delete\n\n        Returns\n        -------\n        Deleted metadata\n        '
        last_metadata = self.find_one({'symbol': symbol}, sort=[('start_time', pymongo.DESCENDING)])
        if last_metadata is None:
            raise NoDataFoundException('No metadata found for symbol {}'.format(symbol))
        self.find_one_and_delete({'symbol': symbol}, sort=[('start_time', pymongo.DESCENDING)])
        mongo_retry(self.find_one_and_update)({'symbol': symbol}, {'$unset': {'end_time': ''}}, sort=[('start_time', pymongo.DESCENDING)])
        return last_metadata

    @mongo_retry
    def purge(self, symbol):
        if False:
            print('Hello World!')
        '\n        Delete all metadata of `symbol`\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name to delete\n        '
        logger.warning('Deleting entire metadata history for %r from %r' % (symbol, self._arctic_lib.get_name()))
        self.delete_many({'symbol': symbol})