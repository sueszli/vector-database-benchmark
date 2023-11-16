"""This module implements IO classes to read and write data on MongoDB.


Read from MongoDB
-----------------
:class:`ReadFromMongoDB` is a ``PTransform`` that reads from a configured
MongoDB source and returns a ``PCollection`` of dict representing MongoDB
documents.
To configure MongoDB source, the URI to connect to MongoDB server, database
name, collection name needs to be provided.

Example usage::

  pipeline | ReadFromMongoDB(uri='mongodb://localhost:27017',
                             db='testdb',
                             coll='input')

To read from MongoDB Atlas, use ``bucket_auto`` option to enable
``@bucketAuto`` MongoDB aggregation instead of ``splitVector``
command which is a high-privilege function that cannot be assigned
to any user in Atlas.

Example usage::

  pipeline | ReadFromMongoDB(uri='mongodb+srv://user:pwd@cluster0.mongodb.net',
                             db='testdb',
                             coll='input',
                             bucket_auto=True)


Write to MongoDB:
-----------------
:class:`WriteToMongoDB` is a ``PTransform`` that writes MongoDB documents to
configured sink, and the write is conducted through a mongodb bulk_write of
``ReplaceOne`` operations. If the document's _id field already existed in the
MongoDB collection, it results in an overwrite, otherwise, a new document
will be inserted.

Example usage::

  pipeline | WriteToMongoDB(uri='mongodb://localhost:27017',
                            db='testdb',
                            coll='output',
                            batch_size=10)


No backward compatibility guarantees. Everything in this module is experimental.
"""
import itertools
import json
import logging
import math
import struct
from typing import Union
import apache_beam as beam
from apache_beam.io import iobase
from apache_beam.io.range_trackers import LexicographicKeyRangeTracker
from apache_beam.io.range_trackers import OffsetRangeTracker
from apache_beam.io.range_trackers import OrderedPositionRangeTracker
from apache_beam.transforms import DoFn
from apache_beam.transforms import PTransform
from apache_beam.transforms import Reshuffle
_LOGGER = logging.getLogger(__name__)
try:
    from bson import json_util
    from bson import objectid
    from bson.objectid import ObjectId
    from pymongo import ASCENDING
    from pymongo import DESCENDING
    from pymongo import MongoClient
    from pymongo import ReplaceOne
except ImportError:
    objectid = None
    json_util = None
    ObjectId = None
    ASCENDING = 1
    DESCENDING = -1
    MongoClient = None
    ReplaceOne = None
    _LOGGER.warning('Could not find a compatible bson package.')
__all__ = ['ReadFromMongoDB', 'WriteToMongoDB']

class ReadFromMongoDB(PTransform):
    """A ``PTransform`` to read MongoDB documents into a ``PCollection``."""

    def __init__(self, uri='mongodb://localhost:27017', db=None, coll=None, filter=None, projection=None, extra_client_params=None, bucket_auto=False):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`ReadFromMongoDB`\n\n    Args:\n      uri (str): The MongoDB connection string following the URI format.\n      db (str): The MongoDB database name.\n      coll (str): The MongoDB collection name.\n      filter: A `bson.SON\n        <https://api.mongodb.com/python/current/api/bson/son.html>`_ object\n        specifying elements which must be present for a document to be included\n        in the result set.\n      projection: A list of field names that should be returned in the result\n        set or a dict specifying the fields to include or exclude.\n      extra_client_params(dict): Optional `MongoClient\n        <https://api.mongodb.com/python/current/api/pymongo/mongo_client.html>`_\n        parameters.\n      bucket_auto (bool): If :data:`True`, use MongoDB `$bucketAuto` aggregation\n        to split collection into bundles instead of `splitVector` command,\n        which does not work with MongoDB Atlas.\n        If :data:`False` (the default), use `splitVector` command for bundling.\n\n    Returns:\n      :class:`~apache_beam.transforms.ptransform.PTransform`\n    '
        if extra_client_params is None:
            extra_client_params = {}
        if not isinstance(db, str):
            raise ValueError('ReadFromMongDB db param must be specified as a string')
        if not isinstance(coll, str):
            raise ValueError('ReadFromMongDB coll param must be specified as a string')
        self._mongo_source = _BoundedMongoSource(uri=uri, db=db, coll=coll, filter=filter, projection=projection, extra_client_params=extra_client_params, bucket_auto=bucket_auto)

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | iobase.Read(self._mongo_source)

class _ObjectIdRangeTracker(OrderedPositionRangeTracker):
    """RangeTracker for tracking mongodb _id of bson ObjectId type."""

    def position_to_fraction(self, pos: ObjectId, start: ObjectId, end: ObjectId):
        if False:
            i = 10
            return i + 15
        'Returns the fraction of keys in the range [start, end) that\n    are less than the given key.\n    '
        pos_number = _ObjectIdHelper.id_to_int(pos)
        start_number = _ObjectIdHelper.id_to_int(start)
        end_number = _ObjectIdHelper.id_to_int(end)
        return (pos_number - start_number) / (end_number - start_number)

    def fraction_to_position(self, fraction: float, start: ObjectId, end: ObjectId):
        if False:
            while True:
                i = 10
        'Converts a fraction between 0 and 1\n    to a position between start and end.\n    '
        start_number = _ObjectIdHelper.id_to_int(start)
        end_number = _ObjectIdHelper.id_to_int(end)
        total = end_number - start_number
        pos = int(total * fraction + start_number)
        if pos <= start_number:
            return _ObjectIdHelper.increment_id(start, 1)
        if pos >= end_number:
            return _ObjectIdHelper.increment_id(end, -1)
        return _ObjectIdHelper.int_to_id(pos)

class _BoundedMongoSource(iobase.BoundedSource):
    """A MongoDB source that reads a finite amount of input records.

  This class defines following operations which can be used to read
  MongoDB source efficiently.

  * Size estimation - method ``estimate_size()`` may return an accurate
    estimation in bytes for the size of the source.
  * Splitting into bundles of a given size - method ``split()`` can be used to
    split the source into a set of sub-sources (bundles) based on a desired
    bundle size.
  * Getting a RangeTracker - method ``get_range_tracker()`` should return a
    ``RangeTracker`` object for a given position range for the position type
    of the records returned by the source.
  * Reading the data - method ``read()`` can be used to read data from the
    source while respecting the boundaries defined by a given
    ``RangeTracker``.

  A runner will perform reading the source in two steps.

  (1) Method ``get_range_tracker()`` will be invoked with start and end
      positions to obtain a ``RangeTracker`` for the range of positions the
      runner intends to read. Source must define a default initial start and end
      position range. These positions must be used if the start and/or end
      positions passed to the method ``get_range_tracker()`` are ``None``
  (2) Method read() will be invoked with the ``RangeTracker`` obtained in the
      previous step.

  **Mutability**

  A ``_BoundedMongoSource`` object should not be mutated while
  its methods (for example, ``read()``) are being invoked by a runner. Runner
  implementations may invoke methods of ``_BoundedMongoSource`` objects through
  multi-threaded and/or reentrant execution modes.
  """

    def __init__(self, uri=None, db=None, coll=None, filter=None, projection=None, extra_client_params=None, bucket_auto=False):
        if False:
            return 10
        if extra_client_params is None:
            extra_client_params = {}
        if filter is None:
            filter = {}
        self.uri = uri
        self.db = db
        self.coll = coll
        self.filter = filter
        self.projection = projection
        self.spec = extra_client_params
        self.bucket_auto = bucket_auto

    def estimate_size(self):
        if False:
            for i in range(10):
                print('nop')
        with MongoClient(self.uri, **self.spec) as client:
            return client[self.db].command('collstats', self.coll).get('size')

    def _estimate_average_document_size(self):
        if False:
            print('Hello World!')
        with MongoClient(self.uri, **self.spec) as client:
            return client[self.db].command('collstats', self.coll).get('avgObjSize')

    def split(self, desired_bundle_size: int, start_position: Union[int, str, bytes, ObjectId]=None, stop_position: Union[int, str, bytes, ObjectId]=None):
        if False:
            while True:
                i = 10
        "Splits the source into a set of bundles.\n\n    Bundles should be approximately of size ``desired_bundle_size`` bytes.\n\n    Args:\n      desired_bundle_size: the desired size (in bytes) of the bundles returned.\n      start_position: if specified the given position must be used as the\n                      starting position of the first bundle.\n      stop_position: if specified the given position must be used as the ending\n                     position of the last bundle.\n    Returns:\n      an iterator of objects of type 'SourceBundle' that gives information about\n      the generated bundles.\n    "
        desired_bundle_size_in_mb = desired_bundle_size // 1024 // 1024
        desired_bundle_size_in_mb = max(desired_bundle_size_in_mb, 1)
        is_initial_split = start_position is None and stop_position is None
        (start_position, stop_position) = self._replace_none_positions(start_position, stop_position)
        if self.bucket_auto:
            split_keys = []
            weights = []
            for bucket in self._get_auto_buckets(desired_bundle_size_in_mb, start_position, stop_position, is_initial_split):
                split_keys.append({'_id': bucket['_id']['max']})
                weights.append(bucket['count'])
        else:
            split_keys = self._get_split_keys(desired_bundle_size_in_mb, start_position, stop_position)
            weights = itertools.cycle((desired_bundle_size_in_mb,))
        bundle_start = start_position
        for (split_key_id, weight) in zip(split_keys, weights):
            if bundle_start >= stop_position:
                break
            bundle_end = min(stop_position, split_key_id['_id'])
            yield iobase.SourceBundle(weight=weight, source=self, start_position=bundle_start, stop_position=bundle_end)
            bundle_start = bundle_end
        if bundle_start < stop_position:
            weight = 1 if self.bucket_auto else desired_bundle_size_in_mb
            yield iobase.SourceBundle(weight=weight, source=self, start_position=bundle_start, stop_position=stop_position)

    def get_range_tracker(self, start_position: Union[int, str, ObjectId]=None, stop_position: Union[int, str, ObjectId]=None) -> Union[_ObjectIdRangeTracker, OffsetRangeTracker, LexicographicKeyRangeTracker]:
        if False:
            while True:
                i = 10
        "Returns a RangeTracker for a given position range depending on type.\n\n    Args:\n      start_position: starting position of the range. If 'None' default start\n                      position of the source must be used.\n      stop_position:  ending position of the range. If 'None' default stop\n                      position of the source must be used.\n    Returns:\n      a ``_ObjectIdRangeTracker``, ``OffsetRangeTracker``\n      or ``LexicographicKeyRangeTracker`` depending on the given position range.\n    "
        (start_position, stop_position) = self._replace_none_positions(start_position, stop_position)
        if isinstance(start_position, ObjectId):
            return _ObjectIdRangeTracker(start_position, stop_position)
        if isinstance(start_position, int):
            return OffsetRangeTracker(start_position, stop_position)
        if isinstance(start_position, str):
            return LexicographicKeyRangeTracker(start_position, stop_position)
        raise NotImplementedError(f'RangeTracker for {type(start_position)} not implemented!')

    def read(self, range_tracker):
        if False:
            for i in range(10):
                print('nop')
        'Returns an iterator that reads data from the source.\n\n    The returned set of data must respect the boundaries defined by the given\n    ``RangeTracker`` object. For example:\n\n      * Returned set of data must be for the range\n        ``[range_tracker.start_position, range_tracker.stop_position)``. Note\n        that a source may decide to return records that start after\n        ``range_tracker.stop_position``. See documentation in class\n        ``RangeTracker`` for more details. Also, note that framework might\n        invoke ``range_tracker.try_split()`` to perform dynamic split\n        operations. range_tracker.stop_position may be updated\n        dynamically due to successful dynamic split operations.\n      * Method ``range_tracker.try_split()`` must be invoked for every record\n        that starts at a split point.\n      * Method ``range_tracker.record_current_position()`` may be invoked for\n        records that do not start at split points.\n\n    Args:\n      range_tracker: a ``RangeTracker`` whose boundaries must be respected\n                     when reading data from the source. A runner that reads this\n                     source muss pass a ``RangeTracker`` object that is not\n                     ``None``.\n    Returns:\n      an iterator of data read by the source.\n    '
        with MongoClient(self.uri, **self.spec) as client:
            all_filters = self._merge_id_filter(range_tracker.start_position(), range_tracker.stop_position())
            docs_cursor = client[self.db][self.coll].find(filter=all_filters, projection=self.projection).sort([('_id', ASCENDING)])
            for doc in docs_cursor:
                if not range_tracker.try_claim(doc['_id']):
                    return
                yield doc

    def display_data(self):
        if False:
            return 10
        'Returns the display data associated to a pipeline component.'
        res = super().display_data()
        res['database'] = self.db
        res['collection'] = self.coll
        res['filter'] = json.dumps(self.filter, default=json_util.default)
        res['projection'] = str(self.projection)
        res['bucket_auto'] = self.bucket_auto
        return res

    @staticmethod
    def _range_is_not_splittable(start_pos: Union[int, str, ObjectId], end_pos: Union[int, str, ObjectId]):
        if False:
            print('Hello World!')
        "Return `True` if splitting range doesn't make sense\n    (single document is not splittable),\n    Return `False` otherwise.\n    "
        return isinstance(start_pos, ObjectId) and start_pos >= _ObjectIdHelper.increment_id(end_pos, -1) or (isinstance(start_pos, int) and start_pos >= end_pos - 1) or (isinstance(start_pos, str) and start_pos >= end_pos)

    def _get_split_keys(self, desired_chunk_size_in_mb: int, start_pos: Union[int, str, ObjectId], end_pos: Union[int, str, ObjectId]):
        if False:
            for i in range(10):
                print('nop')
        'Calls MongoDB `splitVector` command\n    to get document ids at split position.\n    '
        if self._range_is_not_splittable(start_pos, end_pos):
            return []
        with MongoClient(self.uri, **self.spec) as client:
            name_space = '%s.%s' % (self.db, self.coll)
            return client[self.db].command('splitVector', name_space, keyPattern={'_id': 1}, min={'_id': start_pos}, max={'_id': end_pos}, maxChunkSize=desired_chunk_size_in_mb)['splitKeys']

    def _get_auto_buckets(self, desired_chunk_size_in_mb: int, start_pos: Union[int, str, ObjectId], end_pos: Union[int, str, ObjectId], is_initial_split: bool) -> list:
        if False:
            i = 10
            return i + 15
        'Use MongoDB `$bucketAuto` aggregation to split collection into bundles\n    instead of `splitVector` command, which does not work with MongoDB Atlas.\n    '
        if self._range_is_not_splittable(start_pos, end_pos):
            return []
        if is_initial_split and (not self.filter):
            size_in_mb = self.estimate_size() / float(1 << 20)
        else:
            documents_count = self._count_id_range(start_pos, end_pos)
            avg_document_size = self._estimate_average_document_size()
            size_in_mb = documents_count * avg_document_size / float(1 << 20)
        if size_in_mb == 0:
            return []
        bucket_count = math.ceil(size_in_mb / desired_chunk_size_in_mb)
        with beam.io.mongodbio.MongoClient(self.uri, **self.spec) as client:
            pipeline = [{'$match': self._merge_id_filter(start_pos, end_pos)}, {'$bucketAuto': {'groupBy': '$_id', 'buckets': bucket_count}}]
            buckets = list(client[self.db][self.coll].aggregate(pipeline, allowDiskUse=True))
            if buckets:
                buckets[-1]['_id']['max'] = end_pos
            return buckets

    def _merge_id_filter(self, start_position: Union[int, str, bytes, ObjectId], stop_position: Union[int, str, bytes, ObjectId]=None) -> dict:
        if False:
            i = 10
            return i + 15
        'Merge the default filter (if any) with refined _id field range\n    of range_tracker.\n    $gte specifies start position (inclusive)\n    and $lt specifies the end position (exclusive),\n    see more at\n    https://docs.mongodb.com/manual/reference/operator/query/gte/ and\n    https://docs.mongodb.com/manual/reference/operator/query/lt/\n    '
        if stop_position is None:
            id_filter = {'_id': {'$gte': start_position}}
        else:
            id_filter = {'_id': {'$gte': start_position, '$lt': stop_position}}
        if self.filter:
            all_filters = {'$and': [self.filter.copy(), id_filter]}
        else:
            all_filters = id_filter
        return all_filters

    def _get_head_document_id(self, sort_order):
        if False:
            print('Hello World!')
        with MongoClient(self.uri, **self.spec) as client:
            cursor = client[self.db][self.coll].find(filter={}, projection=[]).sort([('_id', sort_order)]).limit(1)
            try:
                return cursor[0]['_id']
            except IndexError:
                raise ValueError('Empty Mongodb collection')

    def _replace_none_positions(self, start_position, stop_position):
        if False:
            print('Hello World!')
        if start_position is None:
            start_position = self._get_head_document_id(ASCENDING)
        if stop_position is None:
            last_doc_id = self._get_head_document_id(DESCENDING)
            if isinstance(last_doc_id, ObjectId):
                stop_position = _ObjectIdHelper.increment_id(last_doc_id, 1)
            elif isinstance(last_doc_id, int):
                stop_position = last_doc_id + 1
            elif isinstance(last_doc_id, str):
                stop_position = last_doc_id + '\x00'
        return (start_position, stop_position)

    def _count_id_range(self, start_position, stop_position):
        if False:
            i = 10
            return i + 15
        'Number of documents between start_position (inclusive)\n    and stop_position (exclusive), respecting the custom filter if any.\n    '
        with MongoClient(self.uri, **self.spec) as client:
            return client[self.db][self.coll].count_documents(filter=self._merge_id_filter(start_position, stop_position))

class _ObjectIdHelper:
    """A Utility class to manipulate bson object ids."""

    @classmethod
    def id_to_int(cls, _id: Union[int, ObjectId]) -> int:
        if False:
            return 10
        "\n    Args:\n      _id: ObjectId required for each MongoDB document _id field.\n\n    Returns: Converted integer value of ObjectId's 12 bytes binary value.\n    "
        if isinstance(_id, int):
            return _id
        ints = struct.unpack('>III', _id.binary)
        return (ints[0] << 64) + (ints[1] << 32) + ints[2]

    @classmethod
    def int_to_id(cls, number):
        if False:
            i = 10
            return i + 15
        '\n    Args:\n      number(int): The integer value to be used to convert to ObjectId.\n\n    Returns: The ObjectId that has the 12 bytes binary converted from the\n      integer value.\n    '
        if number < 0 or number >= 1 << 96:
            raise ValueError('number value must be within [0, %s)' % (1 << 96))
        ints = [(number & 79228162495817593519834398720) >> 64, (number & 18446744069414584320) >> 32, number & 4294967295]
        number_bytes = struct.pack('>III', *ints)
        return ObjectId(number_bytes)

    @classmethod
    def increment_id(cls, _id: ObjectId, inc: int) -> ObjectId:
        if False:
            return 10
        '\n    Increment object_id binary value by inc value and return new object id.\n\n    Args:\n      _id: The `_id` to change.\n      inc(int): The incremental int value to be added to `_id`.\n\n    Returns:\n        `_id` incremented by `inc` value\n    '
        id_number = _ObjectIdHelper.id_to_int(_id)
        new_number = id_number + inc
        if new_number < 0 or new_number >= 1 << 96:
            raise ValueError('invalid incremental, inc value must be within [%s, %s)' % (0 - id_number, 1 << 96 - id_number))
        return _ObjectIdHelper.int_to_id(new_number)

class WriteToMongoDB(PTransform):
    """WriteToMongoDB is a ``PTransform`` that writes a ``PCollection`` of
  mongodb document to the configured MongoDB server.

  In order to make the document writes idempotent so that the bundles are
  retry-able without creating duplicates, the PTransform added 2 transformations
  before final write stage:
  a ``GenerateId`` transform and a ``Reshuffle`` transform.::

                  -----------------------------------------------
    Pipeline -->  |GenerateId --> Reshuffle --> WriteToMongoSink|
                  -----------------------------------------------
                                  (WriteToMongoDB)

  The ``GenerateId`` transform adds a random and unique*_id* field to the
  documents if they don't already have one, it uses the same format as MongoDB
  default. The ``Reshuffle`` transform makes sure that no fusion happens between
  ``GenerateId`` and the final write stage transform,so that the set of
  documents and their unique IDs are not regenerated if final write step is
  retried due to a failure. This prevents duplicate writes of the same document
  with different unique IDs.

  """

    def __init__(self, uri='mongodb://localhost:27017', db=None, coll=None, batch_size=100, extra_client_params=None):
        if False:
            i = 10
            return i + 15
        '\n\n    Args:\n      uri (str): The MongoDB connection string following the URI format\n      db (str): The MongoDB database name\n      coll (str): The MongoDB collection name\n      batch_size(int): Number of documents per bulk_write to  MongoDB,\n        default to 100\n      extra_client_params(dict): Optional `MongoClient\n       <https://api.mongodb.com/python/current/api/pymongo/mongo_client.html>`_\n       parameters as keyword arguments\n\n    Returns:\n      :class:`~apache_beam.transforms.ptransform.PTransform`\n\n    '
        if extra_client_params is None:
            extra_client_params = {}
        if not isinstance(db, str):
            raise ValueError('WriteToMongoDB db param must be specified as a string')
        if not isinstance(coll, str):
            raise ValueError('WriteToMongoDB coll param must be specified as a string')
        self._uri = uri
        self._db = db
        self._coll = coll
        self._batch_size = batch_size
        self._spec = extra_client_params

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        return pcoll | beam.ParDo(_GenerateObjectIdFn()) | Reshuffle() | beam.ParDo(_WriteMongoFn(self._uri, self._db, self._coll, self._batch_size, self._spec))

class _GenerateObjectIdFn(DoFn):

    def process(self, element, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if '_id' not in element:
            element['_id'] = objectid.ObjectId()
        yield element

class _WriteMongoFn(DoFn):

    def __init__(self, uri=None, db=None, coll=None, batch_size=100, extra_params=None):
        if False:
            print('Hello World!')
        if extra_params is None:
            extra_params = {}
        self.uri = uri
        self.db = db
        self.coll = coll
        self.spec = extra_params
        self.batch_size = batch_size
        self.batch = []

    def finish_bundle(self):
        if False:
            print('Hello World!')
        self._flush()

    def process(self, element, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.batch.append(element)
        if len(self.batch) >= self.batch_size:
            self._flush()

    def _flush(self):
        if False:
            while True:
                i = 10
        if len(self.batch) == 0:
            return
        with _MongoSink(self.uri, self.db, self.coll, self.spec) as sink:
            sink.write(self.batch)
            self.batch = []

    def display_data(self):
        if False:
            print('Hello World!')
        res = super().display_data()
        res['database'] = self.db
        res['collection'] = self.coll
        res['batch_size'] = self.batch_size
        return res

class _MongoSink:

    def __init__(self, uri=None, db=None, coll=None, extra_params=None):
        if False:
            print('Hello World!')
        if extra_params is None:
            extra_params = {}
        self.uri = uri
        self.db = db
        self.coll = coll
        self.spec = extra_params
        self.client = None

    def write(self, documents):
        if False:
            print('Hello World!')
        if self.client is None:
            self.client = MongoClient(host=self.uri, **self.spec)
        requests = []
        for doc in documents:
            requests.append(ReplaceOne(filter={'_id': doc.get('_id', None)}, replacement=doc, upsert=True))
        resp = self.client[self.db][self.coll].bulk_write(requests)
        _LOGGER.debug('BulkWrite to MongoDB result in nModified:%d, nUpserted:%d, nMatched:%d, Errors:%s' % (resp.modified_count, resp.upserted_count, resp.matched_count, resp.bulk_api_result.get('writeErrors')))

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        if self.client is None:
            self.client = MongoClient(host=self.uri, **self.spec)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        if self.client is not None:
            self.client.close()