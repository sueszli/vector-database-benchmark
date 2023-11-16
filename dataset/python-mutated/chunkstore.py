import hashlib
import logging
from collections import defaultdict
from itertools import groupby
import pymongo
from bson.binary import Binary
from pandas import DataFrame, Series
from pymongo.errors import OperationFailure
from .date_chunker import DateChunker, START, END
from .passthrough_chunker import PassthroughChunker
from .._util import indent, mongo_count, enable_sharding
from ..decorators import mongo_retry
from ..exceptions import NoDataFoundException
from ..serialization.numpy_arrays import FrametoArraySerializer, DATA, METADATA, COLUMNS
logger = logging.getLogger(__name__)
CHUNK_STORE_TYPE = 'ChunkStoreV1'
SYMBOL = 'sy'
SHA = 'sh'
CHUNK_SIZE = 'cs'
CHUNK_COUNT = 'cc'
SEGMENT = 'sg'
APPEND_COUNT = 'ac'
LEN = 'l'
SERIALIZER = 'se'
CHUNKER = 'ch'
USERMETA = 'u'
MAX_CHUNK_SIZE = 15 * 1024 * 1024
SER_MAP = {FrametoArraySerializer.TYPE: FrametoArraySerializer()}
CHUNKER_MAP = {DateChunker.TYPE: DateChunker(), PassthroughChunker.TYPE: PassthroughChunker()}

class ChunkStore(object):

    @classmethod
    def initialize_library(cls, arctic_lib, hashed=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ChunkStore(arctic_lib)._ensure_index()
        logger.info('Trying to enable sharding...')
        try:
            enable_sharding(arctic_lib.arctic, arctic_lib.get_name(), hashed=hashed, key=SYMBOL)
        except OperationFailure as e:
            logger.warning("Library created, but couldn't enable sharding: %s. This is OK if you're not 'admin'" % str(e))

    @mongo_retry
    def _ensure_index(self):
        if False:
            while True:
                i = 10
        self._symbols.create_index([(SYMBOL, pymongo.ASCENDING)], unique=True, background=True)
        self._collection.create_index([(SYMBOL, pymongo.HASHED)], background=True)
        self._collection.create_index([(SYMBOL, pymongo.ASCENDING), (SHA, pymongo.ASCENDING)], unique=True, background=True)
        self._collection.create_index([(SYMBOL, pymongo.ASCENDING), (START, pymongo.ASCENDING), (SEGMENT, pymongo.ASCENDING), (END, pymongo.ASCENDING)], unique=True, background=True)
        self._collection.create_index([(SYMBOL, pymongo.ASCENDING), (START, pymongo.ASCENDING), (SEGMENT, pymongo.ASCENDING)], unique=True, background=True)
        self._collection.create_index([(SEGMENT, pymongo.ASCENDING)], unique=False, background=True)
        self._mdata.create_index([(SYMBOL, pymongo.ASCENDING), (START, pymongo.ASCENDING), (END, pymongo.ASCENDING)], unique=True, background=True)

    def __init__(self, arctic_lib):
        if False:
            while True:
                i = 10
        self._arctic_lib = arctic_lib
        self.serializer = FrametoArraySerializer()
        self._allow_secondary = self._arctic_lib.arctic._allow_secondary
        self._reset()

    @mongo_retry
    def _reset(self):
        if False:
            print('Hello World!')
        self._collection = self._arctic_lib.get_top_level_collection()
        self._symbols = self._collection.symbols
        self._mdata = self._collection.metadata
        self._audit = self._collection.audit

    def __getstate__(self):
        if False:
            print('Hello World!')
        return {'arctic_lib': self._arctic_lib}

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        return ChunkStore.__init__(self, state['arctic_lib'])

    def __str__(self):
        if False:
            print('Hello World!')
        return '<%s at %s>\n%s' % (self.__class__.__name__, hex(id(self)), indent(str(self._arctic_lib), 4))

    def __repr__(self):
        if False:
            return 10
        return str(self)

    def _checksum(self, fields, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checksum the passed in dictionary\n        '
        sha = hashlib.sha1()
        for field in fields:
            sha.update(field)
        sha.update(data)
        return Binary(sha.digest())

    def delete(self, symbol, chunk_range=None, audit=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete all chunks for a symbol, or optionally, chunks within a range\n\n        Parameters\n        ----------\n        symbol : str\n            symbol name for the item\n        chunk_range: range object\n            a date range to delete\n        audit: dict\n            dict to store in the audit log\n        '
        if chunk_range is not None:
            sym = self._get_symbol_info(symbol)
            df = self.read(symbol, chunk_range=chunk_range, filter_data=False)
            row_adjust = len(df)
            if not df.empty:
                df = CHUNKER_MAP[sym[CHUNKER]].exclude(df, chunk_range)
                query = {SYMBOL: symbol}
                query.update(CHUNKER_MAP[sym[CHUNKER]].to_mongo(chunk_range))
                self._collection.delete_many(query)
                self._mdata.delete_many(query)
                self.update(symbol, df)
                sym = self._get_symbol_info(symbol)
                sym[LEN] -= row_adjust
                sym[CHUNK_COUNT] = mongo_count(self._collection, filter={SYMBOL: symbol})
                self._symbols.replace_one({SYMBOL: symbol}, sym)
        else:
            query = {SYMBOL: symbol}
            self._collection.delete_many(query)
            self._symbols.delete_many(query)
            self._mdata.delete_many(query)
        if audit is not None:
            audit['symbol'] = symbol
            if chunk_range is not None:
                audit['rows_deleted'] = row_adjust
                audit['action'] = 'range delete'
            else:
                audit['action'] = 'symbol delete'
            self._audit.insert_one(audit)

    def list_symbols(self, partial_match=None):
        if False:
            while True:
                i = 10
        '\n        Returns all symbols in the library\n\n        Parameters\n        ----------\n        partial: None or str\n            if not none, use this string to do a partial match on symbol names\n\n        Returns\n        -------\n        list of str\n        '
        symbols = self._symbols.distinct(SYMBOL)
        if partial_match is None:
            return symbols
        return [x for x in symbols if partial_match in x]

    def _get_symbol_info(self, symbol):
        if False:
            while True:
                i = 10
        if isinstance(symbol, list):
            return list(self._symbols.find({SYMBOL: {'$in': symbol}}))
        return self._symbols.find_one({SYMBOL: symbol})

    def rename(self, from_symbol, to_symbol, audit=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Rename a symbol\n\n        Parameters\n        ----------\n        from_symbol: str\n            the existing symbol that will be renamed\n        to_symbol: str\n            the new symbol name\n        audit: dict\n            audit information\n        '
        sym = self._get_symbol_info(from_symbol)
        if not sym:
            raise NoDataFoundException('No data found for %s' % from_symbol)
        if self._get_symbol_info(to_symbol) is not None:
            raise Exception('Symbol %s already exists' % to_symbol)
        mongo_retry(self._collection.update_many)({SYMBOL: from_symbol}, {'$set': {SYMBOL: to_symbol}})
        mongo_retry(self._symbols.update_one)({SYMBOL: from_symbol}, {'$set': {SYMBOL: to_symbol}})
        mongo_retry(self._mdata.update_many)({SYMBOL: from_symbol}, {'$set': {SYMBOL: to_symbol}})
        mongo_retry(self._audit.update_many)({'symbol': from_symbol}, {'$set': {'symbol': to_symbol}})
        if audit is not None:
            audit['symbol'] = to_symbol
            audit['action'] = 'symbol rename'
            audit['old_symbol'] = from_symbol
            self._audit.insert_one(audit)

    def read(self, symbol, chunk_range=None, filter_data=True, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Reads data for a given symbol from the database.\n\n        Parameters\n        ----------\n        symbol: str, or list of str\n            the symbol(s) to retrieve\n        chunk_range: object\n            corresponding range object for the specified chunker (for\n            DateChunker it is a DateRange object or a DatetimeIndex,\n            as returned by pandas.date_range\n        filter_data: boolean\n            perform chunk level filtering on the data (see filter in _chunker)\n            only applicable when chunk_range is specified\n        kwargs: ?\n            values passed to the serializer. Varies by serializer\n\n        Returns\n        -------\n        DataFrame or Series, or in the case when multiple symbols are given,\n        returns a dict of symbols (symbol -> dataframe/series)\n        '
        if not isinstance(symbol, list):
            symbol = [symbol]
        sym = self._get_symbol_info(symbol)
        if not sym:
            raise NoDataFoundException('No data found for %s' % symbol)
        spec = {SYMBOL: {'$in': symbol}}
        chunker = CHUNKER_MAP[sym[0][CHUNKER]]
        deser = SER_MAP[sym[0][SERIALIZER]].deserialize
        if chunk_range is not None:
            spec.update(chunker.to_mongo(chunk_range))
        by_start_segment = [(SYMBOL, pymongo.ASCENDING), (START, pymongo.ASCENDING), (SEGMENT, pymongo.ASCENDING)]
        segment_cursor = self._collection.find(spec, sort=by_start_segment)
        chunks = defaultdict(list)
        for (_, segments) in groupby(segment_cursor, key=lambda x: (x[START], x[SYMBOL])):
            segments = list(segments)
            mdata = self._mdata.find_one({SYMBOL: segments[0][SYMBOL], START: segments[0][START], END: segments[0][END]})
            chunk_data = b''.join([doc[DATA] for doc in segments])
            chunks[segments[0][SYMBOL]].append({DATA: chunk_data, METADATA: mdata})
        skip_filter = not filter_data or chunk_range is None
        if len(symbol) > 1:
            return {sym: deser(chunks[sym], **kwargs) if skip_filter else chunker.filter(deser(chunks[sym], **kwargs), chunk_range) for sym in symbol}
        else:
            return deser(chunks[symbol[0]], **kwargs) if skip_filter else chunker.filter(deser(chunks[symbol[0]], **kwargs), chunk_range)

    def read_audit_log(self, symbol=None):
        if False:
            while True:
                i = 10
        "\n        Reads the audit log\n\n        Parameters\n        ----------\n        symbol: str\n            optionally only retrieve specific symbol's audit information\n\n        Returns\n        -------\n        list of dicts\n        "
        if symbol:
            return [x for x in self._audit.find({'symbol': symbol}, {'_id': False})]
        return [x for x in self._audit.find({}, {'_id': False})]

    def write(self, symbol, item, metadata=None, chunker=DateChunker(), audit=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Writes data from item to symbol in the database\n\n        Parameters\n        ----------\n        symbol: str\n            the symbol that will be used to reference the written data\n        item: Dataframe or Series\n            the data to write the database\n        metadata: ?\n            optional per symbol metadata\n        chunker: Object of type Chunker\n            A chunker that chunks the data in item\n        audit: dict\n            audit information\n        kwargs:\n            optional keyword args that are passed to the chunker. Includes:\n            chunk_size:\n                used by chunker to break data into discrete chunks.\n                see specific chunkers for more information about this param.\n            func: function\n                function to apply to each chunk before writing. Function\n                can not modify the date column.\n        '
        if not isinstance(item, (DataFrame, Series)):
            raise Exception('Can only chunk DataFrames and Series')
        self._arctic_lib.check_quota()
        previous_shas = []
        doc = {}
        meta = {}
        doc[SYMBOL] = symbol
        doc[LEN] = len(item)
        doc[SERIALIZER] = self.serializer.TYPE
        doc[CHUNKER] = chunker.TYPE
        doc[USERMETA] = metadata
        sym = self._get_symbol_info(symbol)
        if sym:
            previous_shas = set([Binary(x[SHA]) for x in self._collection.find({SYMBOL: symbol}, projection={SHA: True, '_id': False})])
        ops = []
        meta_ops = []
        chunk_count = 0
        for (start, end, chunk_size, record) in chunker.to_chunks(item, **kwargs):
            chunk_count += 1
            data = self.serializer.serialize(record)
            doc[CHUNK_SIZE] = chunk_size
            doc[METADATA] = {'columns': data[METADATA][COLUMNS] if COLUMNS in data[METADATA] else ''}
            meta = data[METADATA]
            for i in range(int(len(data[DATA]) / MAX_CHUNK_SIZE + 1)):
                chunk = {DATA: Binary(data[DATA][i * MAX_CHUNK_SIZE:(i + 1) * MAX_CHUNK_SIZE])}
                chunk[SEGMENT] = i
                chunk[START] = meta[START] = start
                chunk[END] = meta[END] = end
                chunk[SYMBOL] = meta[SYMBOL] = symbol
                dates = [chunker.chunk_to_str(start), chunker.chunk_to_str(end), str(chunk[SEGMENT]).encode('ascii')]
                chunk[SHA] = self._checksum(dates, chunk[DATA])
                meta_ops.append(pymongo.ReplaceOne({SYMBOL: symbol, START: start, END: end}, meta, upsert=True))
                if chunk[SHA] not in previous_shas:
                    ops.append(pymongo.UpdateOne({SYMBOL: symbol, START: start, END: end, SEGMENT: chunk[SEGMENT]}, {'$set': chunk}, upsert=True))
                else:
                    previous_shas.remove(chunk[SHA])
        if ops:
            self._collection.bulk_write(ops, ordered=False)
        if meta_ops:
            self._mdata.bulk_write(meta_ops, ordered=False)
        doc[CHUNK_COUNT] = chunk_count
        doc[APPEND_COUNT] = 0
        if previous_shas:
            mongo_retry(self._collection.delete_many)({SYMBOL: symbol, SHA: {'$in': list(previous_shas)}})
        mongo_retry(self._symbols.update_one)({SYMBOL: symbol}, {'$set': doc}, upsert=True)
        if audit is not None:
            audit['symbol'] = symbol
            audit['action'] = 'write'
            audit['chunks'] = chunk_count
            self._audit.insert_one(audit)

    def __update(self, sym, item, metadata=None, combine_method=None, chunk_range=None, audit=None):
        if False:
            print('Hello World!')
        '\n        helper method used by update and append since they very closely\n        resemble eachother. Really differ only by the combine method.\n        append will combine existing date with new data (within a chunk),\n        whereas update will replace existing data with new data (within a\n        chunk).\n        '
        if not isinstance(item, (DataFrame, Series)):
            raise Exception('Can only chunk DataFrames and Series')
        self._arctic_lib.check_quota()
        symbol = sym[SYMBOL]
        if chunk_range is not None:
            self.delete(symbol, chunk_range)
            sym = self._get_symbol_info(symbol)
        ops = []
        meta_ops = []
        chunker = CHUNKER_MAP[sym[CHUNKER]]
        appended = 0
        new_chunks = 0
        for (start, end, _, record) in chunker.to_chunks(item, chunk_size=sym[CHUNK_SIZE]):
            df = self.read(symbol, chunk_range=chunker.to_range(start, end), filter_data=False)
            if len(df) > 0:
                record = combine_method(df, record)
                if record is None or record.equals(df):
                    continue
                sym[APPEND_COUNT] += len(record) - len(df)
                appended += len(record) - len(df)
                sym[LEN] += len(record) - len(df)
            else:
                sym[CHUNK_COUNT] += 1
                new_chunks += 1
                sym[LEN] += len(record)
            data = SER_MAP[sym[SERIALIZER]].serialize(record)
            meta = data[METADATA]
            chunk_count = int(len(data[DATA]) / MAX_CHUNK_SIZE + 1)
            seg_count = mongo_count(self._collection, filter={SYMBOL: symbol, START: start, END: end})
            if seg_count > chunk_count:
                self._collection.delete_many({SYMBOL: symbol, START: start, END: end, SEGMENT: {'$gte': chunk_count}})
            for i in range(chunk_count):
                chunk = {DATA: Binary(data[DATA][i * MAX_CHUNK_SIZE:(i + 1) * MAX_CHUNK_SIZE])}
                chunk[SEGMENT] = i
                chunk[START] = start
                chunk[END] = end
                chunk[SYMBOL] = symbol
                dates = [chunker.chunk_to_str(start), chunker.chunk_to_str(end), str(chunk[SEGMENT]).encode('ascii')]
                sha = self._checksum(dates, data[DATA])
                chunk[SHA] = sha
                ops.append(pymongo.UpdateOne({SYMBOL: symbol, START: start, END: end, SEGMENT: chunk[SEGMENT]}, {'$set': chunk}, upsert=True))
                meta_ops.append(pymongo.UpdateOne({SYMBOL: symbol, START: start, END: end}, {'$set': meta}, upsert=True))
        if ops:
            self._collection.bulk_write(ops, ordered=False)
            self._mdata.bulk_write(meta_ops, ordered=False)
        sym[USERMETA] = metadata
        self._symbols.replace_one({SYMBOL: symbol}, sym)
        if audit is not None:
            if new_chunks > 0:
                audit['new_chunks'] = new_chunks
            if appended > 0:
                audit['appended_rows'] = appended
            self._audit.insert_one(audit)

    def append(self, symbol, item, upsert=False, metadata=None, audit=None, **kwargs):
        if False:
            return 10
        "\n        Appends data from item to symbol's data in the database.\n\n        Is not idempotent\n\n        Parameters\n        ----------\n        symbol: str\n            the symbol for the given item in the DB\n        item: DataFrame or Series\n            the data to append\n        upsert:\n            write data if symbol does not exist\n        metadata: ?\n            optional per symbol metadata\n        audit: dict\n            optional audit information\n        kwargs:\n            passed to write if upsert is true and symbol does not exist\n        "
        sym = self._get_symbol_info(symbol)
        if not sym:
            if upsert:
                return self.write(symbol, item, metadata=metadata, audit=audit, **kwargs)
            else:
                raise NoDataFoundException('Symbol does not exist.')
        if audit is not None:
            audit['symbol'] = symbol
            audit['action'] = 'append'
        self.__update(sym, item, metadata=metadata, combine_method=SER_MAP[sym[SERIALIZER]].combine, audit=audit)

    def update(self, symbol, item, metadata=None, chunk_range=None, upsert=False, audit=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Overwrites data in DB with data in item for the given symbol.\n\n        Is idempotent\n\n        Parameters\n        ----------\n        symbol: str\n            the symbol for the given item in the DB\n        item: DataFrame or Series\n            the data to update\n        metadata: ?\n            optional per symbol metadata\n        chunk_range: None, or a range object\n            If a range is specified, it will clear/delete the data within the\n            range and overwrite it with the data in item. This allows the user\n            to update with data that might only be a subset of the\n            original data.\n        upsert: bool\n            if True, will write the data even if the symbol does not exist.\n        audit: dict\n            optional audit information\n        kwargs:\n            optional keyword args passed to write during an upsert. Includes:\n            chunk_size\n            chunker\n        '
        sym = self._get_symbol_info(symbol)
        if not sym:
            if upsert:
                return self.write(symbol, item, metadata=metadata, audit=audit, **kwargs)
            else:
                raise NoDataFoundException('Symbol does not exist.')
        if audit is not None:
            audit['symbol'] = symbol
            audit['action'] = 'update'
        if chunk_range is not None:
            if len(CHUNKER_MAP[sym[CHUNKER]].filter(item, chunk_range)) == 0:
                raise Exception('Range must be inclusive of data')
            self.__update(sym, item, metadata=metadata, combine_method=self.serializer.combine, chunk_range=chunk_range, audit=audit)
        else:
            self.__update(sym, item, metadata=metadata, combine_method=lambda old, new: new, chunk_range=chunk_range, audit=audit)

    def get_info(self, symbol):
        if False:
            i = 10
            return i + 15
        '\n        Returns information about the symbol, in a dictionary\n\n        Parameters\n        ----------\n        symbol: str\n            the symbol for the given item in the DB\n\n        Returns\n        -------\n        dictionary\n        '
        sym = self._get_symbol_info(symbol)
        if not sym:
            raise NoDataFoundException('Symbol does not exist.')
        ret = {}
        ret['chunk_count'] = sym[CHUNK_COUNT]
        ret['len'] = sym[LEN]
        ret['appended_rows'] = sym[APPEND_COUNT]
        ret['metadata'] = sym[METADATA] if METADATA in sym else None
        ret['chunker'] = sym[CHUNKER]
        ret['chunk_size'] = sym[CHUNK_SIZE] if CHUNK_SIZE in sym else 0
        ret['serializer'] = sym[SERIALIZER]
        return ret

    def read_metadata(self, symbol):
        if False:
            while True:
                i = 10
        '\n        Reads user defined metadata out for the given symbol\n\n        Parameters\n        ----------\n        symbol: str\n            symbol for the given item in the DB\n\n        Returns\n        -------\n        ?\n        '
        sym = self._get_symbol_info(symbol)
        if not sym:
            raise NoDataFoundException('Symbol does not exist.')
        x = self._symbols.find_one({SYMBOL: symbol})
        return x[USERMETA] if USERMETA in x else None

    def write_metadata(self, symbol, metadata):
        if False:
            while True:
                i = 10
        '\n        writes user defined metadata for the given symbol\n\n        Parameters\n        ----------\n        symbol: str\n            symbol for the given item in the DB\n        metadata: ?\n            metadata to write\n        '
        sym = self._get_symbol_info(symbol)
        if not sym:
            raise NoDataFoundException('Symbol does not exist.')
        sym[USERMETA] = metadata
        self._symbols.replace_one({SYMBOL: symbol}, sym)

    def get_chunk_ranges(self, symbol, chunk_range=None, reverse=False):
        if False:
            return 10
        '\n        Returns a generator of (Start, End) tuples for each chunk in the symbol\n\n        Parameters\n        ----------\n        symbol: str\n            the symbol for the given item in the DB\n        chunk_range: None, or a range object\n            allows you to subset the chunks by range\n        reverse: boolean\n            return the chunk ranges in reverse order\n\n        Returns\n        -------\n        generator\n        '
        sym = self._get_symbol_info(symbol)
        if not sym:
            raise NoDataFoundException('Symbol does not exist.')
        c = CHUNKER_MAP[sym[CHUNKER]]
        spec = {SYMBOL: symbol, SEGMENT: 0}
        if chunk_range is not None:
            spec.update(CHUNKER_MAP[sym[CHUNKER]].to_mongo(chunk_range))
        for x in self._collection.find(spec, projection=[START, END], sort=[(START, pymongo.ASCENDING if not reverse else pymongo.DESCENDING)]):
            yield (c.chunk_to_str(x[START]), c.chunk_to_str(x[END]))

    def iterator(self, symbol, chunk_range=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns a generator that accesses each chunk in ascending order\n\n        Parameters\n        ----------\n        symbol: str\n            the symbol for the given item in the DB\n        chunk_range: None, or a range object\n            allows you to subset the chunks by range\n\n        Returns\n        -------\n        generator\n        '
        sym = self._get_symbol_info(symbol)
        if not sym:
            raise NoDataFoundException('Symbol does not exist.')
        c = CHUNKER_MAP[sym[CHUNKER]]
        for chunk in list(self.get_chunk_ranges(symbol, chunk_range=chunk_range)):
            yield self.read(symbol, chunk_range=c.to_range(chunk[0], chunk[1]), **kwargs)

    def reverse_iterator(self, symbol, chunk_range=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns a generator that accesses each chunk in descending order\n\n        Parameters\n        ----------\n        symbol: str\n            the symbol for the given item in the DB\n        chunk_range: None, or a range object\n            allows you to subset the chunks by range\n\n        Returns\n        -------\n        generator\n        '
        sym = self._get_symbol_info(symbol)
        if not sym:
            raise NoDataFoundException('Symbol does not exist.')
        c = CHUNKER_MAP[sym[CHUNKER]]
        for chunk in list(self.get_chunk_ranges(symbol, chunk_range=chunk_range, reverse=True)):
            yield self.read(symbol, chunk_range=c.to_range(chunk[0], chunk[1]), **kwargs)

    def stats(self):
        if False:
            while True:
                i = 10
        '\n        Return storage statistics about the library\n\n        Returns\n        -------\n        dictionary of storage stats\n        '
        res = {}
        db = self._collection.database
        conn = db.connection
        res['sharding'] = {}
        try:
            sharding = conn.config.databases.find_one({'_id': db.name})
            if sharding:
                res['sharding'].update(sharding)
            res['sharding']['collections'] = list(conn.config.collections.find({'_id': {'$regex': '^' + db.name + '\\..*'}}))
        except OperationFailure:
            pass
        res['dbstats'] = db.command('dbstats')
        res['chunks'] = db.command('collstats', self._collection.name)
        res['symbols'] = db.command('collstats', self._symbols.name)
        res['metadata'] = db.command('collstats', self._mdata.name)
        res['totals'] = {'count': res['chunks']['count'], 'size': res['chunks']['size'] + res['symbols']['size'] + res['metadata']['size']}
        return res

    def has_symbol(self, symbol):
        if False:
            return 10
        '\n        Check if symbol exists in collection\n\n        Parameters\n        ----------\n        symbol: str\n            The symbol to look up in the collection\n\n        Returns\n        -------\n        bool\n        '
        return self._get_symbol_info(symbol) is not None