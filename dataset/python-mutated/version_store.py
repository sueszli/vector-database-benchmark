import logging
from datetime import datetime as dt, timedelta
import bson
import pymongo
from pymongo import ReadPreference
from pymongo.errors import OperationFailure, AutoReconnect, DuplicateKeyError
from ._pickle_store import PickleStore
from ._version_store_utils import cleanup, get_symbol_alive_shas, _get_symbol_pointer_cfgs
from .versioned_item import VersionedItem
from .._config import STRICT_WRITE_HANDLER_MATCH, FW_POINTERS_REFS_KEY, FW_POINTERS_CONFIG_KEY, FwPointersCfg
from .._util import indent, enable_sharding, mongo_count, get_fwptr_config
from ..date import mktz, datetime_to_ms, ms_to_datetime
from ..decorators import mongo_retry
from ..exceptions import NoDataFoundException, DuplicateSnapshotException, ArcticException
from ..hooks import log_exception
logger = logging.getLogger(__name__)
VERSION_STORE_TYPE = 'VersionStore'
_TYPE_HANDLERS = []
ARCTIC_VERSION = None
ARCTIC_VERSION_NUMERICAL = None

def register_version(version, numerical):
    if False:
        print('Hello World!')
    global ARCTIC_VERSION, ARCTIC_VERSION_NUMERICAL
    ARCTIC_VERSION = version
    ARCTIC_VERSION_NUMERICAL = numerical

def register_versioned_storage(storageClass):
    if False:
        print('Hello World!')
    existing_instances = [i for (i, v) in enumerate(_TYPE_HANDLERS) if str(v.__class__) == str(storageClass)]
    if existing_instances:
        for i in existing_instances:
            _TYPE_HANDLERS[i] = storageClass()
    else:
        _TYPE_HANDLERS.append(storageClass())
    return storageClass

class VersionStore(object):
    _bson_handler = PickleStore()

    @classmethod
    def initialize_library(cls, arctic_lib, hashed=True, **kwargs):
        if False:
            return 10
        c = arctic_lib.get_top_level_collection()
        if 'strict_write_handler' in kwargs:
            arctic_lib.set_library_metadata('STRICT_WRITE_HANDLER_MATCH', bool(kwargs.pop('strict_write_handler')))
        for th in _TYPE_HANDLERS:
            th.initialize_library(arctic_lib, **kwargs)
        VersionStore._bson_handler.initialize_library(arctic_lib, **kwargs)
        VersionStore(arctic_lib)._ensure_index()
        logger.info('Trying to enable sharding...')
        try:
            enable_sharding(arctic_lib.arctic, arctic_lib.get_name(), hashed=hashed)
        except OperationFailure as e:
            logger.warning("Library created, but couldn't enable sharding: %s. This is OK if you're not 'admin'" % str(e))

    @mongo_retry
    def _last_version_seqnum(self, symbol):
        if False:
            i = 10
            return i + 15
        last_seq = self._version_nums.find_one({'symbol': symbol})
        return last_seq['version'] if last_seq else 0

    @mongo_retry
    def _ensure_index(self):
        if False:
            print('Hello World!')
        collection = self._collection
        collection.snapshots.create_index([('name', pymongo.ASCENDING)], unique=True, background=True)
        collection.versions.create_index([('symbol', pymongo.ASCENDING), ('_id', pymongo.DESCENDING)], background=True)
        collection.versions.create_index([('symbol', pymongo.ASCENDING), ('version', pymongo.DESCENDING)], unique=True, background=True)
        collection.versions.create_index([('symbol', pymongo.ASCENDING), ('version', pymongo.DESCENDING), ('metadata.deleted', pymongo.ASCENDING)], name='versionstore_idx', background=True)
        collection.versions.create_index([('parent', pymongo.ASCENDING)], background=True)
        collection.version_nums.create_index('symbol', unique=True, background=True)
        for th in _TYPE_HANDLERS:
            th._ensure_index(collection)

    def __init__(self, arctic_lib):
        if False:
            print('Hello World!')
        self._arctic_lib = arctic_lib
        self._allow_secondary = self._arctic_lib.arctic._allow_secondary
        self._reset()
        self._with_strict_handler = None

    @property
    def _with_strict_handler_match(self):
        if False:
            return 10
        if self._with_strict_handler is None:
            strict_meta = self._arctic_lib.get_library_metadata('STRICT_WRITE_HANDLER_MATCH')
            self._with_strict_handler = STRICT_WRITE_HANDLER_MATCH if strict_meta is None else strict_meta
        return self._with_strict_handler

    @mongo_retry
    def _reset(self):
        if False:
            for i in range(10):
                print('nop')
        self._collection = self._arctic_lib.get_top_level_collection()
        self._audit = self._collection.audit
        self._snapshots = self._collection.snapshots
        self._versions = self._collection.versions
        self._version_nums = self._collection.version_nums

    def __getstate__(self):
        if False:
            return 10
        return {'arctic_lib': self._arctic_lib}

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        return VersionStore.__init__(self, state['arctic_lib'])

    def __str__(self):
        if False:
            return 10
        return '<%s at %s>\n%s' % (self.__class__.__name__, hex(id(self)), indent(str(self._arctic_lib), 4))

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self)

    def _read_preference(self, allow_secondary):
        if False:
            i = 10
            return i + 15
        " Return the mongo read preference given an 'allow_secondary' argument\n        "
        allow_secondary = self._allow_secondary if allow_secondary is None else allow_secondary
        return ReadPreference.NEAREST if allow_secondary else ReadPreference.PRIMARY

    @mongo_retry
    def list_symbols(self, all_symbols=False, snapshot=None, regex=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Return the symbols in this library.\n\n        Parameters\n        ----------\n        all_symbols : `bool`\n            If True returns all symbols under all snapshots, even if the symbol has been deleted\n            in the current version (i.e. it exists under a snapshot... Default: False\n        snapshot : `str`\n            Return the symbols available under the snapshot.\n        regex : `str`\n            filter symbols by the passed in regular expression\n        kwargs :\n            kwarg keys are used as fields to query for symbols with metadata matching\n            the kwargs query\n\n        Returns\n        -------\n        String list of symbols in the library\n        '
        query = {}
        if regex is not None:
            query['symbol'] = {'$regex': regex}
        if kwargs:
            for (k, v) in kwargs.items():
                query['metadata.' + k] = v
        if snapshot is not None:
            try:
                query['parent'] = self._snapshots.find_one({'name': snapshot})['_id']
            except TypeError:
                raise NoDataFoundException('No snapshot %s in library %s' % (snapshot, self._arctic_lib.get_name()))
        elif all_symbols:
            return self._versions.find(query).distinct('symbol')
        pipeline = []
        if query:
            pipeline.append({'$match': query})
        pipeline.extend([{'$sort': bson.SON([('symbol', pymongo.ASCENDING), ('version', pymongo.DESCENDING)])}, {'$group': {'_id': '$symbol', 'deleted': {'$first': '$metadata.deleted'}}}, {'$match': {'deleted': {'$ne': True}}}])
        return sorted([x['_id'] for x in self._versions.aggregate(pipeline, allowDiskUse=True)])

    @mongo_retry
    def has_symbol(self, symbol, as_of=None):
        if False:
            return 10
        "\n        Return True if the 'symbol' exists in this library AND the symbol\n        isn't deleted in the specified as_of.\n\n        It's possible for a deleted symbol to exist in older snapshots.\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        as_of : `str` or int or `datetime.datetime`\n            Return the data as it was as_of the point in time.\n            `int` : specific version number\n            `str` : snapshot name which contains the version\n            `datetime.datetime` : the version of the data that existed as_of the requested point in time\n        "
        try:
            self._read_metadata(symbol, as_of=as_of, read_preference=ReadPreference.PRIMARY)
            return True
        except NoDataFoundException:
            return False

    def read_audit_log(self, symbol=None, message=None):
        if False:
            return 10
        '\n        Return the audit log associated with a given symbol\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        '
        query = {}
        if symbol:
            if isinstance(symbol, str):
                query['symbol'] = {'$regex': symbol}
            else:
                query['symbol'] = {'$in': list(symbol)}
        if message is not None:
            query['message'] = message

        def _pop_id(x):
            if False:
                i = 10
                return i + 15
            x.pop('_id')
            return x
        return [_pop_id(x) for x in self._audit.find(query, sort=[('_id', -1)])]

    def list_versions(self, symbol=None, snapshot=None, latest_only=False):
        if False:
            return 10
        '\n        Return a list of versions filtered by the passed in parameters.\n\n        Parameters\n        ----------\n        symbol : `str`\n            Symbol to return versions for.  If None returns versions across all\n            symbols in the library.\n        snapshot : `str`\n            Return the versions contained in the named snapshot\n        latest_only : `bool`\n            Only include the latest version for a specific symbol\n\n        Returns\n        -------\n        List of dictionaries describing the discovered versions in the library\n        '
        if symbol is None:
            symbols = self.list_symbols(snapshot=snapshot)
        else:
            symbols = [symbol]
        query = {}
        if snapshot is not None:
            try:
                query['parent'] = self._snapshots.find_one({'name': snapshot})['_id']
            except TypeError:
                raise NoDataFoundException('No snapshot %s in library %s' % (snapshot, self._arctic_lib.get_name()))
        versions = []
        snapshots = {ss.get('_id'): ss.get('name') for ss in self._snapshots.find()}
        for symbol in symbols:
            query['symbol'] = symbol
            seen_symbols = set()
            for version in self._versions.find(query, projection=['symbol', 'version', 'parent', 'metadata.deleted'], sort=[('version', -1)]):
                if latest_only and version['symbol'] in seen_symbols:
                    continue
                seen_symbols.add(version['symbol'])
                meta = version.get('metadata')
                versions.append({'symbol': version['symbol'], 'version': version['version'], 'deleted': meta.get('deleted', False) if meta else False, 'date': ms_to_datetime(datetime_to_ms(version['_id'].generation_time)), 'snapshots': [snapshots[s] for s in version.get('parent', []) if s in snapshots]})
        return versions

    def _find_snapshots(self, parent_ids):
        if False:
            for i in range(10):
                print('nop')
        snapshots = []
        for p in parent_ids:
            snap = self._snapshots.find_one({'_id': p})
            if snap:
                snapshots.append(snap['name'])
            else:
                snapshots.append(str(p))
        return snapshots

    def _read_handler(self, version, symbol):
        if False:
            i = 10
            return i + 15
        handler = None
        for h in _TYPE_HANDLERS:
            if h.can_read(version, symbol):
                handler = h
                break
        if handler is None:
            handler = self._bson_handler
        return handler

    @staticmethod
    def handler_can_write_type(handler, data):
        if False:
            print('Hello World!')
        type_method = getattr(handler, 'can_write_type', None)
        if callable(type_method):
            return type_method(data)
        return False

    def _write_handler(self, version, symbol, data, **kwargs):
        if False:
            return 10
        handler = None
        for h in _TYPE_HANDLERS:
            if h.can_write(version, symbol, data, **kwargs):
                handler = h
                break
            if self._with_strict_handler_match and self.handler_can_write_type(h, data):
                raise ArcticException('Not falling back to default handler for %s' % symbol)
        if handler is None:
            version['type'] = 'default'
            handler = self._bson_handler
        return handler

    def read(self, symbol, as_of=None, date_range=None, from_version=None, allow_secondary=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Read data for the named symbol.  Returns a VersionedItem object with\n        a data and metdata element (as passed into write).\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        as_of : `str` or `int` or `datetime.datetime`\n            Return the data as it was as_of the point in time.\n            `int` : specific version number\n            `str` : snapshot name which contains the version\n            `datetime.datetime` : the version of the data that existed as_of the requested point in time\n        date_range: `arctic.date.DateRange`\n            DateRange to read data for.  Applies to Pandas data, with a DateTime index\n            returns only the part of the data that falls in the DateRange.\n        allow_secondary : `bool` or `None`\n            Override the default behavior for allowing reads from secondary members of a cluster:\n            `None` : use the settings from the top-level `Arctic` object used to query this version store.\n            `True` : allow reads from secondary members\n            `False` : only allow reads from primary members\n\n        Returns\n        -------\n        VersionedItem namedtuple which contains a .data and .metadata element\n        '
        try:
            read_preference = self._read_preference(allow_secondary)
            _version = self._read_metadata(symbol, as_of=as_of, read_preference=read_preference)
            return self._do_read(symbol, _version, from_version, date_range=date_range, read_preference=read_preference, **kwargs)
        except (OperationFailure, AutoReconnect) as e:
            log_exception('read', e, 1)
            _version = mongo_retry(self._read_metadata)(symbol, as_of=as_of, read_preference=ReadPreference.PRIMARY)
            return self._do_read_retry(symbol, _version, from_version, date_range=date_range, read_preference=ReadPreference.PRIMARY, **kwargs)
        except Exception as e:
            log_exception('read', e, 1)
            raise

    @mongo_retry
    def get_info(self, symbol, as_of=None):
        if False:
            return 10
        '\n        Reads and returns information about the data stored for symbol\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        as_of : `str` or int or `datetime.datetime`\n            Return the data as it was as_of the point in time.\n            `int` : specific version number\n            `str` : snapshot name which contains the version\n            `datetime.datetime` : the version of the data that existed as_of the requested point in time\n\n        Returns\n        -------\n        dictionary of the information (specific to the type of data)\n        '
        version = self._read_metadata(symbol, as_of=as_of, read_preference=None)
        handler = self._read_handler(version, symbol)
        if handler and hasattr(handler, 'get_info'):
            return handler.get_info(version)
        return {}

    @staticmethod
    def handler_supports_read_option(handler, option):
        if False:
            for i in range(10):
                print('nop')
        options_method = getattr(handler, 'read_options', None)
        if callable(options_method):
            return option in options_method()
        return True

    def get_arctic_version(self, symbol, as_of=None):
        if False:
            i = 10
            return i + 15
        '\n        Return the numerical representation of the arctic version used to write the last (or as_of) version for\n        the given symbol.\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        as_of : `str` or int or `datetime.datetime`\n            Return the data as it was as_of the point in time.\n            `int` : specific version number\n            `str` : snapshot name which contains the version\n            `datetime.datetime` : the version of the data that existed as_of the requested point in time\n\n        Returns\n        -------\n        arctic_version : int\n            The numerical representation of Arctic version, used to create the specified symbol version\n        '
        return self._read_metadata(symbol, as_of=as_of).get('arctic_version', 0)

    def _do_read(self, symbol, version, from_version=None, **kwargs):
        if False:
            return 10
        if version.get('deleted'):
            raise NoDataFoundException('No data found for %s in library %s' % (symbol, self._arctic_lib.get_name()))
        handler = self._read_handler(version, symbol)
        if self._with_strict_handler_match and kwargs.get('date_range') and (not self.handler_supports_read_option(handler, 'date_range')):
            raise ArcticException('Date range arguments not supported by handler in %s' % symbol)
        data = handler.read(self._arctic_lib, version, symbol, from_version=from_version, **kwargs)
        return VersionedItem(symbol=symbol, library=self._arctic_lib.get_name(), version=version['version'], metadata=version.pop('metadata', None), data=data, host=self._arctic_lib.arctic.mongo_host)
    _do_read_retry = mongo_retry(_do_read)

    @mongo_retry
    def read_metadata(self, symbol, as_of=None, allow_secondary=None):
        if False:
            i = 10
            return i + 15
        "\n        Return the metadata saved for a symbol.  This method is fast as it doesn't\n        actually load the data.\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        as_of : `str` or int or `datetime.datetime`\n            Return the data as it was as_of the point in time.\n            `int` : specific version number\n            `str` : snapshot name which contains the version\n            `datetime.datetime` : the version of the data that existed as_of the requested point in time\n        allow_secondary : `bool` or `None`\n            Override the default behavior for allowing reads from secondary members of a cluster:\n            `None` : use the settings from the top-level `Arctic` object used to query this version store.\n            `True` : allow reads from secondary members\n            `False` : only allow reads from primary members\n        "
        _version = self._read_metadata(symbol, as_of=as_of, read_preference=self._read_preference(allow_secondary))
        return VersionedItem(symbol=symbol, library=self._arctic_lib.get_name(), version=_version['version'], metadata=_version.pop('metadata', None), data=None, host=self._arctic_lib.arctic.mongo_host)

    def _read_metadata(self, symbol, as_of=None, read_preference=None):
        if False:
            for i in range(10):
                print('nop')
        if read_preference is None:
            read_preference = ReadPreference.PRIMARY_PREFERRED if not self._allow_secondary else ReadPreference.SECONDARY_PREFERRED
        versions_coll = self._versions.with_options(read_preference=read_preference)
        _version = None
        if as_of is None:
            _version = versions_coll.find_one({'symbol': symbol}, sort=[('version', pymongo.DESCENDING)])
        elif isinstance(as_of, str):
            snapshot = self._snapshots.find_one({'name': as_of})
            if snapshot:
                _version = versions_coll.find_one({'symbol': symbol, 'parent': snapshot['_id']})
        elif isinstance(as_of, dt):
            if not as_of.tzinfo:
                as_of = as_of.replace(tzinfo=mktz())
            _version = versions_coll.find_one({'symbol': symbol, '_id': {'$lt': bson.ObjectId.from_datetime(as_of + timedelta(seconds=1))}}, sort=[('symbol', pymongo.DESCENDING), ('version', pymongo.DESCENDING)])
        else:
            _version = versions_coll.find_one({'symbol': symbol, 'version': as_of})
        if not _version:
            raise NoDataFoundException('No data found for %s in library %s' % (symbol, self._arctic_lib.get_name()))
        metadata = _version.get('metadata', None)
        if metadata is not None and metadata.get('deleted', False) is True:
            raise NoDataFoundException('No data found for %s in library %s' % (symbol, self._arctic_lib.get_name()))
        return _version

    def _insert_version(self, version):
        if False:
            return 10
        try:
            mongo_retry(self._versions.insert_one)(version)
        except DuplicateKeyError as err:
            logger.exception(err)
            raise OperationFailure('A version with the same _id exists, force a clean retry')

    @mongo_retry
    def append(self, symbol, data, metadata=None, prune_previous_version=True, upsert=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Append 'data' under the specified 'symbol' name to this library.\n        The exact meaning of 'append' is left up to the underlying store implementation.\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        data :\n            to be persisted\n        metadata : `dict`\n            an optional dictionary of metadata to persist along with the symbol.\n        prune_previous_version : `bool`\n            Removes previous (non-snapshotted) versions from the database.\n            Default: True\n        upsert : `bool`\n            Write 'data' if no previous version exists.\n        "
        self._arctic_lib.check_quota()
        version = {'_id': bson.ObjectId()}
        version['arctic_version'] = ARCTIC_VERSION_NUMERICAL
        version['symbol'] = symbol
        spec = {'symbol': symbol}
        previous_version = self._versions.find_one(spec, sort=[('version', pymongo.DESCENDING)])
        if len(data) == 0 and previous_version is not None:
            return VersionedItem(symbol=symbol, library=self._arctic_lib.get_name(), version=previous_version['version'], metadata=version.pop('metadata', None), data=None, host=self._arctic_lib.arctic.mongo_host)
        if upsert and previous_version is None:
            return self.write(symbol=symbol, data=data, prune_previous_version=prune_previous_version, metadata=metadata)
        assert previous_version is not None
        dirty_append = False
        next_ver = self._version_nums.find_one_and_update({'symbol': symbol}, {'$inc': {'version': 1}}, upsert=False, new=True)['version']
        if next_ver != previous_version['version'] + 1:
            dirty_append = True
            logger.debug('version_nums is out of sync with previous version document.\n            This probably means that either a version document write has previously failed, or the previous version has been deleted.')
        previous_metadata = previous_version.get('metadata', None)
        if upsert and previous_metadata is not None and (previous_metadata.get('deleted', False) is True):
            return self.write(symbol=symbol, data=data, prune_previous_version=prune_previous_version, metadata=metadata)
        handler = self._read_handler(previous_version, symbol)
        if metadata is not None:
            version['metadata'] = metadata
        elif 'metadata' in previous_version:
            version['metadata'] = previous_version['metadata']
        if handler and hasattr(handler, 'append') and callable(handler.append):
            handler.append(self._arctic_lib, version, symbol, data, previous_version, dirty_append=dirty_append, **kwargs)
        else:
            raise Exception('Append not implemented for handler %s' % handler)
        if prune_previous_version and previous_version:
            self._prune_previous_versions(symbol, keep_version=version.get('base_version_id'), new_version_shas=version.get(FW_POINTERS_REFS_KEY), keep_mins=kwargs.get('keep_mins', 120))
        version['version'] = next_ver
        self._insert_version(version)
        return VersionedItem(symbol=symbol, library=self._arctic_lib.get_name(), version=version['version'], metadata=version.pop('metadata', None), data=None, host=self._arctic_lib.arctic.mongo_host)

    @mongo_retry
    def write(self, symbol, data, metadata=None, prune_previous_version=True, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Write 'data' under the specified 'symbol' name to this library.\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        data :\n            to be persisted\n        metadata : `dict`\n            an optional dictionary of metadata to persist along with the symbol.\n            Default: None\n        prune_previous_version : `bool`\n            Removes previous (non-snapshotted) versions from the database.\n            Default: True\n        kwargs :\n            passed through to the write handler\n\n        Returns\n        -------\n        VersionedItem named tuple containing the metadata and version number\n        of the written symbol in the store.\n        "
        self._arctic_lib.check_quota()
        version = {'_id': bson.ObjectId()}
        version['arctic_version'] = ARCTIC_VERSION_NUMERICAL
        version['symbol'] = symbol
        version['version'] = self._version_nums.find_one_and_update({'symbol': symbol}, {'$inc': {'version': 1}}, upsert=True, new=True)['version']
        version['metadata'] = metadata
        previous_version = self._versions.find_one({'symbol': symbol, 'version': {'$lt': version['version']}}, sort=[('version', pymongo.DESCENDING)])
        handler = self._write_handler(version, symbol, data, **kwargs)
        handler.write(self._arctic_lib, version, symbol, data, previous_version, **kwargs)
        if prune_previous_version and previous_version:
            self._prune_previous_versions(symbol, keep_mins=kwargs.get('keep_mins', 120), new_version_shas=version.get(FW_POINTERS_REFS_KEY))
        self._insert_version(version)
        logger.debug('Finished writing versions for %s', symbol)
        return VersionedItem(symbol=symbol, library=self._arctic_lib.get_name(), version=version['version'], metadata=version.pop('metadata', None), data=None, host=self._arctic_lib.arctic.mongo_host)

    def _add_new_version_using_reference(self, symbol, new_version, reference_version, prune_previous_version):
        if False:
            i = 10
            return i + 15
        constraints = new_version and reference_version and (new_version['symbol'] == reference_version['symbol']) and (new_version['_id'] != reference_version['_id']) and new_version['base_version_id']
        assert constraints
        lastv_seqn = self._last_version_seqnum(symbol)
        if lastv_seqn != new_version['version']:
            raise OperationFailure('The symbol {} has been modified concurrently ({} != {})'.format(symbol, lastv_seqn, new_version['version']))
        self._insert_version(new_version)
        last_look = self._versions.find_one({'_id': reference_version['_id']})
        if last_look is None or last_look.get('deleted'):
            mongo_retry(self._versions.delete_one)({'_id': new_version['_id']})
            raise OperationFailure('Failed to write metadata for symbol %s. The previous version (%s, %d) has been removed during the update' % (symbol, str(reference_version['_id']), reference_version['version']))
        if prune_previous_version and reference_version:
            self._prune_previous_versions(symbol, new_version_shas=new_version.get(FW_POINTERS_REFS_KEY))
        logger.debug('Finished updating versions with new metadata for %s', symbol)
        return VersionedItem(symbol=symbol, library=self._arctic_lib.get_name(), version=new_version['version'], metadata=new_version.get('metadata'), data=None, host=self._arctic_lib.arctic.mongo_host)

    @mongo_retry
    def write_metadata(self, symbol, metadata, prune_previous_version=True, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Write 'metadata' under the specified 'symbol' name to this library.\n        The data will remain unchanged. A new version will be created.\n        If the symbol is missing, it causes a write with empty data (None, pickled, can't append)\n        and the supplied metadata.\n        Returns a VersionedItem object only with a metadata element.\n        Fast operation: Zero data/segment read/write operations.\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        metadata : `dict` or `None`\n            dictionary of metadata to persist along with the symbol\n        prune_previous_version : `bool`\n            Removes previous (non-snapshotted) versions from the database.\n            Default: True\n        kwargs :\n            passed through to the write handler (only used if symbol does not already exist or is deleted)\n\n        Returns\n        -------\n        `VersionedItem`\n            VersionedItem named tuple containing the metadata of the written symbol's version document in the store.\n        "
        try:
            previous_version = self._read_metadata(symbol)
        except NoDataFoundException:
            return self.write(symbol, data=None, metadata=metadata, prune_previous_version=prune_previous_version, **kwargs)
        new_version_num = self._version_nums.find_one_and_update({'symbol': symbol}, {'$inc': {'version': 1}}, upsert=True, new=True)['version']
        version = {k: previous_version[k] for k in previous_version.keys() if k != 'parent'}
        version['_id'] = bson.ObjectId()
        version['version'] = new_version_num
        version['metadata'] = metadata
        version['base_version_id'] = previous_version.get('base_version_id', previous_version['_id'])
        return self._add_new_version_using_reference(symbol, version, previous_version, prune_previous_version)

    @mongo_retry
    def restore_version(self, symbol, as_of, prune_previous_version=True):
        if False:
            while True:
                i = 10
        "\n        Restore the specified 'symbol' data and metadata to the state of a given version/snapshot/date.\n        Returns a VersionedItem object only with a metadata element.\n        Fast operation: Zero data/segment read/write operations.\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        as_of : `str` or `int` or `datetime.datetime`\n            Return the data as it was as_of the point in time.\n            `int` : specific version number\n            `str` : snapshot name which contains the version\n            `datetime.datetime` : the version of the data that existed as_of the requested point in time\n        prune_previous_version : `bool`\n            Removes previous (non-snapshotted) versions from the database.\n            Default: True\n\n        Returns\n        -------\n        `VersionedItem`\n            VersionedItem named tuple containing the metadata of the written symbol's version document in the store.\n        "
        version_to_restore = self._read_metadata(symbol, as_of=as_of)
        if self._last_version_seqnum(symbol) == version_to_restore['version']:
            return VersionedItem(symbol=symbol, library=self._arctic_lib.get_name(), version=version_to_restore['version'], host=self._arctic_lib.arctic.mongo_host, metadata=version_to_restore.pop('metadata', None), data=None)
        item = self.read(symbol, as_of=as_of)
        new_item = self.write(symbol, data=item.data, metadata=item.metadata, prune_previous_version=prune_previous_version)
        return new_item

    @mongo_retry
    def _find_prunable_version_ids(self, symbol, keep_mins):
        if False:
            while True:
                i = 10
        "\n        Find all non-snapshotted versions of a symbol that are older than a version that's at least keep_mins\n        minutes old.\n\n        Based on documents available on the secondary.\n        "
        read_preference = ReadPreference.SECONDARY_PREFERRED if keep_mins > 0 else ReadPreference.PRIMARY
        versions = self._versions.with_options(read_preference=read_preference)
        query = {'symbol': symbol, '$or': [{'parent': {'$exists': False}}, {'parent': []}], '_id': {'$lt': bson.ObjectId.from_datetime(dt.utcnow() + timedelta(seconds=1) - timedelta(minutes=keep_mins))}}
        cursor = versions.find(query, sort=[('version', pymongo.DESCENDING)], skip=1, projection={'_id': 1, FW_POINTERS_REFS_KEY: 1, FW_POINTERS_CONFIG_KEY: 1})
        return {v['_id']: ([bson.binary.Binary(x) for x in v.get(FW_POINTERS_REFS_KEY, [])], get_fwptr_config(v)) for v in cursor}

    @mongo_retry
    def _find_base_version_ids(self, symbol, version_ids):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all base_version_ids for a symbol that are not bases of version_ids\n        '
        cursor = self._versions.find({'symbol': symbol, '_id': {'$nin': version_ids}, 'base_version_id': {'$exists': True}}, projection={'base_version_id': 1})
        return [version['base_version_id'] for version in cursor]

    def _prune_previous_versions(self, symbol, keep_mins=120, keep_version=None, new_version_shas=None):
        if False:
            print('Hello World!')
        '\n        Prune versions, not pointed at by snapshots which are at least keep_mins old. Prune will never\n        remove all versions.\n        '
        new_version_shas = new_version_shas if new_version_shas else []
        prunable_ids_to_shas = self._find_prunable_version_ids(symbol, keep_mins)
        prunable_ids = list(prunable_ids_to_shas.keys())
        if keep_version is not None:
            try:
                prunable_ids.remove(keep_version)
            except ValueError:
                pass
        if not prunable_ids:
            return
        base_version_ids = self._find_base_version_ids(symbol, prunable_ids)
        version_ids = list(set(prunable_ids) - set(base_version_ids))
        if not version_ids:
            return
        mongo_retry(self._versions.delete_many)({'_id': {'$in': version_ids}})
        prunable_ids_to_shas = {k: prunable_ids_to_shas[k] for k in version_ids}
        shas_to_delete = [sha for v in prunable_ids_to_shas.values() for sha in v[0] if sha not in new_version_shas]
        mongo_retry(cleanup)(self._arctic_lib, symbol, version_ids, self._versions, shas_to_delete=shas_to_delete, pointers_cfgs=[v[1] for v in prunable_ids_to_shas.values()])

    @mongo_retry
    def _delete_version(self, symbol, version_num, do_cleanup=True):
        if False:
            i = 10
            return i + 15
        "\n        Delete the n'th version of this symbol from the historical collection.\n        "
        version = self._versions.find_one({'symbol': symbol, 'version': version_num})
        if not version:
            logger.error("Can't delete %s:%s as not found in DB" % (symbol, version_num))
            return
        if version.get('parent', None):
            for parent in version['parent']:
                snap_name = self._snapshots.find_one({'_id': parent})
                if snap_name:
                    snap_name = snap_name['name']
                logger.error("Can't delete: %s:%s as pointed to by snapshot: %s" % (symbol, version['version'], snap_name))
                return
        self._versions.delete_one({'_id': version['_id']})
        if do_cleanup:
            cleanup(self._arctic_lib, symbol, [version['_id']], self._versions, shas_to_delete=tuple((bson.binary.Binary(s) for s in version.get(FW_POINTERS_REFS_KEY, []))), pointers_cfgs=(get_fwptr_config(version),))

    @mongo_retry
    def delete(self, symbol):
        if False:
            return 10
        "\n        Delete all versions of the item from the current library which aren't\n        currently part of some snapshot.\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name to delete\n        "
        logger.info('Deleting data item: %r from %r' % (symbol, self._arctic_lib.get_name()))
        sentinel = self.write(symbol, None, prune_previous_version=False, metadata={'deleted': True})
        self._prune_previous_versions(symbol, 0)
        snapped_version = self._versions.find_one({'symbol': symbol, 'metadata.deleted': {'$ne': True}})
        if not snapped_version:
            self._delete_version(symbol, sentinel.version)
        assert not self.has_symbol(symbol)

    def _write_audit(self, user, message, changed_version):
        if False:
            i = 10
            return i + 15
        '\n        Creates an audit entry, which is much like a snapshot in that\n        it references versions and provides some history of the changes made.\n        '
        audit = {'_id': bson.ObjectId(), 'user': user, 'message': message, 'symbol': changed_version.symbol}
        orig_version = changed_version.orig_version.version
        new_version = changed_version.new_version.version
        audit['orig_v'] = orig_version
        audit['new_v'] = new_version
        mongo_retry(self._versions.update_many)({'symbol': changed_version.symbol, 'version': {'$in': [orig_version, new_version]}}, {'$addToSet': {'parent': audit['_id']}})
        mongo_retry(self._audit.insert_one)(audit)

    @mongo_retry
    def snapshot(self, snap_name, metadata=None, skip_symbols=None, versions=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Snapshot versions of symbols in the library.  Can be used like:\n\n        Parameters\n        ----------\n        snap_name : `str`\n            name of the snapshot\n        metadata : `dict`\n            an optional dictionary of metadata to persist along with the symbol.\n        skip_symbols : `collections.Iterable`\n            optional symbols to be excluded from the snapshot\n        versions: `dict`\n            an optional dictionary of versions of the symbols to be snapshot\n        '
        snapshot = self._snapshots.find_one({'name': snap_name})
        if snapshot:
            raise DuplicateSnapshotException("Snapshot '%s' already exists." % snap_name)
        snapshot = {'_id': bson.ObjectId()}
        snapshot['name'] = snap_name
        snapshot['metadata'] = metadata
        skip_symbols = set() if skip_symbols is None else set(skip_symbols)
        if versions is None:
            versions = {sym: None for sym in set(self.list_symbols()) - skip_symbols}
        for sym in versions:
            try:
                sym = self._read_metadata(sym, read_preference=ReadPreference.PRIMARY, as_of=versions[sym])
                mongo_retry(self._versions.update_one)({'_id': sym['_id']}, {'$addToSet': {'parent': snapshot['_id']}})
            except NoDataFoundException:
                pass
        mongo_retry(self._snapshots.insert_one)(snapshot)

    @mongo_retry
    def delete_snapshot(self, snap_name):
        if False:
            while True:
                i = 10
        '\n        Delete a named snapshot\n\n        Parameters\n        ----------\n        symbol : `str`\n            The snapshot name to delete\n        '
        snapshot = self._snapshots.find_one({'name': snap_name})
        if not snapshot:
            raise NoDataFoundException('Snapshot %s not found!' % snap_name)
        self._versions.update_many({'parent': snapshot['_id']}, {'$pull': {'parent': snapshot['_id']}})
        self._snapshots.delete_one({'name': snap_name})

    @mongo_retry
    def list_snapshots(self):
        if False:
            print('Hello World!')
        '\n        List the snapshots in the library\n\n        Returns\n        -------\n        string list of snapshot names\n        '
        return dict(((i['name'], i['metadata']) for i in self._snapshots.find()))

    @mongo_retry
    def stats(self):
        if False:
            i = 10
            return i + 15
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
        res['versions'] = db.command('collstats', self._versions.name)
        res['snapshots'] = db.command('collstats', self._snapshots.name)
        res['totals'] = {'count': res['chunks']['count'], 'size': res['chunks']['size'] + res['versions']['size'] + res['snapshots']['size']}
        return res

    def _fsck(self, dry_run):
        if False:
            while True:
                i = 10
        '\n        Run a consistency check on this VersionStore library.\n        '
        self._cleanup_orphaned_chunks(dry_run)
        self._cleanup_unreachable_shas(dry_run)
        self._cleanup_orphaned_versions(dry_run)

    def _cleanup_unreachable_shas(self, dry_run):
        if False:
            for i in range(10):
                print('nop')
        lib = self
        chunks_coll = lib._collection
        versions_coll = chunks_coll.versions
        for symbol in chunks_coll.distinct('symbol'):
            logger.debug('Checking %s (forward pointers)' % symbol)
            all_symbol_pointers_cfgs = _get_symbol_pointer_cfgs(symbol, versions_coll)
            if FwPointersCfg.DISABLED not in all_symbol_pointers_cfgs:
                symbol_alive_shas = get_symbol_alive_shas(symbol, versions_coll)
                all_symbol_shas = set(chunks_coll.distinct('sha', {'symbol': symbol}))
                unreachable_shas = all_symbol_shas - symbol_alive_shas
                logger.info('Cleaning up {} SHAs for symbol {}'.format(len(unreachable_shas), symbol))
                if not dry_run:
                    id_time_constraint = {'$lt': bson.ObjectId.from_datetime(dt.now() - timedelta(days=1))}
                    chunks_coll.delete_many({'_id': id_time_constraint, 'symbol': symbol, 'parent': id_time_constraint, 'sha': {'$in': list(unreachable_shas)}})

    def _cleanup_orphaned_chunks(self, dry_run):
        if False:
            i = 10
            return i + 15
        '\n        Fixes any chunks who have parent pointers to missing versions.\n        Removes the broken parent pointer and, if there are no other parent pointers for the chunk,\n        removes the chunk.\n        '
        lib = self
        chunks_coll = lib._collection
        versions_coll = chunks_coll.versions
        logger.info('ORPHANED CHUNK CHECK: %s' % self._arctic_lib.get_name())
        for symbol in chunks_coll.distinct('symbol'):
            logger.debug('Checking %s' % symbol)
            gen_time = dt.now() - timedelta(days=1)
            parent_id_constraint = {'$lt': bson.ObjectId.from_datetime(gen_time)}
            versions = set(versions_coll.find({'symbol': symbol, '_id': parent_id_constraint}).distinct('_id'))
            parents = chunks_coll.aggregate([{'$match': {'symbol': symbol}}, {'$project': {'parent': True}}, {'$unwind': '$parent'}, {'$match': {'parent': parent_id_constraint}}, {'$group': {'_id': '$parent'}}])
            parent_ids = set([x['_id'] for x in parents])
            leaked_versions = sorted(parent_ids - versions)
            if len(leaked_versions):
                logger.info('%s leaked %d versions' % (symbol, len(leaked_versions)))
            for x in leaked_versions:
                chunk_count = mongo_count(chunks_coll, filter={'symbol': symbol, 'parent': x})
                logger.info("%s: Missing Version %s (%s) ; %s chunks ref'd" % (symbol, x.generation_time, x, chunk_count))
                if versions_coll.find_one({'symbol': symbol, '_id': x}) is not None:
                    raise Exception("Error: version (%s) is found for (%s), but shouldn't be!" % (x, symbol))
            if not dry_run:
                cleanup(lib._arctic_lib, symbol, leaked_versions, versions_coll)

    def _cleanup_orphaned_versions(self, dry_run):
        if False:
            while True:
                i = 10
        "\n        Fixes any versions who have parent pointers to missing snapshots.\n        Note, doesn't delete the versions, just removes the parent pointer if it no longer\n        exists in snapshots.\n        "
        lib = self
        versions_coll = lib._collection.versions
        snapshots_coll = lib._collection.snapshots
        logger.info('ORPHANED SNAPSHOT CHECK: %s' % self._arctic_lib.get_name())
        gen_time = dt.now() - timedelta(days=1)
        parent_id_constraint = {'$lt': bson.ObjectId.from_datetime(gen_time)}
        snapshots = set(snapshots_coll.distinct('_id'))
        snapshots |= set(lib._audit.distinct('_id'))
        parents = versions_coll.aggregate([{'$project': {'parent': True}}, {'$unwind': '$parent'}, {'$match': {'parent': parent_id_constraint}}, {'$group': {'_id': '$parent'}}])
        parent_ids = set([x['_id'] for x in parents])
        leaked_snaps = sorted(parent_ids - snapshots)
        if len(leaked_snaps):
            logger.info('leaked %d snapshots' % len(leaked_snaps))
        for x in leaked_snaps:
            ver_count = mongo_count(versions_coll, filter={'parent': x})
            logger.info("Missing Snapshot %s (%s) ; %s versions ref'd" % (x.generation_time, x, ver_count))
            if snapshots_coll.find_one({'_id': x}) is not None:
                raise Exception("Error: snapshot (%s) is found, but shouldn't be!" % x)
            if not dry_run:
                versions_coll.update_many({'parent': x}, {'$pull': {'parent': x}})