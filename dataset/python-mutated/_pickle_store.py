import io
import logging
from operator import itemgetter
import bson
from bson.binary import Binary
from bson.errors import InvalidDocument
import pickle
from ._version_store_utils import checksum, pickle_compat_load, version_base_or_id
from .._compression import decompress, compress_array
from ..exceptions import UnsupportedPickleStoreVersion
from .._config import SKIP_BSON_ENCODE_PICKLE_STORE, MAX_BSON_ENCODE
_MAGIC_CHUNKED = '__chunked__'
_MAGIC_CHUNKEDV2 = '__chunked__V2'
_CHUNK_SIZE = 15 * 1024 * 1024
_HARD_MAX_BSON_ENCODE = 10 * 1024 * 1024
logger = logging.getLogger(__name__)

class PickleStore(object):

    @classmethod
    def initialize_library(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_info(self, _version):
        if False:
            i = 10
            return i + 15
        return {'type': 'blob', 'handler': self.__class__.__name__}

    def read(self, mongoose_lib, version, symbol, **kwargs):
        if False:
            i = 10
            return i + 15
        blob = version.get('blob')
        if blob is not None:
            if blob == _MAGIC_CHUNKEDV2:
                collection = mongoose_lib.get_top_level_collection()
                data = b''.join((decompress(x['data']) for x in sorted(collection.find({'symbol': symbol, 'parent': version_base_or_id(version)}), key=itemgetter('segment'))))
            elif blob == _MAGIC_CHUNKED:
                collection = mongoose_lib.get_top_level_collection()
                data = b''.join((x['data'] for x in sorted(collection.find({'symbol': symbol, 'parent': version_base_or_id(version)}), key=itemgetter('segment'))))
                data = decompress(data)
            else:
                if blob[:len(_MAGIC_CHUNKED)] == _MAGIC_CHUNKED:
                    logger.error('Data was written by unsupported version of pickle store for symbol %s. Upgrade Arctic and try again' % symbol)
                    raise UnsupportedPickleStoreVersion('Data was written by unsupported version of pickle store')
                try:
                    data = decompress(blob)
                except:
                    logger.error('Failed to read symbol %s' % symbol)
            try:
                return pickle_compat_load(io.BytesIO(data))
            except UnicodeDecodeError as ue:
                logger.info('Could not Unpickle with ascii, Using latin1.')
                encoding = kwargs.get('encoding', 'latin_1')
                return pickle_compat_load(io.BytesIO(data), encoding=encoding)
        return version['data']

    @staticmethod
    def read_options():
        if False:
            i = 10
            return i + 15
        return []

    def write(self, arctic_lib, version, symbol, item, _previous_version):
        if False:
            print('Hello World!')
        if not SKIP_BSON_ENCODE_PICKLE_STORE:
            try:
                b = bson.BSON.encode({'data': item})
                if len(b) < min(MAX_BSON_ENCODE, _HARD_MAX_BSON_ENCODE):
                    version['data'] = item
                    return
            except InvalidDocument:
                pass
        collection = arctic_lib.get_top_level_collection()
        version['blob'] = _MAGIC_CHUNKEDV2
        pickle_protocol = min(pickle.HIGHEST_PROTOCOL, 4)
        pickled = pickle.dumps(item, protocol=pickle_protocol)
        data = compress_array([pickled[i * _CHUNK_SIZE:(i + 1) * _CHUNK_SIZE] for i in range(int(len(pickled) / _CHUNK_SIZE + 1))])
        for (seg, d) in enumerate(data):
            segment = {'data': Binary(d)}
            segment['segment'] = seg
            seg += 1
            sha = checksum(symbol, segment)
            collection.update_one({'symbol': symbol, 'sha': sha}, {'$set': segment, '$addToSet': {'parent': version['_id']}}, upsert=True)