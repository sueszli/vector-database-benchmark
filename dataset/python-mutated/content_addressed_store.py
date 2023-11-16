import gzip
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from ..exception import MetaflowInternalError
from .exceptions import DataException

class ContentAddressedStore(object):
    """
    This class is not meant to be overridden and is meant to be common across
    different datastores.
    """
    save_blobs_result = namedtuple('save_blobs_result', 'uri key')

    def __init__(self, prefix, storage_impl):
        if False:
            while True:
                i = 10
        '\n        Initialize a ContentAddressedStore\n\n        A content-addressed store stores data using a name/key that is a hash\n        of the content. This means that duplicate content is only stored once.\n\n        Parameters\n        ----------\n        prefix : string\n            Prefix that will be prepended when storing a file\n        storage_impl : type\n            Implementation for the backing storage implementation to use\n        '
        self._prefix = prefix
        self._storage_impl = storage_impl
        self.TYPE = self._storage_impl.TYPE
        self._blob_cache = None

    def set_blob_cache(self, blob_cache):
        if False:
            i = 10
            return i + 15
        self._blob_cache = blob_cache

    def save_blobs(self, blob_iter, raw=False, len_hint=0):
        if False:
            for i in range(10):
                print('nop')
        "\n        Saves blobs of data to the datastore\n\n        The blobs of data are saved as is if raw is True. If raw is False, the\n        datastore may process the blobs and they should then only be loaded\n        using load_blob\n\n        NOTE: The idea here is that there are two modes to access the file once\n        it is saved to the datastore:\n          - if raw is True, you would be able to access it directly using the\n            URI returned; the bytes that are passed in as 'blob' would be\n            returned directly by reading the object at that URI. You would also\n            be able to access it using load_blob passing the key returned\n          - if raw is False, no URI would be returned (the URI would be None)\n            and you would only be able to access the object using load_blob.\n          - The API also specifically takes a list to allow for parallel writes\n            if available in the datastore. We could also make a single\n            save_blob' API and save_blobs but this seems superfluous\n\n        Parameters\n        ----------\n        blob_iter : Iterator over bytes objects to save\n        raw : bool, optional\n            Whether to save the bytes directly or process them, by default False\n        len_hint : Hint of the number of blobs that will be produced by the\n            iterator, by default 0\n\n        Returns\n        -------\n        List of save_blobs_result:\n            The list order is the same as the blobs passed in. The URI will be\n            None if raw is False.\n        "
        results = []

        def packing_iter():
            if False:
                while True:
                    i = 10
            for blob in blob_iter:
                sha = sha1(blob).hexdigest()
                path = self._storage_impl.path_join(self._prefix, sha[:2], sha)
                results.append(self.save_blobs_result(uri=self._storage_impl.full_uri(path) if raw else None, key=sha))
                if not self._storage_impl.is_file([path])[0]:
                    meta = {'cas_raw': raw, 'cas_version': 1}
                    if raw:
                        yield (path, (BytesIO(blob), meta))
                    else:
                        yield (path, (self._pack_v1(blob), meta))
        self._storage_impl.save_bytes(packing_iter(), overwrite=True, len_hint=len_hint)
        return results

    def load_blobs(self, keys, force_raw=False):
        if False:
            print('Hello World!')
        '\n        Mirror function of save_blobs\n\n        This function is guaranteed to return the bytes passed to save_blob for\n        the keys\n\n        Parameters\n        ----------\n        keys : List of string\n            Key describing the object to load\n        force_raw : bool, optional\n            Support for backward compatibility with previous datastores. If\n            True, this will force the key to be loaded as is (raw). By default,\n            False\n\n        Returns\n        -------\n        Returns an iterator of (string, bytes) tuples; the iterator may return keys\n        in a different order than were passed in.\n        '
        load_paths = []
        for key in keys:
            blob = None
            if self._blob_cache:
                blob = self._blob_cache.load_key(key)
            if blob is not None:
                yield (key, blob)
            else:
                path = self._storage_impl.path_join(self._prefix, key[:2], key)
                load_paths.append((key, path))
        with self._storage_impl.load_bytes([p for (_, p) in load_paths]) as loaded:
            for (path_key, file_path, meta) in loaded:
                key = self._storage_impl.path_split(path_key)[-1]
                with open(file_path, 'rb') as f:
                    if force_raw or (meta and meta.get('cas_raw', False)):
                        blob = f.read()
                    else:
                        if meta is None:
                            unpack_code = self._unpack_backward_compatible
                        else:
                            version = meta.get('cas_version', -1)
                            if version == -1:
                                raise DataException("Could not extract encoding version for '%s'" % path)
                            unpack_code = getattr(self, '_unpack_v%d' % version, None)
                            if unpack_code is None:
                                raise DataException("Unknown encoding version %d for '%s' -- the artifact is either corrupt or you need to update Metaflow to the latest version" % (version, path))
                        try:
                            blob = unpack_code(f)
                        except Exception as e:
                            raise DataException("Could not unpack artifact '%s': %s" % (path, e))
                if self._blob_cache:
                    self._blob_cache.store_key(key, blob)
                yield (key, blob)

    def _unpack_backward_compatible(self, blob):
        if False:
            for i in range(10):
                print('nop')
        return self._unpack_v1(blob)

    def _pack_v1(self, blob):
        if False:
            i = 10
            return i + 15
        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=3) as f:
            f.write(blob)
        buf.seek(0)
        return buf

    def _unpack_v1(self, blob):
        if False:
            i = 10
            return i + 15
        with gzip.GzipFile(fileobj=blob, mode='rb') as f:
            return f.read()

class BlobCache(object):

    def load_key(self, key):
        if False:
            i = 10
            return i + 15
        pass

    def store_key(self, key, blob):
        if False:
            i = 10
            return i + 15
        pass