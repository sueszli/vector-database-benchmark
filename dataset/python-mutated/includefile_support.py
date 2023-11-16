import io
import os
import shutil
import uuid
from tempfile import mkdtemp
from metaflow.exception import MetaflowException, MetaflowInternalError

class Azure(object):

    @classmethod
    def get_root_from_config(cls, echo, create_on_absent=True):
        if False:
            for i in range(10):
                print('nop')
        from metaflow.metaflow_config import DATATOOLS_AZUREROOT
        return DATATOOLS_AZUREROOT

    def __init__(self):
        if False:
            while True:
                i = 10
        self._tmpdir = None

    def _get_storage_backend(self, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an AzureDatastore, rooted at the container level, no prefix.\n        Key MUST be a fully qualified path. e.g. <container_name>/b/l/o/b/n/a/m/e\n        '
        from metaflow.plugins.azure.azure_utils import parse_azure_full_path
        (container_name, _) = parse_azure_full_path(key)
        from metaflow.plugins import DATASTORES
        storage_impl = [d for d in DATASTORES if d.TYPE == 'azure'][0]
        return storage_impl(container_name)

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, *args):
        if False:
            return 10
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir)

    def get(self, key=None, return_missing=False):
        if False:
            print('Hello World!')
        'Key MUST be a fully qualified path with uri scheme.  azure://<container_name>/b/l/o/b/n/a/m/e'
        if not return_missing:
            raise MetaflowException('Azure object supports only return_missing=True')
        if not key.startswith('azure://'):
            raise MetaflowInternalError(msg="Expected Azure object key to start with 'azure://'")
        uri_style_key = key
        short_key = key[8:]
        storage = self._get_storage_backend(short_key)
        azure_object = None
        with storage.load_bytes([short_key]) as load_result:
            for (_, tmpfile, _) in load_result:
                if tmpfile is None:
                    azure_object = AzureObject(uri_style_key, None, False, None)
                else:
                    if not self._tmpdir:
                        self._tmpdir = mkdtemp(prefix='metaflow.includefile.azure.')
                    output_file_path = os.path.join(self._tmpdir, str(uuid.uuid4()))
                    shutil.move(tmpfile, output_file_path)
                    sz = os.stat(output_file_path).st_size
                    azure_object = AzureObject(uri_style_key, output_file_path, True, sz)
                break
        return azure_object

    def put(self, key, obj, overwrite=True):
        if False:
            i = 10
            return i + 15
        'Key MUST be a fully qualified path.  <container_name>/b/l/o/b/n/a/m/e'
        storage = self._get_storage_backend(key)
        storage.save_bytes([(key, io.BytesIO(obj))], overwrite=overwrite)
        return 'azure://%s' % key

    def info(self, key=None, return_missing=False):
        if False:
            print('Hello World!')
        if not key.startswith('azure://'):
            raise MetaflowInternalError(msg="Expected Azure object key to start with 'azure://'")
        uri_style_key = key
        short_key = key[8:]
        storage = self._get_storage_backend(short_key)
        blob_size = storage.size_file(short_key)
        blob_exists = blob_size is not None
        if not blob_exists and (not return_missing):
            raise MetaflowException("Azure blob '%s' not found" % uri_style_key)
        return AzureObject(uri_style_key, None, blob_exists, blob_size)

class AzureObject(object):

    def __init__(self, url, path, exists, size):
        if False:
            while True:
                i = 10
        self._path = path
        self._url = url
        self._exists = exists
        self._size = size

    @property
    def path(self):
        if False:
            i = 10
            return i + 15
        return self._path

    @property
    def url(self):
        if False:
            return 10
        return self._url

    @property
    def exists(self):
        if False:
            print('Hello World!')
        return self._exists

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return self._size