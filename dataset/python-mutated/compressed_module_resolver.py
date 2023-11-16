"""Functions to resolve TF-Hub Module stored in compressed TGZ format."""
import hashlib
import logging
import urllib
import tensorflow as tf
from tensorflow_hub import resolver
LOCK_FILE_TIMEOUT_SEC = 10 * 60
_HUB_TF_GOOGLE_CN = 'https://hub.tensorflow.google.cn/'
_GCS_GOOGLE_CN_TEMPLATE = 'https://gcs.tensorflow.google.cn/tfhub-modules/%s.tar.gz'
_COMPRESSED_FORMAT_QUERY = ('tf-hub-format', 'compressed')

def _module_dir(handle):
    if False:
        i = 10
        return i + 15
    'Returns the directory where to cache the module.'
    cache_dir = resolver.tfhub_cache_dir(use_temp=True)
    return resolver.create_local_module_dir(cache_dir, hashlib.sha1(handle.encode('utf8')).hexdigest())

def _is_tarfile(filename):
    if False:
        return 10
    "Returns true if 'filename' is TAR file."
    return filename.endswith(('.tar', '.tar.gz', '.tgz'))

class HttpCompressedFileResolver(resolver.HttpResolverBase):
    """Resolves HTTP handles by downloading and decompressing them to local fs."""

    def is_supported(self, handle):
        if False:
            while True:
                i = 10
        if not self.is_http_protocol(handle):
            return False
        load_format = resolver.model_load_format()
        return load_format in [resolver.ModelLoadFormat.COMPRESSED.value, resolver.ModelLoadFormat.AUTO.value]

    def __call__(self, handle):
        if False:
            for i in range(10):
                print('nop')
        module_dir = _module_dir(handle)

        def download(handle, tmp_dir):
            if False:
                return 10
            'Fetch a module via HTTP(S), handling redirect and download headers.'
            if handle.startswith(_HUB_TF_GOOGLE_CN):
                full_model_name = handle[len(_HUB_TF_GOOGLE_CN):]
                gcs_cn_url = _GCS_GOOGLE_CN_TEMPLATE % full_model_name
                logging.info('Directly downloading %s', gcs_cn_url)
                response = self._call_urlopen(gcs_cn_url)
                return resolver.DownloadManager(handle).download_and_uncompress(response, tmp_dir)
            request = urllib.request.Request(self._append_compressed_format_query(handle))
            response = self._call_urlopen(request)
            return resolver.DownloadManager(handle).download_and_uncompress(response, tmp_dir)
        return resolver.atomic_download(handle, download, module_dir, self._lock_file_timeout_sec())

    def _lock_file_timeout_sec(self):
        if False:
            for i in range(10):
                print('nop')
        return LOCK_FILE_TIMEOUT_SEC

    def _append_compressed_format_query(self, handle):
        if False:
            i = 10
            return i + 15
        return self._append_format_query(handle, _COMPRESSED_FORMAT_QUERY)

class GcsCompressedFileResolver(resolver.Resolver):
    """Resolves GCS handles by downloading and decompressing them to local fs."""

    def is_supported(self, handle):
        if False:
            i = 10
            return i + 15
        return handle.startswith('gs://') and _is_tarfile(handle)

    def __call__(self, handle):
        if False:
            i = 10
            return i + 15
        module_dir = _module_dir(handle)

        def download(handle, tmp_dir):
            if False:
                return 10
            return resolver.DownloadManager(handle).download_and_uncompress(tf.compat.v1.gfile.GFile(handle, 'rb'), tmp_dir)
        return resolver.atomic_download(handle, download, module_dir, LOCK_FILE_TIMEOUT_SEC)