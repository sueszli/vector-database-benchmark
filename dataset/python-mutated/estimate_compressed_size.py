"""Code to estimate the entropy of content"""
from __future__ import absolute_import
import zlib

class ZLibEstimator(object):
    """Uses zlib.compressobj to estimate compressed size."""

    def __init__(self, target_size, min_compression=2.0):
        if False:
            i = 10
            return i + 15
        "Create a new estimator.\n\n        :param target_size: The desired size of the compressed content.\n        :param min_compression: Estimated minimum compression. By default we\n            assume that the content is 'text', which means a min compression of\n            about 2:1.\n        "
        self._target_size = target_size
        self._compressor = zlib.compressobj()
        self._uncompressed_size_added = 0
        self._compressed_size_added = 0
        self._unflushed_size_added = 0
        self._estimated_compression = 2.0

    def add_content(self, content):
        if False:
            print('Hello World!')
        self._uncompressed_size_added += len(content)
        self._unflushed_size_added += len(content)
        z_size = len(self._compressor.compress(content))
        if z_size > 0:
            self._record_z_len(z_size)

    def _record_z_len(self, count):
        if False:
            while True:
                i = 10
        self._compressed_size_added += count
        self._unflushed_size_added = 0
        self._estimated_compression = float(self._uncompressed_size_added) / self._compressed_size_added

    def full(self):
        if False:
            print('Hello World!')
        'Have we reached the target size?'
        if self._unflushed_size_added:
            remaining_size = self._target_size - self._compressed_size_added
            est_z_size = self._unflushed_size_added / self._estimated_compression
            if est_z_size >= remaining_size:
                z_size = len(self._compressor.flush(zlib.Z_SYNC_FLUSH))
                self._record_z_len(z_size)
        return self._compressed_size_added >= self._target_size