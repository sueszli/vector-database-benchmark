"""ChunkWriter: write compressed data out with a fixed upper bound."""
from __future__ import absolute_import
import zlib
from zlib import Z_FINISH, Z_SYNC_FLUSH

class ChunkWriter(object):
    """ChunkWriter allows writing of compressed data with a fixed size.

    If less data is supplied than fills a chunk, the chunk is padded with
    NULL bytes. If more data is supplied, then the writer packs as much
    in as it can, but never splits any item it was given.

    The algorithm for packing is open to improvement! Current it is:
     - write the bytes given
     - if the total seen bytes so far exceeds the chunk size, flush.

    :cvar _max_repack: To fit the maximum number of entries into a node, we
        will sometimes start over and compress the whole list to get tighter
        packing. We get diminishing returns after a while, so this limits the
        number of times we will try.
        The default is to try to avoid recompressing entirely, but setting this
        to something like 20 will give maximum compression.

    :cvar _max_zsync: Another tunable nob. If _max_repack is set to 0, then you
        can limit the number of times we will try to pack more data into a
        node. This allows us to do a single compression pass, rather than
        trying until we overflow, and then recompressing again.
    """
    _repack_opts_for_speed = (0, 8)
    _repack_opts_for_size = (20, 0)

    def __init__(self, chunk_size, reserved=0, optimize_for_size=False):
        if False:
            while True:
                i = 10
        'Create a ChunkWriter to write chunk_size chunks.\n\n        :param chunk_size: The total byte count to emit at the end of the\n            chunk.\n        :param reserved: How many bytes to allow for reserved data. reserved\n            data space can only be written to via the write(..., reserved=True).\n        '
        self.chunk_size = chunk_size
        self.compressor = zlib.compressobj()
        self.bytes_in = []
        self.bytes_list = []
        self.bytes_out_len = 0
        self.unflushed_in_bytes = 0
        self.num_repack = 0
        self.num_zsync = 0
        self.unused_bytes = None
        self.reserved_size = reserved
        self.set_optimize(for_size=optimize_for_size)

    def finish(self):
        if False:
            i = 10
            return i + 15
        'Finish the chunk.\n\n        This returns the final compressed chunk, and either None, or the\n        bytes that did not fit in the chunk.\n\n        :return: (compressed_bytes, unused_bytes, num_nulls_needed)\n\n            * compressed_bytes: a list of bytes that were output from the\n              compressor. If the compressed length was not exactly chunk_size,\n              the final string will be a string of all null bytes to pad this\n              to chunk_size\n            * unused_bytes: None, or the last bytes that were added, which we\n              could not fit.\n            * num_nulls_needed: How many nulls are padded at the end\n        '
        self.bytes_in = None
        out = self.compressor.flush(Z_FINISH)
        self.bytes_list.append(out)
        self.bytes_out_len += len(out)
        if self.bytes_out_len > self.chunk_size:
            raise AssertionError('Somehow we ended up with too much compressed data, %d > %d' % (self.bytes_out_len, self.chunk_size))
        nulls_needed = self.chunk_size - self.bytes_out_len
        if nulls_needed:
            self.bytes_list.append('\x00' * nulls_needed)
        return (self.bytes_list, self.unused_bytes, nulls_needed)

    def set_optimize(self, for_size=True):
        if False:
            print('Hello World!')
        'Change how we optimize our writes.\n\n        :param for_size: If True, optimize for minimum space usage, otherwise\n            optimize for fastest writing speed.\n        :return: None\n        '
        if for_size:
            opts = ChunkWriter._repack_opts_for_size
        else:
            opts = ChunkWriter._repack_opts_for_speed
        (self._max_repack, self._max_zsync) = opts

    def _recompress_all_bytes_in(self, extra_bytes=None):
        if False:
            return 10
        'Recompress the current bytes_in, and optionally more.\n\n        :param extra_bytes: Optional, if supplied we will add it with\n            Z_SYNC_FLUSH\n        :return: (bytes_out, bytes_out_len, alt_compressed)\n\n            * bytes_out: is the compressed bytes returned from the compressor\n            * bytes_out_len: the length of the compressed output\n            * compressor: An object with everything packed in so far, and\n              Z_SYNC_FLUSH called.\n        '
        compressor = zlib.compressobj()
        bytes_out = []
        append = bytes_out.append
        compress = compressor.compress
        for accepted_bytes in self.bytes_in:
            out = compress(accepted_bytes)
            if out:
                append(out)
        if extra_bytes:
            out = compress(extra_bytes)
            out += compressor.flush(Z_SYNC_FLUSH)
            append(out)
        bytes_out_len = sum(map(len, bytes_out))
        return (bytes_out, bytes_out_len, compressor)

    def write(self, bytes, reserved=False):
        if False:
            print('Hello World!')
        'Write some bytes to the chunk.\n\n        If the bytes fit, False is returned. Otherwise True is returned\n        and the bytes have not been added to the chunk.\n\n        :param bytes: The bytes to include\n        :param reserved: If True, we can use the space reserved in the\n            constructor.\n        '
        if self.num_repack > self._max_repack and (not reserved):
            self.unused_bytes = bytes
            return True
        if reserved:
            capacity = self.chunk_size
        else:
            capacity = self.chunk_size - self.reserved_size
        comp = self.compressor
        next_unflushed = self.unflushed_in_bytes + len(bytes)
        remaining_capacity = capacity - self.bytes_out_len - 10
        if next_unflushed < remaining_capacity:
            out = comp.compress(bytes)
            if out:
                self.bytes_list.append(out)
                self.bytes_out_len += len(out)
            self.bytes_in.append(bytes)
            self.unflushed_in_bytes += len(bytes)
        else:
            self.num_zsync += 1
            if self._max_repack == 0 and self.num_zsync > self._max_zsync:
                self.num_repack += 1
                self.unused_bytes = bytes
                return True
            out = comp.compress(bytes)
            out += comp.flush(Z_SYNC_FLUSH)
            self.unflushed_in_bytes = 0
            if out:
                self.bytes_list.append(out)
                self.bytes_out_len += len(out)
            if self.num_repack == 0:
                safety_margin = 100
            else:
                safety_margin = 10
            if self.bytes_out_len + safety_margin <= capacity:
                self.bytes_in.append(bytes)
            else:
                self.num_repack += 1
                (bytes_out, this_len, compressor) = self._recompress_all_bytes_in(bytes)
                if self.num_repack >= self._max_repack:
                    self.num_repack += 1
                if this_len + 10 > capacity:
                    (bytes_out, this_len, compressor) = self._recompress_all_bytes_in()
                    self.compressor = compressor
                    self.num_repack = self._max_repack + 1
                    self.bytes_list = bytes_out
                    self.bytes_out_len = this_len
                    self.unused_bytes = bytes
                    return True
                else:
                    self.compressor = compressor
                    self.bytes_in.append(bytes)
                    self.bytes_list = bytes_out
                    self.bytes_out_len = this_len
        return False