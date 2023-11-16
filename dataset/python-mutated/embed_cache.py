import os.path
import logging
import hashlib
import argparse
import struct
import itertools
import sys
import subprocess
import re
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='%(asctime)-15s %(message)s')
CHAR_MAP = {i: '{}'.format(i) for i in range(256)}

def _u32(data):
    if False:
        return 10
    return struct.unpack('<I', data)[0]

class CacheDataGenerator:
    _cache_files = None

    def __init__(self, cache_files, remove_plat_info=True, append_cache=True):
        if False:
            while True:
                i = 10
        self._cache_files = cache_files
        self._remove_plat_info = remove_plat_info
        self._append_cache = append_cache

    def _get_hash(self):
        if False:
            for i in range(10):
                print('nop')
        return _u32(self._hash.digest()[:4])

    def gen_cache_data(self, fpath):
        if False:
            i = 10
            return i + 15
        fname = os.path.basename(fpath)
        with open(fpath, 'rb') as fcache:
            cache_data = fcache.read()
        if self._remove_plat_info:
            for matched in re.finditer(b'(layout_transform_profile:plat=.*);dev=.*;cap=\\d.\\d', cache_data):
                plat_info = matched.group(1)
                cat_info = cache_data[matched.span()[0] - 4:matched.span()[1]]
                cache_data = re.sub(cat_info, struct.pack('I', len(plat_info)) + plat_info, cache_data)
        cache_data = struct.unpack('<{}B'.format(len(cache_data)), cache_data)
        ret = list(map(CHAR_MAP.__getitem__, cache_data))
        for i in range(50, len(ret), 50):
            ret[i] = '\n' + ret[i]
        return ','.join(ret)

    def gen_cache_data_header(self, fout, src_map):
        if False:
            while True:
                i = 10
        if not self._append_cache:
            fout.write('// generated embed_cache.py\n')
            fout.write('#include <vector>\n')
            fout.write('#include <stdint.h>\n')
        for (k, v) in sorted(src_map.items()):
            fout.write('\nstatic const std::vector<uint8_t> {} = {{\n'.format(k.replace('.', '_')))
            fout.write('{}'.format(v))
            fout.write('};\n')

    def invoke(self, output):
        if False:
            for i in range(10):
                print('nop')
        logger.info('generate cache_data.h ...')
        fname2cache_data = {}
        for fname in self._cache_files:
            (base, ext) = os.path.splitext(os.path.basename(fname))
            assert ext == '.cache', 'ext: {}, fname {}'.format(ext, fname)
            assert base not in fname2cache_data, 'duplicated kernel: ' + base
            fname2cache_data[base] = self.gen_cache_data(fname)
        if self._append_cache:
            mode = 'a'
        else:
            mode = 'w'
        with open(output, mode) as fout:
            self.gen_cache_data_header(fout, fname2cache_data)
        logger.info('done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embed cubin into cpp source file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='output source file', required=True)
    parser.add_argument('-r', '--remove-plat-info', action='store_true', default=True, help='whether remove platform infomation in the cache (default: True)')
    parser.add_argument('-a', '--append-cache', action='store_true', default=True, help='whether append the cache (default: True)')
    parser.add_argument('cache', help='cache files to be embedded', nargs='+')
    args = parser.parse_args()
    cache_generator = CacheDataGenerator(args.cache, args.remove_plat_info, args.append_cache)
    cache_generator.invoke(args.output)