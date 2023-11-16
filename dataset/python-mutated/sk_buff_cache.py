"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization:
"""
import os
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
from volatility.plugins.linux.slab_info import linux_slabinfo

class linux_sk_buff_cache(linux_common.AbstractLinuxCommand):
    """Recovers packets from the sk_buff kmem_cache"""

    def __init__(self, config, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.edir = None
        linux_common.AbstractLinuxCommand.__init__(self, config, *args, **kwargs)
        self._config.add_option('UNALLOCATED', short_option='u', default=False, help='Show unallocated', action='store_true')
        self._config.add_option('DUMP-DIR', short_option='D', default=None, help='output directory for recovered packets', action='store', type='str')

    def write_sk_buff(self, s):
        if False:
            while True:
                i = 10
        pkt_len = s.len
        if 0 < pkt_len < 104857600:
            start = s.data
            data = self.addr_space.zread(start, pkt_len)
            fname = '{0:x}'.format(s.obj_offset)
            fd = open(os.path.join(self.edir, fname), 'wb')
            fd.write(data)
            fd.close()
            yield 'Wrote {0:d} bytes to {1:s}'.format(pkt_len, fname)

    def walk_cache(self, cache_name):
        if False:
            while True:
                i = 10
        cache = linux_slabinfo(self._config).get_kmem_cache(cache_name, self._config.UNALLOCATED, struct_name='sk_buff')
        if not cache:
            return
        for s in cache:
            for msg in self.write_sk_buff(s):
                yield msg

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        linux_common.set_plugin_members(self)
        self.edir = self._config.DUMP_DIR
        if not self.edir:
            debug.error('No output directory given.')
        for msg in self.walk_cache('skbuff_head_cache'):
            yield msg
        for msg in self.walk_cache('skbuff_fclone_cache'):
            yield msg

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        for msg in data:
            outfd.write('{0:s}\n'.format(msg))