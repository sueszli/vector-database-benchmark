"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization:
"""
import volatility.plugins.linux.common as linux_common
from volatility.plugins.linux.slab_info import linux_slabinfo

class linux_dentry_cache(linux_common.AbstractLinuxCommand):
    """Gather files from the dentry cache"""

    def __init__(self, config, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        linux_common.AbstractLinuxCommand.__init__(self, config, *args, **kwargs)
        self._config.add_option('UNALLOCATED', short_option='u', default=False, help='Show unallocated', action='store_true')

    def make_body(self, dentry):
        if False:
            i = 10
            return i + 15
        'Create a pipe-delimited bodyfile from a dentry structure. \n        \n        MD5|name|inode|mode_as_string|UID|GID|size|atime|mtime|ctime|crtime\n        '
        path = dentry.get_partial_path() or ''
        i = dentry.d_inode
        if i:
            ret = [0, path, i.i_ino, 0, i.uid, i.gid, i.i_size, i.i_atime, i.i_mtime, 0, i.i_ctime]
        else:
            ret = [0, path] + [0] * 8
        ret = '|'.join([str(val) for val in ret])
        return ret

    def calculate(self):
        if False:
            while True:
                i = 10
        linux_common.set_plugin_members(self)
        cache = linux_slabinfo(self._config).get_kmem_cache('dentry', self._config.UNALLOCATED)
        if cache == []:
            cache = linux_slabinfo(self._config).get_kmem_cache('dentry_cache', self._config.UNALLOCATED, struct_name='dentry')
        for dentry in cache:
            yield self.make_body(dentry)

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        for bodyline in data:
            outfd.write(bodyline + '\n')