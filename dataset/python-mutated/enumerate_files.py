"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.find_file as linux_find_file
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_enumerate_files(linux_common.AbstractLinuxCommand):
    """Lists files referenced by the filesystem cache"""

    def calculate(self):
        if False:
            print('Hello World!')
        linux_common.set_plugin_members(self)
        for (_, _, file_path, file_dentry) in linux_find_file.linux_find_file(self._config).walk_sbs():
            inode = file_dentry.d_inode
            yield (inode, inode.i_ino, file_path)

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Inode Address', Address), ('Inode Number', int), ('Path', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for (inode, inum, path) in data:
            yield (0, [Address(inode.v()), int(inum), str(path)])

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('Inode Address', '[addr]'), ('Inode Number', '25'), ('Path', '')])
        for (inode, inum, path) in data:
            self.table_row(outfd, inode, inum, path)