"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.pstasks as pstasks
import volatility.plugins.mac.common as common
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class mac_dyld_maps(pstasks.mac_tasks):
    """ Gets memory maps of processes from dyld data structures """

    def unified_output(self, data):
        if False:
            for i in range(10):
                print('nop')
        common.set_plugin_members(self)
        return TreeGrid([('Pid', int), ('Name', str), ('Start', Address), ('Map Name', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for proc in data:
            for map in proc.get_dyld_maps():
                yield (0, [int(proc.p_pid), str(proc.p_comm), Address(map.imageLoadAddress), str(map.imageFilePath)])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        common.set_plugin_members(self)
        self.table_header(outfd, [('Pid', '8'), ('Name', '20'), ('Start', '#018x'), ('Map Name', '')])
        for proc in data:
            for map in proc.get_dyld_maps():
                self.table_row(outfd, str(proc.p_pid), proc.p_comm, map.imageLoadAddress, map.imageFilePath)