"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization:
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_library_list(linux_pslist.linux_pslist):
    """ Lists libraries loaded into a process """

    def calculate(self):
        if False:
            return 10
        linux_common.set_plugin_members(self)
        tasks = linux_pslist.linux_pslist.calculate(self)
        for task in tasks:
            for mapping in task.get_libdl_maps():
                if mapping.l_name == '' or mapping.l_addr == 0:
                    continue
                yield (task, mapping)

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Task', str), ('Pid', int), ('LoadAddress', Address), ('Path', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        for (task, mapping) in data:
            yield (0, [str(task.comm), int(task.pid), Address(mapping.l_addr), str(mapping.l_name)])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        self.table_header(outfd, [('Task', '16'), ('Pid', '8'), ('Load Address', '[addrpad]'), ('Path', '')])
        for (task, mapping) in data:
            self.table_row(outfd, task.comm, task.pid, mapping.l_addr, mapping.l_name)