"""
@author:       Georg Wicherski
@license:      GNU General Public License 2.0
@contact:      georg@crowdstrike.com
@organization: CrowdStrike, Inc.
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist
import volatility.plugins.linux.dump_map as linux_dump_map
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_elfs(linux_pslist.linux_pslist):
    """Find ELF binaries in process mappings"""

    def calculate(self):
        if False:
            i = 10
            return i + 15
        linux_common.set_plugin_members(self)
        tasks = linux_pslist.linux_pslist.calculate(self)
        for task in tasks:
            for (elf, elf_start, elf_end, soname, needed) in task.elfs():
                yield (task, elf, elf_start, elf_end, soname, needed)

    def unified_output(self, data):
        if False:
            for i in range(10):
                print('nop')
        return TreeGrid([('Pid', int), ('Name', str), ('Start', Address), ('End', Address), ('Path', str), ('Needed', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for (task, elf, start, end, soname, needed) in data:
            yield (0, [int(task.pid), str(task.comm), Address(start), Address(end), str(soname), ','.join(needed)])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Pid', '8'), ('Name', '17'), ('Start', '[addrpad]'), ('End', '[addrpad]'), ('Elf Path', '60'), ('Needed', '')])
        for (task, elf, start, end, soname, needed) in data:
            self.table_row(outfd, task.pid, task.comm, start, end, soname, ','.join(needed))