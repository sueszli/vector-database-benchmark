"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
"""
import volatility.obj as obj
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_proc_maps(linux_pslist.linux_pslist):
    """Gathers process memory maps"""

    def calculate(self):
        if False:
            while True:
                i = 10
        linux_common.set_plugin_members(self)
        tasks = linux_pslist.linux_pslist.calculate(self)
        for task in tasks:
            if task.mm:
                for vma in task.get_proc_maps():
                    yield (task, vma)

    def unified_output(self, data):
        if False:
            for i in range(10):
                print('nop')
        return TreeGrid([('Offset', Address), ('Pid', int), ('Name', str), ('Start', Address), ('End', Address), ('Flags', str), ('Pgoff', Address), ('Major', int), ('Minor', int), ('Inode', int), ('Path', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for (task, vma) in data:
            (fname, major, minor, ino, pgoff) = vma.info(task)
            yield (0, [Address(task.obj_offset), int(task.pid), str(task.comm), Address(vma.vm_start), Address(vma.vm_end), str(vma.vm_flags), Address(pgoff), int(major), int(minor), int(ino), str(fname)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Offset', '#018x'), ('Pid', '8'), ('Name', '20'), ('Start', '#018x'), ('End', '#018x'), ('Flags', '6'), ('Pgoff', '[addr]'), ('Major', '6'), ('Minor', '6'), ('Inode', '10'), ('File Path', '')])
        for (task, vma) in data:
            (fname, major, minor, ino, pgoff) = vma.info(task)
            self.table_row(outfd, task.obj_offset, task.pid, task.comm, vma.vm_start, vma.vm_end, str(vma.vm_flags), pgoff, major, minor, ino, fname)