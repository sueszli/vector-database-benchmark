"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_ldrmodules(linux_pslist.linux_pslist):
    """Compares the output of proc maps with the list of libraries from libdl"""

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Pid', int), ('Name', str), ('Start', Address), ('Path', str), ('Kernel', str), ('Libc', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        for task in data:
            for (vm_start, vma_name, pmaps, dmaps) in task.ldrmodules():
                yield (0, [int(task.pid), str(task.comm), Address(vm_start), str(vma_name), str(pmaps), str(dmaps)])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Pid', '8'), ('Name', '16'), ('Start', '#018x'), ('File Path', '50'), ('Kernel', '6'), ('Libc', '6')])
        for task in data:
            for (vm_start, vma_name, pmaps, dmaps) in task.ldrmodules():
                self.table_row(outfd, task.pid, str(task.comm), vm_start, vma_name, pmaps, dmaps)