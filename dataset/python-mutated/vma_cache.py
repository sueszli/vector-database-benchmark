"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization:
"""
import volatility.obj as obj
import volatility.plugins.linux.common as linux_common
from volatility.plugins.linux.slab_info import linux_slabinfo

class linux_vma_cache(linux_common.AbstractLinuxCommand):
    """Gather VMAs from the vm_area_struct cache"""

    def __init__(self, config, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        linux_common.AbstractLinuxCommand.__init__(self, config, *args, **kwargs)
        self._config.add_option('UNALLOCATED', short_option='u', default=False, help='Show unallocated', action='store_true')

    def calculate(self):
        if False:
            return 10
        linux_common.set_plugin_members(self)
        has_owner = self.profile.obj_has_member('mm_struct', 'owner')
        cache = linux_slabinfo(self._config).get_kmem_cache('vm_area_struct', self._config.UNALLOCATED)
        for vm in cache:
            start = vm.vm_start
            end = vm.vm_end
            if has_owner and vm.vm_mm and vm.vm_mm.is_valid():
                task = vm.vm_mm.owner
                (task_name, pid) = (task.comm, task.pid)
            else:
                (task_name, pid) = ('', '')
            if vm.vm_file and vm.vm_file.is_valid():
                path = vm.vm_file.dentry.get_partial_path()
            else:
                path = ''
            yield (task_name, pid, start, end, path)

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Process', '16'), ('PID', '6'), ('Start', '[addrpad]'), ('End', '[addrpad]'), ('Path', '')])
        for (task_name, pid, start, end, path) in data:
            self.table_row(outfd, task_name, pid, start, end, path)