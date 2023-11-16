import volatility.obj as obj
import volatility.utils as utils
import volatility.plugins.linux.pslist as linux_pslist
import volatility.plugins.linux.pidhashtable as linux_pidhashtable
import volatility.plugins.linux.pslist_cache as linux_pslist_cache
import volatility.plugins.linux.psscan as linux_psscan
import volatility.plugins.linux.common as linux_common
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_psxview(linux_common.AbstractLinuxCommand):
    """Find hidden processes with various process listings"""

    def _get_pslist(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.addr_space.vtop(x.obj_offset) for x in linux_pslist.linux_pslist(self._config).calculate()]

    def _get_pid_hash(self):
        if False:
            print('Hello World!')
        return [self.addr_space.vtop(x.obj_offset) for x in linux_pidhashtable.linux_pidhashtable(self._config).calculate()]

    def _get_kmem_cache(self):
        if False:
            return 10
        return [self.addr_space.vtop(x.obj_offset) for x in linux_pslist_cache.linux_pslist_cache(self._config).calculate()]

    def _get_task_parents(self):
        if False:
            for i in range(10):
                print('nop')
        if self.addr_space.profile.obj_has_member('task_struct', 'real_parent'):
            ret = [self.addr_space.vtop(x.real_parent.v()) for x in linux_pslist.linux_pslist(self._config).calculate()]
        else:
            ret = [self.addr_space.vtop(x.parent.v()) for x in linux_pslist.linux_pslist(self._config).calculate()]
        return ret

    def _get_thread_leaders(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.addr_space.vtop(x.group_leader.v()) for x in linux_pidhashtable.linux_pidhashtable(self._config).calculate()]

    def _get_psscan(self):
        if False:
            while True:
                i = 10
        return [x.obj_offset for x in linux_psscan.linux_psscan(self._config).calculate()]

    def calculate(self):
        if False:
            i = 10
            return i + 15
        linux_common.set_plugin_members(self)
        phys_addr_space = utils.load_as(self._config, astype='physical')
        ps_sources = {}
        ps_sources['pslist'] = self._get_pslist()
        ps_sources['pid_hash'] = self._get_pid_hash()
        ps_sources['kmem_cache'] = self._get_kmem_cache()
        ps_sources['parents'] = self._get_task_parents()
        ps_sources['thread_leaders'] = self._get_thread_leaders()
        ps_sources['psscan'] = self._get_psscan()
        seen_offsets = []
        for source in ps_sources:
            tasks = ps_sources[source]
            for offset in tasks:
                if offset and offset not in seen_offsets:
                    seen_offsets.append(offset)
                    yield (offset, obj.Object('task_struct', offset=offset, vm=phys_addr_space), ps_sources)

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('Offset(V)', Address), ('Name', str), ('PID', int), ('pslist', str), ('psscan', str), ('pid_hash', str), ('kmem_cache', str), ('parents', str), ('leaders', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for (offset, process, ps_sources) in data:
            yield (0, [Address(offset), str(process.comm), int(process.pid), str(ps_sources['pslist'].__contains__(offset)), str(ps_sources['psscan'].__contains__(offset)), str(ps_sources['pid_hash'].__contains__(offset)), str(ps_sources['kmem_cache'].__contains__(offset)), str(ps_sources['parents'].__contains__(offset)), str(ps_sources['thread_leaders'].__contains__(offset))])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        self.table_header(outfd, [('Offset(V)', '[addrpad]'), ('Name', '<20'), ('PID', '>6'), ('pslist', '5'), ('psscan', '5'), ('pid_hash', '5'), ('kmem_cache', '5'), ('parents', '5'), ('leaders', '5')])
        for (offset, process, ps_sources) in data:
            self.table_row(outfd, offset, process.comm, process.pid, str(ps_sources['pslist'].__contains__(offset)), str(ps_sources['psscan'].__contains__(offset)), str(ps_sources['pid_hash'].__contains__(offset)), str(ps_sources['kmem_cache'].__contains__(offset)), str(ps_sources['parents'].__contains__(offset)), str(ps_sources['thread_leaders'].__contains__(offset)))