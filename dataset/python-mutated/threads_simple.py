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

class mac_threads_simple(pstasks.mac_tasks):
    """ Lists threads along with their start time and priority """

    def unified_output(self, data):
        if False:
            return 10
        common.set_plugin_members(self)
        return TreeGrid([('PID', int), ('Name', str), ('Start Time', str), ('Priority', int), ('Start Function', Address), ('Function Map', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        kaddr_info = common.get_handler_name_addrs(self)
        for proc in data:
            for th in proc.threads():
                func_addr = th.continuation
                (module, handler_sym) = common.get_handler_name(kaddr_info, func_addr)
                if handler_sym:
                    handler = handler_sym
                elif module:
                    handler = module
                else:
                    handler = proc.find_map_path(func_addr)
                yield (0, [int(proc.p_pid), str(proc.p_comm), str(th.start_time()), int(th.sched_pri), Address(func_addr), str(handler)])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        common.set_plugin_members(self)
        self.table_header(outfd, [('PID', '8'), ('Name', '16'), ('Start Time', '32'), ('Priority', '6'), ('Start Function', '[addrpad]'), ('Function Map', '')])
        kaddr_info = common.get_handler_name_addrs(self)
        for proc in data:
            for th in proc.threads():
                func_addr = th.continuation
                (module, handler_sym) = common.get_handler_name(kaddr_info, func_addr)
                if handler_sym:
                    handler = handler_sym
                elif module:
                    handler = module
                else:
                    handler = proc.find_map_path(func_addr)
                self.table_row(outfd, proc.p_pid, proc.p_comm, th.start_time(), th.sched_pri, func_addr, handler)