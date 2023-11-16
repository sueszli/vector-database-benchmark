"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.plthook as linux_plthook
import volatility.plugins.linux.pslist as linux_pslist
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_apihooks(linux_pslist.linux_pslist):
    """Checks for userland apihooks"""

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Pid', int), ('Name', str), ('HookVMA', str), ('HookSymbol', str), ('HookedAddress', Address), ('HookType', str), ('HookAddress', Address), ('HookLibrary', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        linux_common.set_plugin_members(self)
        try:
            import distorm3
        except ImportError:
            debug.error('this plugin requres the distorm library to operate.')
        for task in data:
            for (hook_desc, sym_name, addr, hook_type, hook_addr, hookfuncdesc) in task.apihook_info():
                yield (0, [int(task.pid), str(task.comm), str(hook_desc), str(sym_name), Address(addr), str(hook_type), Address(hook_addr), str(hookfuncdesc)])

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.table_header(outfd, [('Pid', '7'), ('Name', '16'), ('Hook VMA', '40'), ('Hook Symbol', '24'), ('Hooked Address', '[addrpad]'), ('Type', '5'), ('Hook Address', '[addrpad]'), ('Hook Library', '')])
        linux_common.set_plugin_members(self)
        try:
            import distorm3
        except ImportError:
            debug.error('this plugin requres the distorm library to operate.')
        for task in data:
            for (hook_desc, sym_name, addr, hook_type, hook_addr, hookfuncdesc) in task.apihook_info():
                self.table_row(outfd, task.pid, task.comm, hook_desc, sym_name, addr, hook_type, hook_addr, hookfuncdesc)