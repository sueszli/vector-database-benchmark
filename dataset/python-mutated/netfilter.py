"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.lsmod as linux_lsmod
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_netfilter(linux_common.AbstractLinuxCommand):
    """Lists Netfilter hooks"""

    def calculate(self):
        if False:
            i = 10
            return i + 15
        linux_common.set_plugin_members(self)
        hook_names = ['PRE_ROUTING', 'LOCAL_IN', 'FORWARD', 'LOCAL_OUT', 'POST_ROUTING']
        proto_names = ['', '', 'IPV4', '', '', '', '', '', '', '', '', '', '', '']
        nf_hooks_addr = self.addr_space.profile.get_symbol('nf_hooks')
        if nf_hooks_addr == None:
            debug.error('Unable to analyze NetFilter. It is either disabled or compiled as a module.')
        modules = linux_lsmod.linux_lsmod(self._config).get_modules()
        list_head_size = self.addr_space.profile.get_obj_size('list_head')
        for outer in range(13):
            arr = nf_hooks_addr + outer * (list_head_size * 8)
            for inner in range(7):
                list_head = obj.Object('list_head', offset=arr + inner * list_head_size, vm=self.addr_space)
                for hook_ops in list_head.list_of_type('nf_hook_ops', 'list'):
                    if self.is_known_address(hook_ops.hook.v(), modules):
                        hooked = 'False'
                    else:
                        hooked = 'True'
                    yield (proto_names[outer], hook_names[inner], hook_ops.hook.v(), hooked)

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('Proto', str), ('Hook', str), ('Handler', Address), ('IsHooked', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for (outer, inner, hook_addr, hooked) in data:
            yield (0, [str(outer), str(inner), Address(hook_addr), str(hooked)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Proto', '5'), ('Hook', '16'), ('Handler', '[addrpad]'), ('Is Hooked', '5')])
        for (outer, inner, hook_addr, hooked) in data:
            self.table_row(outfd, outer, inner, hook_addr, hooked)