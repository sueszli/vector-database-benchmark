"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import sys
import volatility.obj as obj
import volatility.plugins.mac.common as common
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address
from lsmod import mac_lsmod as mac_lsmod

class mac_trustedbsd(mac_lsmod):
    """ Lists malicious trustedbsd policies """

    def get_members(self):
        if False:
            return 10
        h = self.profile.types['mac_policy_ops']
        return h.keywords['members']

    def calculate(self):
        if False:
            while True:
                i = 10
        common.set_plugin_members(self)
        ops_members = self.get_members()
        (kernel_symbol_addresses, kmods) = common.get_kernel_addrs(self)
        list_addr = self.addr_space.profile.get_symbol('_mac_policy_list')
        plist = obj.Object('mac_policy_list', offset=list_addr, vm=self.addr_space)
        parray = obj.Object('Array', offset=plist.entries, vm=self.addr_space, targetType='mac_policy_list_element', count=plist.staticmax + 1)
        for ent in parray:
            if ent.mpc == None:
                continue
            name = ent.mpc.mpc_name.dereference()
            ops = obj.Object('mac_policy_ops', offset=ent.mpc.mpc_ops, vm=self.addr_space)
            for check in ops_members:
                ptr = ops.__getattr__(check)
                if ptr.v() != 0 and ptr.is_valid():
                    (good, module) = common.is_known_address_name(ptr, kernel_symbol_addresses, kmods)
                    yield (good, check, module, name, ptr)

    def unified_output(self, data):
        if False:
            print('Hello World!')
        return TreeGrid([('Check', str), ('Name', str), ('Pointer', Address), ('Module', str), ('Status', str)], self.generator(data))

    def generator(self, data):
        if False:
            i = 10
            return i + 15
        for (good, check, module, name, ptr) in data:
            status = 'HOOKED'
            if good:
                status = 'OK'
            yield (0, [str(check), str(name), Address(ptr), str(module), str(status)])

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.table_header(outfd, [('Check', '40'), ('Name', '20'), ('Pointer', '[addrpad]'), ('Module', ''), ('Status', '')])
        for (good, check, module, name, ptr) in data:
            status = 'HOOKED'
            if good:
                status = 'OK'
            self.table_row(outfd, check, name, ptr, module, status)