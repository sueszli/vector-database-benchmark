"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.common as common
import volatility.plugins.mac.lsmod as lsmod
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class mac_ip_filters(lsmod.mac_lsmod):
    """ Reports any hooked IP filters """

    def check_filter(self, context, fname, ptr, kernel_symbol_addresses, kmods):
        if False:
            i = 10
            return i + 15
        if ptr == None:
            return
        good = common.is_known_address_name(ptr, kernel_symbol_addresses, kmods)
        return (good, context, fname, ptr)

    def calculate(self):
        if False:
            return 10
        common.set_plugin_members(self)
        (kernel_symbol_addresses, kmods) = common.get_kernel_addrs(self)
        list_addrs = [self.addr_space.profile.get_symbol('_ipv4_filters'), self.addr_space.profile.get_symbol('_ipv6_filters')]
        for list_addr in list_addrs:
            plist = obj.Object('ipfilter_list', offset=list_addr, vm=self.addr_space)
            cur = plist.tqh_first
            while cur:
                filter = cur.ipf_filter
                name = filter.name.dereference()
                yield self.check_filter('INPUT', name, filter.ipf_input, kernel_symbol_addresses, kmods)
                yield self.check_filter('OUTPUT', name, filter.ipf_output, kernel_symbol_addresses, kmods)
                yield self.check_filter('DETACH', name, filter.ipf_detach, kernel_symbol_addresses, kmods)
                cur = cur.ipf_link.tqe_next

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('Context', str), ('Filter', str), ('Pointer', Address), ('Status', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for (good, context, fname, ptr) in data:
            status = 'OK'
            if good == 0:
                status = 'UNKNOWN'
            yield (0, [str(context), str(fname), Address(ptr), str(status)])

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.table_header(outfd, [('Context', '10'), ('Filter', '16'), ('Pointer', '[addrpad]'), ('Status', '')])
        for (good, context, fname, ptr) in data:
            status = 'OK'
            if good == 0:
                status = 'UNKNOWN'
            self.table_row(outfd, context, fname, ptr, status)