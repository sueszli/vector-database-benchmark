"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.plugins.linux.common as linux_common
import volatility.debug as debug
import volatility.obj as obj
from volatility.renderers import TreeGrid

class linux_ifconfig(linux_common.AbstractLinuxCommand):
    """Gathers active interfaces"""

    def _get_devs_base(self):
        if False:
            for i in range(10):
                print('nop')
        net_device_ptr = obj.Object('Pointer', offset=self.addr_space.profile.get_symbol('dev_base'), vm=self.addr_space)
        net_device = net_device_ptr.dereference_as('net_device')
        for net_dev in linux_common.walk_internal_list('net_device', 'next', net_device):
            yield net_dev

    def _get_devs_namespace(self):
        if False:
            print('Hello World!')
        nslist_addr = self.addr_space.profile.get_symbol('net_namespace_list')
        nethead = obj.Object('list_head', offset=nslist_addr, vm=self.addr_space)
        for net in nethead.list_of_type('net', 'list'):
            for net_dev in net.dev_base_head.list_of_type('net_device', 'dev_list'):
                yield net_dev

    def _gather_net_dev_info(self, net_dev):
        if False:
            i = 10
            return i + 15
        mac_addr = net_dev.mac_addr
        promisc = str(net_dev.promisc)
        in_dev = obj.Object('in_device', offset=net_dev.ip_ptr, vm=self.addr_space)
        for dev in in_dev.devices():
            ip_addr = dev.ifa_address.cast('IpAddress')
            name = dev.ifa_label
            yield (name, ip_addr, mac_addr, promisc)

    def calculate(self):
        if False:
            while True:
                i = 10
        linux_common.set_plugin_members(self)
        if self.addr_space.profile.get_symbol('net_namespace_list'):
            func = self._get_devs_namespace
        elif self.addr_space.profile.get_symbol('dev_base'):
            func = self._get_devs_base
        else:
            debug.error('Unable to determine ifconfig information')
        for net_dev in func():
            for (name, ip_addr, mac_addr, promisc) in self._gather_net_dev_info(net_dev):
                yield (name, ip_addr, mac_addr, promisc)

    def unified_output(self, data):
        if False:
            for i in range(10):
                print('nop')
        return TreeGrid([('Interface', str), ('IP', str), ('MAC', str), ('Promiscuous', str)], self.generator(data))

    def generator(self, data):
        if False:
            i = 10
            return i + 15
        for (name, ip_addr, mac_addr, promisc) in data:
            yield (0, [str(name), str(ip_addr), str(mac_addr), str(promisc)])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Interface', '16'), ('IP Address', '20'), ('MAC Address', '18'), ('Promiscous Mode', '5')])
        for (name, ip_addr, mac_addr, promisc) in data:
            self.table_row(outfd, name, ip_addr, mac_addr, promisc)