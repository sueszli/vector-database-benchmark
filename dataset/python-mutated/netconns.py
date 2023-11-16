"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.common as common
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class mac_network_conns(common.AbstractMacCommand):
    """ Lists network connections from kernel network structures """

    def _walk_pcb_hash(self, proto_pcbinfo):
        if False:
            return 10
        pcb_hash = obj.Object('Array', offset=proto_pcbinfo.hashbase, vm=self.addr_space, targetType='Pointer', count=proto_pcbinfo.hashmask + 1)
        for pcb_ent in pcb_hash:
            head = pcb_ent.cast('inpcbhead')
            if not head:
                continue
            inpcb = head.lh_first.dereference_as('inpcb')
            while inpcb:
                yield inpcb
                inpcb = inpcb.inp_hash.le_next

    def _walk_pcb_list(self, proto_pcbinfo):
        if False:
            i = 10
            return i + 15
        inpcb = proto_pcbinfo.listhead.lh_first.dereference_as('inpcb')
        while inpcb:
            yield inpcb
            inpcb = inpcb.inp_list.le_next

    def _walk_pcb_entries(self, inpcbinfo_addr):
        if False:
            for i in range(10):
                print('nop')
        pcbs = {}
        inpcbinfo = obj.Object('inpcbinfo', offset=inpcbinfo_addr, vm=self.addr_space)
        for pcbinfo in self._walk_pcb_list(inpcbinfo):
            pcbs[pcbinfo.obj_offset] = pcbinfo
        for pcbinfo in self._walk_pcb_hash(inpcbinfo):
            pcbs[pcbinfo.obj_offset] = pcbinfo
        for pcbinfo in pcbs.values():
            (lip, lport, rip, rport) = pcbinfo.ipv4_info()
            yield (pcbinfo, lip, lport, rip, rport)

    def calculate(self):
        if False:
            while True:
                i = 10
        common.set_plugin_members(self)
        entries = []
        tcbinfo_addr = self.addr_space.profile.get_symbol('_tcbinfo')
        udbinfo_addr = self.addr_space.profile.get_symbol('_udbinfo')
        ripdbinfo_addr = self.addr_space.profile.get_symbol('_ripcbinfo')
        info_addrs = [('TCP', tcbinfo_addr), ('UDP', udbinfo_addr), ('RAW', ripdbinfo_addr)]
        for (proto_str, info_addr) in info_addrs:
            for (pcbinfo, lip, lport, rip, rport) in self._walk_pcb_entries(info_addr):
                if proto_str == 'TCP':
                    state = pcbinfo.get_tcp_state()
                else:
                    state = ''
                yield (proto_str, pcbinfo, lip, lport, rip, rport, state)

    def unified_output(self, data):
        if False:
            print('Hello World!')
        return TreeGrid([('Offset (V)', Address), ('Protocol', str), ('Local IP', str), ('Local Port', int), ('Remote IP', str), ('Remote Port', int), ('State', str)], self.generator(data))

    def generator(self, data):
        if False:
            i = 10
            return i + 15
        for (proto, pcb, lip, lport, rip, rport, state) in data:
            yield (0, [Address(pcb.obj_offset), str(proto), str(lip), int(lport), str(rip), int(rport), str(state)])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        self.table_header(outfd, [('Offset (V)', '[addrpad]'), ('Protocol', '4'), ('Local IP', '20'), ('Local Port', '6'), ('Remote IP', '20'), ('Remote Port', '6'), ('State', '')])
        for (proto, pcb, lip, lport, rip, rport, state) in data:
            self.table_row(outfd, pcb.obj_offset, proto, lip, lport, rip, rport, state)