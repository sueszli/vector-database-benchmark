"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import os
import volatility.obj as obj
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.lsmod as linux_lsmod

class linux_check_afinfo(linux_common.AbstractLinuxCommand):
    """Verifies the operation function pointers of network protocols"""

    def check_members(self, var_ops, members, modules):
        if False:
            for i in range(10):
                print('nop')
        for (hooked_member, hook_address) in self.verify_ops(var_ops, members, modules):
            yield (hooked_member, hook_address)

    def check_afinfo(self, var_name, var, op_members, seq_members, modules):
        if False:
            for i in range(10):
                print('nop')
        for (hooked_member, hook_address) in self.check_members(var.seq_fops, op_members, modules):
            yield (var_name, hooked_member, hook_address)
        if hasattr(var, 'seq_ops'):
            for (hooked_member, hook_address) in self.check_members(var.seq_ops, seq_members, modules):
                yield (var_name, hooked_member, hook_address)
        elif not self.is_known_address(var.seq_show, modules):
            yield (var_name, 'show', var.seq_show)

    def _pre_4_18(self, modules, seq_members):
        if False:
            print('Hello World!')
        op_members = self.profile.types['file_operations'].keywords['members'].keys()
        tcp = ('tcp_seq_afinfo', ['tcp6_seq_afinfo', 'tcp4_seq_afinfo'])
        udp = ('udp_seq_afinfo', ['udplite6_seq_afinfo', 'udp6_seq_afinfo', 'udplite4_seq_afinfo', 'udp4_seq_afinfo'])
        protocols = [tcp, udp]
        for proto in protocols:
            struct_type = proto[0]
            for global_var_name in proto[1]:
                global_var_addr = self.addr_space.profile.get_symbol(global_var_name)
                if not global_var_addr:
                    continue
                global_var = obj.Object(struct_type, offset=global_var_addr, vm=self.addr_space)
                for (name, member, address) in self.check_afinfo(global_var_name, global_var, op_members, seq_members, modules):
                    yield (name, member, address)

    def _4_18_plus(self, modules, seq_members):
        if False:
            i = 10
            return i + 15
        ops_structs = ['raw_seq_ops', 'udp_seq_ops', 'arp_seq_ops', 'unix_seq_ops', 'udp6_seq_opsraw6_seq_ops', 'tcp_seq_ops', 'tcp4_seq_ops', 'tcp6_seq_ops', 'packet_seq_ops']
        for op_struct in ops_structs:
            var_addr = self.profile.get_symbol(op_struct)
            if var_addr == None:
                continue
            var = obj.Object('seq_operations', offset=var_addr, vm=self.addr_space)
            for (hooked_member, hook_address) in self.check_members(var, seq_members, modules):
                yield (op_struct, hooked_member, hook_address)

    def calculate(self):
        if False:
            print('Hello World!')
        linux_common.set_plugin_members(self)
        modules = linux_lsmod.linux_lsmod(self._config).get_modules()
        seq_members = self.profile.types['seq_operations'].keywords['members'].keys()
        if self.addr_space.profile.obj_has_member('tcp_seq_afinfo', 'seq_fops'):
            func = self._pre_4_18
        else:
            func = self._4_18_plus
        for (name, member, address) in func(modules, seq_members):
            yield (name, member, address)

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Symbol Name', '42'), ('Member', '30'), ('Address', '[addrpad]')])
        for (what, member, address) in data:
            self.table_row(outfd, what, member, address)