"""
@author:       Joe Sylve
@license:      GNU General Public License 2.0
@contact:      joe.sylve@gmail.com
@organization: 504ENSICS Labs
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_check_syscall_arm(linux_common.AbstractLinuxARMCommand):
    """ Checks if the system call table has been altered """

    def _get_syscall_table_size(self):
        if False:
            for i in range(10):
                print('nop')
        ' Get size of syscall table from the vector_swi function '
        vector_swi_addr = self.addr_space.profile.get_symbol('vector_swi')
        max_opcodes_to_check = 1024
        while max_opcodes_to_check:
            opcode = obj.Object('unsigned int', offset=vector_swi_addr, vm=self.addr_space)
            if opcode & 4294901760 == 3814129664:
                shift = 16 - ((opcode & 65280) >> 8)
                size = (opcode & 255) << 2 * shift
                return size
                break
            vector_swi_addr += 4
            max_opcodes_to_check -= 1
        debug.error('Syscall table size could not be determined.')

    def _get_syscall_table_address(self):
        if False:
            i = 10
            return i + 15
        ' returns the address of the syscall table '
        syscall_table_address = self.addr_space.profile.get_symbol('sys_call_table')
        if syscall_table_address:
            return syscall_table_address
        debug.error('Symbol sys_call_table not export.  Please file a bug report.')

    def calculate(self):
        if False:
            while True:
                i = 10
        ' \n        This works by walking the system call table \n        and verifies that each is a symbol in the kernel\n        '
        linux_common.set_plugin_members(self)
        num_syscalls = self._get_syscall_table_size()
        syscall_addr = self._get_syscall_table_address()
        sym_addrs = self.profile.get_all_addresses()
        table = obj.Object('Array', offset=syscall_addr, vm=self.addr_space, targetType='unsigned int', count=num_syscalls)
        for (i, call_addr) in enumerate(table):
            if not call_addr:
                continue
            call_addr = call_addr & 4294967295
            if not call_addr in sym_addrs:
                yield (i, call_addr, 1)
            else:
                yield (i, call_addr, 0)

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('Index', Address), ('Address', Address), ('Symbol', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for (i, call_addr, hooked) in data:
            if hooked == 0:
                sym_name = self.profile.get_symbol_by_address('kernel', call_addr)
            else:
                sym_name = 'HOOKED'
            yield 0[Address(i), Address(call_addr), str(sym_name)]

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Index', '[addr]'), ('Address', '[addrpad]'), ('Symbol', '<30')])
        for (i, call_addr, hooked) in data:
            if hooked == 0:
                sym_name = self.profile.get_symbol_by_address('kernel', call_addr)
            else:
                sym_name = 'HOOKED'
            self.table_row(outfd, i, call_addr, sym_name)