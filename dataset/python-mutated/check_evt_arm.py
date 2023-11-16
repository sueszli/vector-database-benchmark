"""
@author:       Joe Sylve
@license:      GNU General Public License 2.0
@contact:      joe.sylve@gmail.com
@organization: 504ENSICS Labs
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common

class linux_check_evt_arm(linux_common.AbstractLinuxARMCommand):
    """ Checks the Exception Vector Table to look for syscall table hooking """
    VECTOR_BASE = 4294901760
    SWI_BASE = VECTOR_BASE + 8

    def calculate(self):
        if False:
            return 10
        linux_common.set_plugin_members(self)
        swi = obj.Object('unsigned int', offset=self.SWI_BASE, vm=self.addr_space)
        offset = (swi & 4095) + 8
        if swi & 4294963200 == 3852464128:
            yield ('SWI Offset Instruction', 'PASS', 'Offset: {0}'.format(offset))
        else:
            yield ('SWI Offset Instruction', 'FAIL', '{0:X}'.format(swi))
            return
        vector_swi_addr = obj.Object('unsigned int', offset=self.SWI_BASE + offset, vm=self.addr_space)
        if vector_swi_addr == self.addr_space.profile.get_symbol('vector_swi'):
            yield ('vector_swi address', 'PASS', '0x{0:X}'.format(vector_swi_addr))
        else:
            yield ('vector_swi address', 'FAIL', '0x{0:X}'.format(vector_swi_addr))
            return
        sc_opcode = None
        max_opcodes_to_check = 1024
        while max_opcodes_to_check:
            opcode = obj.Object('unsigned int', offset=vector_swi_addr, vm=self.addr_space)
            if opcode & 4294967040 == 3801055232:
                sc_opcode = opcode
                break
            vector_swi_addr += 4
            max_opcodes_to_check -= 1
        if sc_opcode:
            yield ('vector_swi code modification', 'PASS', '{0:X}'.format(sc_opcode))
        else:
            yield ('vector_swi code modification', 'FAIL', 'Opcode E28F80?? not found')
            return

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('Check', '<30'), ('PASS/FAIL', '<5'), ('Info', '<30')])
        for (check, result, info) in data:
            self.table_row(outfd, check, result, info)