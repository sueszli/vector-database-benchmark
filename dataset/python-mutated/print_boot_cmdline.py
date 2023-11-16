"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.common as common
from volatility.renderers import TreeGrid

class mac_print_boot_cmdline(common.AbstractMacCommand):
    """ Prints kernel boot arguments """

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        common.set_plugin_members(self)
        pe_state_addr = self.addr_space.profile.get_symbol('_PE_state')
        pe_state = obj.Object('PE_state', offset=pe_state_addr, vm=self.addr_space)
        bootargs = pe_state.bootArgs.dereference_as('boot_args')
        yield bootargs.CommandLine

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Command Line', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for cmdline in data:
            yield (0, [str(cmdline)])

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.table_header(outfd, [('Command Line', '')])
        for cmdline in data:
            self.table_row(outfd, cmdline)