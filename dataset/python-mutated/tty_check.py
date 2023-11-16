"""
@author:       Joe Sylve
@license:      GNU General Public License 2.0
@contact:      joe.sylve@gmail.com
@organization: 504ENSICS Labs
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.lsmod as linux_lsmod
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_check_tty(linux_common.AbstractLinuxCommand):
    """Checks tty devices for hooks"""

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        linux_common.set_plugin_members(self)
        modules = linux_lsmod.linux_lsmod(self._config).get_modules()
        tty_addr = self.addr_space.profile.get_symbol('tty_drivers')
        if not tty_addr:
            debug.error('Symbol tty_drivers not found in kernel')
        drivers = obj.Object('list_head', offset=tty_addr, vm=self.addr_space)
        sym_cache = {}
        for tty in drivers.list_of_type('tty_driver', 'tty_drivers'):
            name = tty.name.dereference_as('String', length=linux_common.MAX_STRING_LENGTH)
            ttys = obj.Object('Array', targetType='Pointer', vm=self.addr_space, offset=tty.ttys, count=tty.num)
            for tty_dev in ttys:
                if tty_dev == 0:
                    continue
                tty_dev = tty_dev.dereference_as('tty_struct')
                name = tty_dev.name
                recv_buf = tty_dev.ldisc.ops.receive_buf
                known = self.is_known_address(recv_buf, modules)
                if not known:
                    sym_name = 'HOOKED'
                    hooked = 1
                else:
                    sym_name = self.profile.get_symbol_by_address('kernel', recv_buf)
                    hooked = 0
                sym_cache[recv_buf] = sym_name
                yield (name, recv_buf, sym_name, hooked)

    def unified_output(self, data):
        if False:
            print('Hello World!')
        return TreeGrid([('Name', str), ('Address', Address), ('Symbol', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        for (name, call_addr, sym_name, _hooked) in data:
            yield (0, [str(name), Address(call_addr), str(sym_name)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Name', '<16'), ('Address', '[addrpad]'), ('Symbol', '<30')])
        for (name, call_addr, sym_name, _hooked) in data:
            self.table_row(outfd, name, call_addr, sym_name)