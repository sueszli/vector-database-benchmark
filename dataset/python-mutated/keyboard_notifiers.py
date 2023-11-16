"""
@author:       Joe Sylve
@license:      GNU General Public License 2.0
@contact:      joe.sylve@gmail.com
@organization: 504ENSICS Labs
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common

class linux_keyboard_notifiers(linux_common.AbstractLinuxCommand):
    """Parses the keyboard notifier call chain"""

    def calculate(self):
        if False:
            while True:
                i = 10
        linux_common.set_plugin_members(self)
        knl_addr = self.addr_space.profile.get_symbol('keyboard_notifier_list')
        if not knl_addr:
            debug.error('Symbol keyboard_notifier_list not found in kernel')
        knl = obj.Object('atomic_notifier_head', offset=knl_addr, vm=self.addr_space)
        symbol_cache = {}
        for call_back in linux_common.walk_internal_list('notifier_block', 'next', knl.head):
            call_addr = call_back.notifier_call
            if symbol_cache.has_key(call_addr):
                sym_name = symbol_cache[call_addr]
                hooked = 0
            else:
                sym_name = self.profile.get_symbol_by_address('kernel', call_addr)
                if not sym_name:
                    sym_name = 'HOOKED'
                hooked = 1
            symbol_cache[call_addr] = sym_name
            yield (call_addr, sym_name, hooked)

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('Address', '[addrpad]'), ('Symbol', '<30')])
        for (call_addr, sym_name, _) in data:
            self.table_row(outfd, call_addr, sym_name)