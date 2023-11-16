"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.utils as utils
import volatility.debug as debug
import volatility.plugins.mac.common as common
import volatility.plugins.mac.list_kauth_scopes as kauth_scopes
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class mac_list_kauth_listeners(kauth_scopes.mac_list_kauth_scopes):
    """ Lists Kauth Scope listeners """

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        common.set_plugin_members(self)
        return TreeGrid([('Offset', Address), ('Scope', str), ('IData', Address), ('Callback Addr', Address), ('Callback Mod', str), ('Callback Sym', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        kaddr_info = common.get_handler_name_addrs(self)
        for scope in data:
            scope_name = scope.ks_identifier
            for ls in scope.listeners():
                cb = ls.kll_callback.v()
                (module, handler_sym) = common.get_handler_name(kaddr_info, cb)
                yield (0, [Address(ls.v()), str(scope_name), Address(ls.kll_idata), Address(cb), str(module), str(handler_sym)])

    def render_text(self, outfd, data):
        if False:
            return 10
        common.set_plugin_members(self)
        self.table_header(outfd, [('Offset', '[addrpad]'), ('Scope', '24'), ('IData', '[addrpad]'), ('Callback Addr', '[addrpad]'), ('Callback Mod', '24'), ('Callback Sym', '')])
        kaddr_info = common.get_handler_name_addrs(self)
        for scope in data:
            scope_name = scope.ks_identifier
            for ls in scope.listeners():
                cb = ls.kll_callback.v()
                (module, handler_sym) = common.get_handler_name(kaddr_info, cb)
                self.table_row(outfd, ls.v(), scope_name, ls.kll_idata, cb, module, handler_sym)