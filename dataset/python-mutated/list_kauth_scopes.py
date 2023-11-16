"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.utils as utils
import volatility.debug as debug
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address
import volatility.plugins.mac.common as common

class mac_list_kauth_scopes(common.AbstractMacCommand):
    """ Lists Kauth Scopes and their status """

    def calculate(self):
        if False:
            i = 10
            return i + 15
        common.set_plugin_members(self)
        scopes_addr = self.addr_space.profile.get_symbol('_kauth_scopes')
        scopes_ptr = obj.Object('Pointer', offset=scopes_addr, vm=self.addr_space)
        scope = scopes_ptr.dereference_as('kauth_scope')
        while scope.is_valid():
            yield scope
            scope = scope.ks_link.tqe_next.dereference()

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        common.set_plugin_members(self)
        return TreeGrid([('Offset', Address), ('Name', str), ('IData', Address), ('Listeners', int), ('Callback Addr', Address), ('Callback Mod', str), ('Callback Sym', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        kaddr_info = common.get_handler_name_addrs(self)
        for scope in data:
            cb = scope.ks_callback.v()
            (module, handler_sym) = common.get_handler_name(kaddr_info, cb)
            yield (0, [Address(scope.v()), str(scope.ks_identifier), Address(scope.ks_idata), int(len([l for l in scope.listeners()])), Address(cb), str(module), str(handler_sym)])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        common.set_plugin_members(self)
        self.table_header(outfd, [('Offset', '[addrpad]'), ('Name', '24'), ('IData', '[addrpad]'), ('Listeners', '5'), ('Callback Addr', '[addrpad]'), ('Callback Mod', '24'), ('Callback Sym', '')])
        kaddr_info = common.get_handler_name_addrs(self)
        for scope in data:
            cb = scope.ks_callback.v()
            (module, handler_sym) = common.get_handler_name(kaddr_info, cb)
            self.table_row(outfd, scope.v(), scope.ks_identifier, scope.ks_idata, len([l for l in scope.listeners()]), cb, module, handler_sym)