"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.common as common
import volatility.plugins.mac.lsmod as lsmod
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class mac_notifiers(lsmod.mac_lsmod):
    """ Detects rootkits that add hooks into I/O Kit (e.g. LogKext) """

    def _struct_or_class(self, type_name):
        if False:
            while True:
                i = 10
        'Return the name of a structure or class. \n\n        More recent versions of OSX define some types as \n        classes instead of structures, so the naming is\n        a little different.   \n        '
        if self.addr_space.profile.vtypes.has_key(type_name):
            return type_name
        else:
            return type_name + '_class'

    def calculate(self):
        if False:
            while True:
                i = 10
        common.set_plugin_members(self)
        (kernel_symbol_addresses, kmods) = common.get_kernel_addrs(self)
        gnotify_addr = common.get_cpp_sym('gNotifications', self.addr_space.profile)
        p = obj.Object('Pointer', offset=gnotify_addr, vm=self.addr_space)
        gnotifications = p.dereference_as(self._struct_or_class('OSDictionary'))
        if gnotifications.count > 1024:
            return
        ents = obj.Object('Array', offset=gnotifications.dictionary, vm=self.addr_space, targetType=self._struct_or_class('dictEntry'), count=gnotifications.count)
        for ent in ents:
            if ent == None or not ent.is_valid():
                continue
            key = str(ent.key.dereference_as(self._struct_or_class('OSString')))
            valset = ent.value.dereference_as(self._struct_or_class('OSOrderedSet'))
            if valset == None or valset.count > 1024:
                continue
            notifiers_ptrs = obj.Object('Array', offset=valset.array, vm=self.addr_space, targetType='Pointer', count=valset.count)
            if notifiers_ptrs == None:
                continue
            for ptr in notifiers_ptrs:
                notifier = ptr.dereference_as(self._struct_or_class('_IOServiceNotifier'))
                if notifier == None:
                    continue
                matches = self.get_matching(notifier)
                if matches == []:
                    continue
                handler = notifier.handler.v()
                ch = notifier.compatHandler.v()
                if ch:
                    handler = ch
                (good, module) = common.is_known_address_name(handler, kernel_symbol_addresses, kmods)
                yield (good, module, key, notifier, matches, handler)

    def get_matching(self, notifier):
        if False:
            i = 10
            return i + 15
        matches = []
        if notifier.matching.count > 1024:
            return matches
        ents = obj.Object('Array', offset=notifier.matching.dictionary, vm=self.addr_space, targetType=self._struct_or_class('dictEntry'), count=notifier.matching.count)
        for ent in ents:
            if ent == None or ent.value == None:
                continue
            match = ent.value.dereference_as(self._struct_or_class('OSString'))
            if len(str(match)) > 0:
                matches.append(str(match))
        return ','.join(matches)

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Key', str), ('Matches', str), ('Handler', Address), ('Module', str), ('Status', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for (good, module, key, _, matches, handler) in data:
            if good == 0:
                status = 'UNKNOWN'
            else:
                status = 'OK'
            yield (0, [str(key), str(matches), Address(handler), str(module), str(status)])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Key', '30'), ('Matches', '40'), ('Handler', '[addrpad]'), ('Module', '40'), ('Status', '')])
        for (good, module, key, _, matches, handler) in data:
            status = 'OK'
            if good == 0:
                status = 'UNKNOWN'
            self.table_row(outfd, key, matches, handler, module, status)