"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.mac.common as common
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class mac_lsmod_iokit(common.AbstractMacCommand):
    """ Lists loaded kernel modules through IOkit """

    def _struct_or_class(self, type_name):
        if False:
            for i in range(10):
                print('nop')
        'Return the name of a structure or class. \n\n        More recent versions of OSX define some types as \n        classes instead of structures, so the naming is\n        a little different.   \n        '
        if self.addr_space.profile.vtypes.has_key(type_name):
            return type_name
        else:
            return type_name + '_class'

    def calculate(self):
        if False:
            i = 10
            return i + 15
        common.set_plugin_members(self)
        saddr = common.get_cpp_sym('sLoadedKexts', self.addr_space.profile)
        p = obj.Object('Pointer', offset=saddr, vm=self.addr_space)
        kOSArr = obj.Object(self._struct_or_class('OSArray'), offset=p, vm=self.addr_space)
        if kOSArr == None:
            debug.error('The OSArray_class type was not found in the profile. Please file a bug if you are running aginst Mac >= 10.7')
        kext_arr = obj.Object(theType='Array', targetType='Pointer', offset=kOSArr.array, count=kOSArr.capacity, vm=self.addr_space)
        for (i, kext) in enumerate(kext_arr):
            kext = kext.dereference_as(self._struct_or_class('OSKext'))
            if kext and kext.is_valid() and kext.kmod_info.address.is_valid():
                yield kext

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Offset (V)', Address), ('Module Address', Address), ('Size', str), ('Refs', str), ('Version', str), ('Name', str), ('Path', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for kext in data:
            path = kext.path
            if path:
                path = str(path.dereference())
            yield (0, [Address(kext.kmod_info), Address(kext.kmod_info.address), str(kext.kmod_info.m('size')), str(kext.kmod_info.reference_count), str(kext.version), str(kext.kmod_info.name), str(path)])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        self.table_header(outfd, [('Offset (V)', '[addrpad]'), ('Module Address', '[addrpad]'), ('Size', '8'), ('Refs', '^8'), ('Version', '12'), ('Name', '48'), ('Path', '')])
        for kext in data:
            path = kext.path
            if path:
                path = str(path.dereference())
            self.table_row(outfd, kext.kmod_info, kext.kmod_info.address, kext.kmod_info.m('size'), kext.kmod_info.reference_count, kext.version, kext.kmod_info.name, str(path))