"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import re
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.lsmod as linux_lsmod
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_hidden_modules(linux_common.AbstractLinuxCommand):
    """Carves memory to find hidden kernel modules"""

    def walk_modules_address_space(self, addr_space):
        if False:
            return 10
        list_mods = [x[0].obj_offset for x in linux_lsmod.linux_lsmod(self._config).calculate()]
        if addr_space.profile.get_symbol('module_addr_min'):
            min_addr_sym = obj.Object('unsigned long', offset=addr_space.profile.get_symbol('module_addr_min'), vm=addr_space)
            max_addr_sym = obj.Object('unsigned long', offset=addr_space.profile.get_symbol('module_addr_max'), vm=addr_space)
        elif addr_space.profile.get_symbol('mod_tree'):
            skip_size = addr_space.profile.get_obj_size('latch_tree_root')
            addr = addr_space.profile.get_symbol('mod_tree')
            ulong_size = addr_space.profile.get_obj_size('unsigned long')
            min_addr_sym = obj.Object('unsigned long', offset=addr + skip_size, vm=addr_space)
            max_addr_sym = obj.Object('unsigned long', offset=addr + skip_size + ulong_size, vm=addr_space)
        else:
            debug.error('Unsupport kernel verison. Please file a bug ticket that includes your kernel version and distribution.')
        min_addr = min_addr_sym & ~4095
        max_addr = (max_addr_sym & ~4095) + 4096
        scan_buf = ''
        llen = max_addr - min_addr
        allfs = 'ÿ' * 4096
        memory_model = self.addr_space.profile.metadata.get('memory_model', '32bit')
        if memory_model == '32bit':
            minus_size = 4
        else:
            minus_size = 8
        check_bufs = []
        replace_bufs = []
        check_nums = [3000, 2800, 2700, 2500, 2300, 2100, 2000, 1500, 1300, 1200, 1024, 512, 256, 128, 96, 64, 48, 32, 24]
        for num in check_nums:
            check_bufs.append('\x00' * num)
            replace_bufs.append('ÿ' * (num - minus_size) + '\x00' * minus_size)
        for page in range(min_addr, max_addr, 4096):
            to_append = allfs
            tmp = addr_space.read(page, 4096)
            if tmp:
                non_zero = False
                for t in tmp:
                    if t != '\x00':
                        non_zero = True
                        break
                if non_zero:
                    for i in range(len(check_nums)):
                        tmp = tmp.replace(check_bufs[i], replace_bufs[i])
                    to_append = tmp
            scan_buf = scan_buf + to_append
        for cur_addr in re.finditer('(?=(\x00\x00\x00\x00|\x01\x00\x00\x00|\x02\x00\x00\x00))', scan_buf):
            mod_addr = min_addr + cur_addr.start()
            if mod_addr in list_mods:
                continue
            m = obj.Object('module', offset=mod_addr, vm=addr_space)
            if m.is_valid():
                yield m

    def calculate(self):
        if False:
            return 10
        linux_common.set_plugin_members(self)
        for mod in self.walk_modules_address_space(self.addr_space):
            yield mod

    def unified_output(self, data):
        if False:
            for i in range(10):
                print('nop')
        return TreeGrid([('Offset(V)', Address), ('Name', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for module in data:
            yield (0, [Address(module.obj_offset), str(module.name)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Offset (V)', '[addrpad]'), ('Name', '')])
        for module in data:
            self.table_row(outfd, module.obj_offset, str(module.name))