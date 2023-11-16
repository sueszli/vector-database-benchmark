import volatility.utils as utils
import volatility.obj as obj
import volatility.plugins.common as common
import volatility.win32.modules as modules
import volatility.win32.tasks as tasks
import volatility.debug as debug
import volatility.plugins.malware.malfind as malfind
import volatility.exceptions as exceptions
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address, Hex
GDT_DESCRIPTORS = dict(enumerate(['Data RO', 'Data RO Ac', 'Data RW', 'Data RW Ac', 'Data RO E', 'Data RO EA', 'Data RW E', 'Data RW EA', 'Code EO', 'Code EO Ac', 'Code RE', 'Code RE Ac', 'Code EO C', 'Code EO CA', 'Code RE C', 'Code RE CA', '<Reserved>', 'TSS16 Avl', 'LDT', 'TSS16 Busy', 'CallGate16', 'TaskGate', 'Int Gate16', 'TrapGate16', '<Reserved>', 'TSS32 Avl', '<Reserved>', 'TSS32 Busy', 'CallGate32', '<Reserved>', 'Int Gate32', 'TrapGate32']))

class _KIDTENTRY(obj.CType):
    """Class for interrupt descriptors"""

    @property
    def Address(self):
        if False:
            i = 10
            return i + 15
        'Return the address of the IDT entry handler'
        if self.ExtendedOffset == 0:
            return 0
        return self.ExtendedOffset.v() << 16 | self.Offset.v()

class _KGDTENTRY(obj.CType):
    """A class for GDT entries"""

    @property
    def Type(self):
        if False:
            print('Hello World!')
        'Get a string name of the descriptor type'
        flag = self.HighWord.Bits.Type.v() & 1 << 4
        typeval = self.HighWord.Bits.Type.v() & ~(1 << 4)
        if flag == 0:
            typeval += 16
        return GDT_DESCRIPTORS.get(typeval, 'UNKNOWN')

    @property
    def Base(self):
        if False:
            while True:
                i = 10
        'Get the base (start) of memory for this GDT'
        return self.BaseLow + (self.HighWord.Bits.BaseMid + (self.HighWord.Bits.BaseHi << 8) << 16)

    @property
    def Limit(self):
        if False:
            while True:
                i = 10
        'Get the limit (end) of memory for this GDT'
        limit = self.HighWord.Bits.LimitHi.v() << 16 | self.LimitLow.v()
        if self.HighWord.Bits.Granularity == 1:
            limit = (limit + 1) * 4096
            limit -= 1
        return limit

    @property
    def CallGate(self):
        if False:
            print('Hello World!')
        'Get the call gate address'
        return self.HighWord.v() & 4294901760 | self.LimitLow.v()

    @property
    def Present(self):
        if False:
            i = 10
            return i + 15
        'Returns True if the entry is present'
        return self.HighWord.Bits.Pres == 1

    @property
    def Granularity(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if page granularity is used. Otherwise\n        returns False indicating byte granularity is used.'
        return self.HighWord.Bits.Granularity == 1

    @property
    def Dpl(self):
        if False:
            while True:
                i = 10
        'Returns the descriptor privilege level'
        return self.HighWord.Bits.Dpl

class MalwareIDTGDTx86(obj.ProfileModification):
    before = ['WindowsObjectClasses', 'WindowsOverlay']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            return 10
        profile.object_classes.update({'_KIDTENTRY': _KIDTENTRY, '_KGDTENTRY': _KGDTENTRY})
        profile.merge_overlay({'_KPCR': [None, {'IDT': [None, ['pointer', ['array', 256, ['_KIDTENTRY']]]]}]})
        profile.merge_overlay({'_KPCR': [None, {'GDT': [None, ['pointer', ['array', 128, ['_KGDTENTRY']]]]}]})

class GDT(common.AbstractWindowsCommand):
    """Display Global Descriptor Table"""

    @staticmethod
    def is_valid_profile(profile):
        if False:
            for i in range(10):
                print('nop')
        return profile.metadata.get('os', 'unknown') == 'windows' and profile.metadata.get('memory_model', '32bit') == '32bit'

    def calculate(self):
        if False:
            print('Hello World!')
        addr_space = utils.load_as(self._config)
        if not self.is_valid_profile(addr_space.profile):
            debug.error('This command does not support the selected profile.')
        for kpcr in tasks.get_kdbg(addr_space).kpcrs():
            for (i, entry) in kpcr.gdt_entries():
                yield (i, entry)

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('CPU', int), ('Sel', Address), ('Base', Address), ('Limit', Address), ('Type', str), ('DPL', int), ('Gr', str), ('Pr', str)], self.generator(data))

    def generator(self, data):
        if False:
            i = 10
            return i + 15
        for (n, entry) in data:
            selector = n * 8
            if entry.Present:
                present = 'P'
            else:
                present = 'Np'
            if entry.Type == 'CallGate32':
                base = entry.CallGate
                limit = 0
                granularity = '-'
            else:
                base = entry.Base
                limit = entry.Limit
                if entry.Granularity:
                    granularity = 'Pg'
                else:
                    granularity = 'By'
            cpu_number = entry.obj_parent.obj_parent.ProcessorBlock.Number
            yield (0, [int(cpu_number), Address(selector), Address(base), Address(limit), str(entry.Type), int(entry.Dpl), str(granularity), str(present)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('CPU', '>6'), ('Sel', '[addr]'), ('Base', '[addrpad]'), ('Limit', '[addrpad]'), ('Type', '<14'), ('DPL', '>6'), ('Gr', '<4'), ('Pr', '<4')])
        for (n, entry) in data:
            selector = n * 8
            if entry.Present:
                present = 'P'
            else:
                present = 'Np'
            if entry.Type == 'CallGate32':
                base = entry.CallGate
                limit = 0
                granularity = '-'
            else:
                base = entry.Base
                limit = entry.Limit
                if entry.Granularity:
                    granularity = 'Pg'
                else:
                    granularity = 'By'
            cpu_number = entry.obj_parent.obj_parent.ProcessorBlock.Number
            self.table_row(outfd, cpu_number, selector, base, limit, entry.Type, entry.Dpl, granularity, present)

class IDT(common.AbstractWindowsCommand):
    """Display Interrupt Descriptor Table"""

    @staticmethod
    def is_valid_profile(profile):
        if False:
            i = 10
            return i + 15
        return profile.metadata.get('os', 'unknown') == 'windows' and profile.metadata.get('memory_model', '32bit') == '32bit'

    @staticmethod
    def get_section_name(mod, addr):
        if False:
            i = 10
            return i + 15
        'Get the name of the PE section containing \n        the specified address. \n\n        @param mod: an _LDR_DATA_TABLE_ENTRY \n        @param addr: virtual address to lookup \n        \n        @returns string PE section name\n        '
        try:
            dos_header = obj.Object('_IMAGE_DOS_HEADER', offset=mod.DllBase, vm=mod.obj_vm)
            nt_header = dos_header.get_nt_header()
        except (ValueError, exceptions.SanityCheckException):
            return ''
        for sec in nt_header.get_sections():
            if addr > mod.DllBase + sec.VirtualAddress and addr < sec.Misc.VirtualSize + (mod.DllBase + sec.VirtualAddress):
                return str(sec.Name or '')
        return ''

    def calculate(self):
        if False:
            return 10
        addr_space = utils.load_as(self._config)
        if not self.is_valid_profile(addr_space.profile):
            debug.error('This command does not support the selected profile.')
        mods = dict(((addr_space.address_mask(mod.DllBase), mod) for mod in modules.lsmod(addr_space)))
        mod_addrs = sorted(mods.keys())
        for kpcr in tasks.get_kdbg(addr_space).kpcrs():
            gdt = dict(((i * 8, sd) for (i, sd) in kpcr.gdt_entries()))
            for (i, entry) in kpcr.idt_entries():
                addr = entry.Address
                gdt_entry = gdt.get(entry.Selector.v())
                if gdt_entry != None and 'Code' in gdt_entry.Type:
                    addr += gdt_entry.Base
                module = tasks.find_module(mods, mod_addrs, addr_space.address_mask(addr))
                yield (i, entry, addr, module)

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('CPU', Hex), ('Index', Hex), ('Selector', Address), ('Value', Address), ('Module', str), ('Section', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for (n, entry, addr, module) in data:
            if addr == 0:
                module_name = 'NOT USED'
                sect_name = ''
            elif module:
                module_name = str(module.BaseDllName or '')
                sect_name = self.get_section_name(module, addr)
            else:
                module_name = 'UNKNOWN'
                sect_name = ''
            cpu_number = entry.obj_parent.obj_parent.ProcessorBlock.Number
            yield (0, [Hex(cpu_number), Hex(n), Address(entry.Selector), Address(addr), str(module_name), str(sect_name)])

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('CPU', '>6X'), ('Index', '>6X'), ('Selector', '[addr]'), ('Value', '[addrpad]'), ('Module', '20'), ('Section', '12')])
        for (n, entry, addr, module) in data:
            if addr == 0:
                module_name = 'NOT USED'
                sect_name = ''
            elif module:
                module_name = str(module.BaseDllName or '')
                sect_name = self.get_section_name(module, addr)
            else:
                module_name = 'UNKNOWN'
                sect_name = ''
            cpu_number = entry.obj_parent.obj_parent.ProcessorBlock.Number
            self.table_row(outfd, cpu_number, n, entry.Selector, addr, module_name, sect_name)
            if self._config.verbose:
                data = entry.obj_vm.zread(addr, 32)
                outfd.write('\n'.join(['{0:#x} {1:<16} {2}'.format(o, h, i) for (o, i, h) in malfind.Disassemble(data=data, start=addr, stoponret=True)]))
                outfd.write('\n')