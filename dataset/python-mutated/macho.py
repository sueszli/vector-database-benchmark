import volatility.obj as obj
macho_types = {'fat_header': [8, {'magic': [0, ['unsigned int']], 'nfat_arch': [4, ['unsigned int']]}], 'fat_arch': [20, {'cputype': [0, ['int']], 'cpusubtype': [4, ['int']], 'offset': [8, ['unsigned int']], 'size': [12, ['unsigned int']], 'align': [16, ['unsigned int']]}], 'macho64_header': [32, {'magic': [0, ['unsigned int']], 'cputype': [4, ['int']], 'cpusubtype': [8, ['int']], 'filetype': [12, ['unsigned int']], 'ncmds': [16, ['unsigned int']], 'sizeofcmds': [20, ['unsigned int']], 'flags': [24, ['unsigned int']], 'reserved': [28, ['unsigned int']]}], 'macho32_header': [28, {'magic': [0, ['unsigned int']], 'cputype': [4, ['int']], 'cpusubtype': [8, ['int']], 'filetype': [12, ['unsigned int']], 'ncmds': [16, ['unsigned int']], 'sizeofcmds': [20, ['unsigned int']], 'flags': [24, ['unsigned int']]}], 'macho32_symtab_command': [24, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']], 'symoff': [8, ['unsigned int']], 'nsyms': [12, ['unsigned int']], 'stroff': [16, ['unsigned int']], 'strsize': [20, ['unsigned int']]}], 'macho64_symtab_command': [24, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']], 'symoff': [8, ['unsigned int']], 'nsyms': [12, ['unsigned int']], 'stroff': [16, ['unsigned int']], 'strsize': [20, ['unsigned int']]}], 'macho64_dysymtab_command': [80, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']], 'ilocalsym': [8, ['unsigned int']], 'nlocalsym': [12, ['unsigned int']], 'iextdefsym': [16, ['unsigned int']], 'nextdefsym': [20, ['unsigned int']], 'iundefsym': [24, ['unsigned int']], 'nundefsym': [28, ['unsigned int']], 'tocoff': [32, ['unsigned int']], 'ntoc': [36, ['unsigned int']], 'modtaboff': [40, ['unsigned int']], 'nmodtab': [44, ['unsigned int']], 'extrefsymoff': [48, ['unsigned int']], 'nextrefsyms': [52, ['unsigned int']], 'indirectsymoff': [56, ['unsigned int']], 'nindirectsyms': [60, ['unsigned int']], 'extreloff': [64, ['unsigned int']], 'nextrel': [68, ['unsigned int']], 'locreloff': [72, ['unsigned int']], 'nlocrel': [76, ['unsigned int']]}], 'macho32_dysymtab_command': [80, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']], 'ilocalsym': [8, ['unsigned int']], 'nlocalsym': [12, ['unsigned int']], 'iextdefsym': [16, ['unsigned int']], 'nextdefsym': [20, ['unsigned int']], 'iundefsym': [24, ['unsigned int']], 'nundefsym': [28, ['unsigned int']], 'tocoff': [32, ['unsigned int']], 'ntoc': [36, ['unsigned int']], 'modtaboff': [40, ['unsigned int']], 'nmodtab': [44, ['unsigned int']], 'extrefsymoff': [48, ['unsigned int']], 'nextrefsyms': [52, ['unsigned int']], 'indirectsymoff': [56, ['unsigned int']], 'nindirectsyms': [60, ['unsigned int']], 'extreloff': [64, ['unsigned int']], 'nextrel': [68, ['unsigned int']], 'locreloff': [72, ['unsigned int']], 'nlocrel': [76, ['unsigned int']]}], 'macho32_load_command': [8, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']]}], 'macho64_load_command': [8, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']]}], 'macho32_dylib_command': [24, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']], 'name': [8, ['unsigned int']], 'timestamp': [12, ['unsigned int']], 'current_version': [16, ['unsigned int']], 'compatibility_version': [20, ['unsigned int']]}], 'macho64_dylib_command': [28, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']], 'name': [8, ['unsigned int']], 'timestamp': [16, ['unsigned int']], 'current_version': [20, ['unsigned int']], 'compatibility_version': [24, ['unsigned int']]}], 'macho32_segment_command': [56, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']], 'segname': [8, ['String', dict(length=16)]], 'vmaddr': [24, ['unsigned int']], 'vmsize': [28, ['unsigned int']], 'fileoff': [32, ['unsigned int']], 'filesize': [36, ['unsigned int']], 'maxprot': [40, ['int']], 'initprot': [44, ['int']], 'nsects': [48, ['unsigned int']], 'flags': [52, ['unsigned int']]}], 'macho64_segment_command': [72, {'cmd': [0, ['unsigned int']], 'cmdsize': [4, ['unsigned int']], 'segname': [8, ['String', dict(length=16)]], 'vmaddr': [24, ['unsigned long long']], 'vmsize': [32, ['unsigned long long']], 'fileoff': [40, ['unsigned long long']], 'filesize': [48, ['unsigned long long']], 'maxprot': [56, ['int']], 'initprot': [60, ['int']], 'nsects': [64, ['unsigned int']], 'flags': [68, ['unsigned int']]}], 'macho64_section': [80, {'sectname': [0, ['array', 16, ['char']]], 'segname': [16, ['array', 16, ['char']]], 'addr': [32, ['unsigned long long']], 'size': [40, ['unsigned long long']], 'offset': [48, ['unsigned int']], 'align': [52, ['unsigned int']], 'reloff': [56, ['unsigned int']], 'nreloc': [60, ['unsigned int']], 'flags': [64, ['unsigned int']], 'reserved1': [68, ['unsigned int']], 'reserved2': [72, ['unsigned int']], 'reserved3': [76, ['unsigned int']]}], 'macho32_section': [68, {'sectname': [0, ['array', 16, ['char']]], 'segname': [16, ['array', 16, ['char']]], 'addr': [32, ['unsigned int']], 'size': [36, ['unsigned int']], 'offset': [40, ['unsigned int']], 'align': [44, ['unsigned int']], 'reloff': [48, ['unsigned int']], 'nreloc': [52, ['unsigned int']], 'flags': [56, ['unsigned int']], 'reserved1': [60, ['unsigned int']], 'reserved2': [64, ['unsigned int']]}], 'macho32_nlist': [12, {'n_strx': [0, ['unsigned int']], 'n_type': [4, ['unsigned char']], 'n_sect': [5, ['unsigned char']], 'n_desc': [6, ['unsigned short']], 'n_value': [8, ['unsigned int']]}], 'macho64_nlist': [16, {'n_strx': [0, ['unsigned int']], 'n_type': [4, ['unsigned char']], 'n_sect': [5, ['unsigned char']], 'n_desc': [6, ['unsigned short']], 'n_value': [8, ['unsigned long long']]}]}

class macho(obj.CType):

    def __init__(self, is_header, name32, name64, theType, offset, vm, name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.name32 = name32
        self.name64 = name64
        self.macho_obj = None
        if is_header:
            self._init_cache(offset, vm)
        else:
            self.size_cache = -39
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

    def is_valid(self):
        if False:
            while True:
                i = 10
        return self.size_cache in [32, 64, -39]

    def _init_cache(self, offset, vm):
        if False:
            i = 10
            return i + 15
        self._set_size_cache(offset, vm)
        self._make_macho_obj(offset, vm)

    def _init_cache_from_parent(self):
        if False:
            while True:
                i = 10
        self.size_cache = self.obj_parent.size_cache
        self._make_macho_obj(self.obj_offset, self.obj_vm)

    def _make_macho_obj(self, offset, vm):
        if False:
            print('Hello World!')
        if self.size_cache == 32:
            self.macho_obj = obj.Object(self.name32, offset=offset, vm=vm, parent=self)
        elif self.size_cache == 64:
            self.macho_obj = obj.Object(self.name64, offset=offset, vm=vm, parent=self)
        else:
            self.macho_obj = None

    def _set_size_cache(self, offset, vm):
        if False:
            return 10
        ei_class = obj.Object('unsigned int', offset=offset + 4, vm=vm)
        if ei_class == 7:
            self.size_cache = 32
        elif ei_class == 16777223:
            self.size_cache = 64
        else:
            self.size_cache = -42

    def _get_typename(self, typename):
        if False:
            i = 10
            return i + 15
        if self.size_cache == -39:
            self._init_cache_from_parent()
        if self.size_cache == 32:
            typename = 'macho32_' + typename
        else:
            typename = 'macho64_' + typename
        return typename

    def get_bits(self):
        if False:
            i = 10
            return i + 15
        return self.size_cache

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        if self.size_cache == -39:
            self._init_cache_from_parent()
        return self.macho_obj.__getattr__(attr)

class macho_header(macho):
    """An macho header"""

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            while True:
                i = 10
        self.cached_strtab = None
        self.cached_symtab = None
        self.cached_dysymtab = None
        self.cached_syms = None
        self.load_diff = 0
        self.link_edit_bias = 0
        macho.__init__(self, 1, 'macho32_header', 'macho64_header', theType, offset, vm, name, **kwargs)
        if self.macho_obj:
            self.calc_load_diff()
            self._calc_linkedit_bias()
            self._build_symbol_caches()

    def is_valid(self):
        if False:
            i = 10
            return i + 15
        return self.macho_obj != None

    def _calc_linkedit_bias(self):
        if False:
            for i in range(10):
                print('nop')
        for s in self.segments():
            if str(s.segname) == '__LINKEDIT':
                self.link_edit_bias = s.vmaddr - s.fileoff
                break

    def calc_load_diff(self):
        if False:
            print('Hello World!')
        seg = None
        for s in self.segments():
            if str(s.segname) == '__PAGEZERO':
                continue
            seg = s
            break
        if seg and seg.vmaddr != self.obj_offset:
            self.load_diff = self.obj_offset - seg.vmaddr

    def load_commands(self):
        if False:
            print('Hello World!')
        rtname = self._get_typename('load_command')
        rtsize = self.obj_vm.profile.get_obj_size(rtname)
        tname = 'macho_load_command'
        if self.macho_obj == None:
            return
        hdr_size = self.macho_obj.size()
        if hdr_size == 0 or hdr_size > 100000000:
            return
        arr_start = self.obj_offset + hdr_size
        offset = 0
        if self.ncmds > 1024:
            return
        for i in range(self.ncmds):
            cmd = obj.Object(tname, offset=arr_start + offset, vm=self.obj_vm, parent=self)
            yield cmd
            offset = offset + cmd.cmdsize

    def load_commands_of_type(self, cmd_type):
        if False:
            print('Hello World!')
        cmds = []
        for cmd in self.load_commands():
            if cmd_type == cmd.cmd.v():
                cmds.append(cmd)
        return cmds

    def load_command_of_type(self, cmd_type):
        if False:
            while True:
                i = 10
        ret = None
        cmds = self.load_commands_of_type(cmd_type)
        if cmds and len(cmds) > 1:
            debug.error('load_command_of_type: Multiple commands of type %d found!' % cmd_type)
        elif cmds:
            ret = cmds[0]
        return ret

    def get_indirect_syms(self):
        if False:
            return 10
        syms = []
        tname = self._get_typename('nlist')
        obj_size = self.obj_vm.profile.get_obj_size(tname)
        indirect_table_addr = self.link_edit_bias + self.cached_dysymtab.indirectsymoff
        if not self.obj_vm.is_valid_address(indirect_table_addr):
            return syms
        cnt = self.cached_dysymtab.nindirectsyms
        if cnt > 100000:
            cnt = 1024
        symtab_idxs = obj.Object(theType='Array', targetType='unsigned int', count=cnt, offset=indirect_table_addr, vm=self.obj_vm, parent=self)
        for idx in symtab_idxs:
            sym_addr = self.cached_symtab + idx * obj_size
            sym = obj.Object('macho_nlist', offset=sym_addr, vm=self.obj_vm, parent=self)
            if sym.is_valid():
                syms.append(sym)
        return syms

    def _get_symtab_syms(self, sym_command, symtab_addr):
        if False:
            while True:
                i = 10
        syms = []
        tname = self._get_typename('nlist')
        obj_size = self.obj_vm.profile.get_obj_size(tname)
        if not self.obj_vm.is_valid_address(symtab_addr):
            return syms
        num_syms = sym_command.nsyms
        if num_syms > 2000:
            return syms
        for i in range(num_syms):
            sym_addr = symtab_addr + i * obj_size
            sym = obj.Object('macho_nlist', offset=sym_addr, vm=self.obj_vm, parent=self)
            if sym.is_valid():
                syms.append(sym)
        return syms

    def _build_symbol_caches(self):
        if False:
            print('Hello World!')
        symtab_cmd = self.load_command_of_type(2)
        symtab_struct_name = self._get_typename('symtab_command')
        if symtab_cmd == None:
            return
        symtab_command = symtab_cmd.cast(symtab_struct_name)
        str_strtab = self.link_edit_bias + symtab_command.stroff
        symtab_addr = self.link_edit_bias + symtab_command.symoff
        self.cached_syms = self._get_symtab_syms(symtab_command, symtab_addr)
        dysymtab_cmd = self.load_command_of_type(11)
        if dysymtab_cmd == None:
            return
        dystruct_name = self._get_typename('dysymtab_command')
        dysymtab_command = dysymtab_cmd.cast(dystruct_name)
        self.cached_strtab = str_strtab
        self.cached_symtab = symtab_addr
        self.cached_dysymtab = dysymtab_command
        self.cached_syms = self.cached_syms + self.get_indirect_syms()

    def symbols(self):
        if False:
            i = 10
            return i + 15
        if self.cached_syms == None:
            ret = []
        else:
            ret = self.cached_syms
        return ret

    def symbol_name(self, sym):
        if False:
            while True:
                i = 10
        if self.cached_symtab == None:
            return ''
        name_addr = self.cached_strtab + sym.n_strx
        name = self.obj_vm.read(name_addr, 64)
        if name:
            idx = name.find('\x00')
            if idx != -1:
                name = name[:idx]
        return name

    def address_for_symbol(self, sym_name):
        if False:
            print('Hello World!')
        ret = None
        for sym in self.symbols():
            if self.symbol_name(sym) == sym_name:
                ret = sym.n_value.v()
                break
        return ret

    def needed_libraries(self):
        if False:
            i = 10
            return i + 15
        for cmd in self.load_commands_of_type(12):
            tname = self._get_typename('dylib_command')
            dylib_command = cmd.cast(tname)
            name_addr = cmd.obj_offset + dylib_command.name
            dylib_name = self.obj_vm.read(name_addr, 256)
            if dylib_name:
                idx = dylib_name.find('\x00')
                if idx != -1:
                    dylib_name = dylib_name[:idx]
                yield dylib_name

    def imports(self):
        if False:
            for i in range(10):
                print('nop')
        sect_type = self._get_typename('section')
        sect_size = self.obj_vm.profile.get_obj_size(sect_type)
        if self.get_bits() == 32:
            idx_type = 'unsigned int'
        else:
            idx_type = 'unsigned long long'
        num_idxs = sect_size / (self.get_bits() / 8)
        for seg in self.segments():
            if str(seg.segname) == '__DATA':
                for sect in self.sections_for_segment(seg):
                    if str(sect.sectname) == '__la_symbol_ptr':
                        sym_ptr_arr = obj.Object(theType='Array', targetType=idx_type, count=num_idxs, offset=self.obj_offset + sect.offset, vm=self.obj_vm)
                        isyms = self.get_indirect_syms()
                        num_isyms = len(isyms)
                        for (i, sym_ptr) in enumerate(sym_ptr_arr):
                            idx = sect.reserved1 + i
                            if idx >= num_isyms:
                                continue
                            sym = isyms[idx]
                            name = self.symbol_name(sym)
                            yield (name, sym_ptr)

    def segments(self):
        if False:
            for i in range(10):
                print('nop')
        LC_SEGMENT = 1
        LC_SEGMENT_64 = 25
        if self.size_cache == 32:
            seg_type = LC_SEGMENT
        else:
            seg_type = LC_SEGMENT_64
        load_commands = self.load_commands_of_type(seg_type)
        for load_command in load_commands:
            segment = obj.Object('macho_segment_command', offset=load_command.obj_offset, vm=self.obj_vm, parent=self)
            yield segment

    def get_segment(self, segment_name):
        if False:
            for i in range(10):
                print('nop')
        ret = None
        for segment in self.get_segments():
            if str(segment.segname) == segment_name:
                ret = segment
                break
        return ret

    def sections_for_segment(self, segment):
        if False:
            for i in range(10):
                print('nop')
        sect_struct = self._get_typename('section')
        sect_size = self.obj_vm.profile.get_obj_size(sect_struct)
        seg_struct = self._get_typename('segment_command')
        seg_size = self.obj_vm.profile.get_obj_size(seg_struct)
        cnt = segment.nsects
        if cnt > 1024:
            cnt = 1024
        for i in range(cnt):
            sect_addr = segment.obj_offset + seg_size + i * sect_size
            sect = obj.Object('macho_section', offset=sect_addr, vm=self.obj_vm, parent=self)
            yield sect

class macho32_header(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            i = 10
            return i + 15
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho64_header(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            i = 10
            return i + 15
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho_section(macho):
    """ An macho section header """

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            print('Hello World!')
        macho.__init__(self, 0, 'macho32_section', 'macho64_section', theType, offset, vm, name, **kwargs)

class macho32_section(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            return 10
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho64_section(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            return 10
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho_segment_command(macho):
    """ A macho segment command """

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            return 10
        macho.__init__(self, 0, 'macho32_segment_command', 'macho64_segment_command', theType, offset, vm, name, **kwargs)

    @property
    def vmaddr(self):
        if False:
            for i in range(10):
                print('nop')
        ret = self.__getattr__('vmaddr')
        if self.obj_parent.load_diff:
            ret = ret + self.obj_parent.load_diff
        if self.obj_parent.filetype == 2:
            ret = ret + self.obj_parent.obj_offset
        return ret

class macho32_segment_command(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            while True:
                i = 10
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho64_segment_command(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            while True:
                i = 10
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho_load_command(macho):
    """ A macho load command """

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            while True:
                i = 10
        macho.__init__(self, 0, 'macho32_load_command', 'macho64_load_command', theType, offset, vm, name, **kwargs)

    @property
    def cmd_type(self):
        if False:
            print('Hello World!')
        cmd_types = {1: 'LC_SEGMENT', 2: 'LC_SYMTAB', 25: 'LC_SEGMENT_64', 12: 'LC_LOAD_DYLIB'}
        cmd = self.cmd.v()
        if cmd in cmd_types:
            ret = cmd_types[cmd]
        else:
            ret = ''
        return ret

class macho32_load_command(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            print('Hello World!')
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho64_load_command(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            return 10
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho_symtab_command(macho):
    """ A macho symtab command """

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            while True:
                i = 10
        macho.__init__(self, 0, 'macho32_symtab_command', 'macho64_symtab_command', theType, offset, vm, name, **kwargs)

class macho32_symtab_command(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            print('Hello World!')
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho64_symtab_command(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            while True:
                i = 10
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho_dysymtab_command(macho):
    """ A macho symtab command """

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        macho.__init__(self, 0, 'macho32_dysymtab_command', 'macho64_dysymtab_command', theType, offset, vm, name, **kwargs)

class macho32_dysymtab_command(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            while True:
                i = 10
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho64_dysymtab_command(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            while True:
                i = 10
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho_nlist(macho):
    """ A macho nlist """

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        macho.__init__(self, 0, 'macho32_nlist', 'macho64_nlist', theType, offset, vm, name, **kwargs)

class macho32_nlist(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            i = 10
            return i + 15
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class macho64_nlist(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            print('Hello World!')
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)

class MachoTypes(obj.ProfileModification):

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update(macho_types)

class MachoModification(obj.ProfileModification):

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.object_classes.update({'macho': macho, 'macho_header': macho_header, 'macho32_header': macho32_header, 'macho64_header': macho64_header, 'macho_section': macho_section, 'macho32_section': macho32_section, 'macho64_section': macho64_section, 'macho_segment_command': macho_segment_command, 'macho32_segment_command': macho32_segment_command, 'macho64_segment_command': macho64_segment_command, 'macho_load_command': macho_load_command, 'macho32_load_command': macho32_load_command, 'macho64_load_command': macho64_load_command, 'macho_symtab_command': macho_symtab_command, 'macho32_symtab_command': macho32_symtab_command, 'macho64_symtab_command': macho64_symtab_command, 'macho_dysymtab_command': macho_dysymtab_command, 'macho32_dysymtab_command': macho32_dysymtab_command, 'macho64_dysymtab_command': macho64_dysymtab_command, 'macho_nlist': macho_nlist, 'macho32_nlist': macho32_nlist, 'macho64_nlist': macho64_nlist})
macho_overlay = {'macho32_segment_command': [None, {'segname': [None, ['String', dict(length=16)]]}], 'macho64_segment_command': [None, {'segname': [None, ['String', dict(length=16)]]}], 'macho32_section': [None, {'sectname': [None, ['String', dict(length=16)]]}], 'macho64_section': [None, {'sectname': [None, ['String', dict(length=16)]]}]}

class MachoOverlay(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'mac'}
    before = ['BasicObjectClasses']

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.merge_overlay(macho_overlay)