"""
Link .o files to .mpy
"""
import sys, os, struct, re
from elftools.elf import elffile
sys.path.append(os.path.dirname(__file__) + '/../py')
import makeqstrdata as qstrutil
MPY_VERSION = 6
MPY_SUB_VERSION = 2
MP_CODE_BYTECODE = 2
MP_CODE_NATIVE_VIPER = 4
MP_NATIVE_ARCH_X86 = 1
MP_NATIVE_ARCH_X64 = 2
MP_NATIVE_ARCH_ARMV6M = 4
MP_NATIVE_ARCH_ARMV7M = 5
MP_NATIVE_ARCH_ARMV7EMSP = 7
MP_NATIVE_ARCH_ARMV7EMDP = 8
MP_NATIVE_ARCH_XTENSA = 9
MP_NATIVE_ARCH_XTENSAWIN = 10
MP_PERSISTENT_OBJ_STR = 5
MP_SCOPE_FLAG_VIPERRELOC = 16
MP_SCOPE_FLAG_VIPERRODATA = 32
MP_SCOPE_FLAG_VIPERBSS = 64
MP_SMALL_INT_BITS = 31
R_386_32 = 1
R_X86_64_64 = 1
R_XTENSA_32 = 1
R_386_PC32 = 2
R_X86_64_PC32 = 2
R_ARM_ABS32 = 2
R_386_GOT32 = 3
R_ARM_REL32 = 3
R_386_PLT32 = 4
R_X86_64_PLT32 = 4
R_XTENSA_PLT = 6
R_386_GOTOFF = 9
R_386_GOTPC = 10
R_ARM_THM_CALL = 10
R_XTENSA_DIFF32 = 19
R_XTENSA_SLOT0_OP = 20
R_ARM_BASE_PREL = 25
R_ARM_GOT_BREL = 26
R_ARM_THM_JUMP24 = 30
R_X86_64_GOTPCREL = 9
R_X86_64_REX_GOTPCRELX = 42
R_386_GOT32X = 43
R_XTENSA_PDIFF32 = 59

def asm_jump_x86(entry):
    if False:
        return 10
    return struct.pack('<BI', 233, entry - 5)

def asm_jump_thumb(entry):
    if False:
        while True:
            i = 10
    b_off = entry - 4
    assert b_off >> 11 == 0 or b_off >> 11 == -1, b_off
    return struct.pack('<H', 57344 | b_off >> 1 & 2047)

def asm_jump_thumb2(entry):
    if False:
        print('Hello World!')
    b_off = entry - 4
    if b_off >> 11 == 0 or b_off >> 11 == -1:
        b0 = 57344 | b_off >> 1 & 2047
        b1 = 0
    else:
        b0 = 61440 | b_off >> 12 & 2047
        b1 = 47104 | b_off >> 1 & 2047
    return struct.pack('<HH', b0, b1)

def asm_jump_xtensa(entry):
    if False:
        i = 10
        return i + 15
    jump_offset = entry - 4
    jump_op = jump_offset << 6 | 6
    return struct.pack('<BH', jump_op & 255, jump_op >> 8)

class ArchData:

    def __init__(self, name, mpy_feature, word_size, arch_got, asm_jump, *, separate_rodata=False):
        if False:
            return 10
        self.name = name
        self.mpy_feature = mpy_feature
        self.qstr_entry_size = 2
        self.word_size = word_size
        self.arch_got = arch_got
        self.asm_jump = asm_jump
        self.separate_rodata = separate_rodata
ARCH_DATA = {'x86': ArchData('EM_386', MP_NATIVE_ARCH_X86 << 2, 4, (R_386_PC32, R_386_GOT32, R_386_GOT32X), asm_jump_x86), 'x64': ArchData('EM_X86_64', MP_NATIVE_ARCH_X64 << 2, 8, (R_X86_64_GOTPCREL, R_X86_64_REX_GOTPCRELX), asm_jump_x86), 'armv6m': ArchData('EM_ARM', MP_NATIVE_ARCH_ARMV6M << 2, 4, (R_ARM_GOT_BREL,), asm_jump_thumb), 'armv7m': ArchData('EM_ARM', MP_NATIVE_ARCH_ARMV7M << 2, 4, (R_ARM_GOT_BREL,), asm_jump_thumb2), 'armv7emsp': ArchData('EM_ARM', MP_NATIVE_ARCH_ARMV7EMSP << 2, 4, (R_ARM_GOT_BREL,), asm_jump_thumb2), 'armv7emdp': ArchData('EM_ARM', MP_NATIVE_ARCH_ARMV7EMDP << 2, 4, (R_ARM_GOT_BREL,), asm_jump_thumb2), 'xtensa': ArchData('EM_XTENSA', MP_NATIVE_ARCH_XTENSA << 2, 4, (R_XTENSA_32, R_XTENSA_PLT), asm_jump_xtensa), 'xtensawin': ArchData('EM_XTENSA', MP_NATIVE_ARCH_XTENSAWIN << 2, 4, (R_XTENSA_32, R_XTENSA_PLT), asm_jump_xtensa, separate_rodata=True)}

def align_to(value, align):
    if False:
        return 10
    return value + align - 1 & ~(align - 1)

def unpack_u24le(data, offset):
    if False:
        return 10
    return data[offset] | data[offset + 1] << 8 | data[offset + 2] << 16

def pack_u24le(data, offset, value):
    if False:
        return 10
    data[offset] = value & 255
    data[offset + 1] = value >> 8 & 255
    data[offset + 2] = value >> 16 & 255

def xxd(text):
    if False:
        return 10
    for i in range(0, len(text), 16):
        print('{:08x}:'.format(i), end='')
        for j in range(4):
            off = i + j * 4
            if off < len(text):
                d = int.from_bytes(text[off:off + 4], 'little')
                print(' {:08x}'.format(d), end='')
        print()
LOG_LEVEL_1 = 1
LOG_LEVEL_2 = 2
LOG_LEVEL_3 = 3
log_level = LOG_LEVEL_1

def log(level, msg):
    if False:
        return 10
    if level <= log_level:
        print(msg)

def extract_qstrs(source_files):
    if False:
        print('Hello World!')

    def read_qstrs(f):
        if False:
            while True:
                i = 10
        with open(f) as f:
            vals = set()
            for line in f:
                for m in re.finditer('MP_QSTR_[A-Za-z0-9_]*', line):
                    vals.add(m.group())
            return vals
    static_qstrs = ['MP_QSTR_' + qstrutil.qstr_escape(q) for q in qstrutil.static_qstr_list]
    qstr_vals = set()
    for f in source_files:
        vals = read_qstrs(f)
        qstr_vals.update(vals)
    qstr_vals.difference_update(static_qstrs)
    return (static_qstrs, qstr_vals)

class LinkError(Exception):
    pass

class Section:

    def __init__(self, name, data, alignment, filename=None):
        if False:
            for i in range(10):
                print('nop')
        self.filename = filename
        self.name = name
        self.data = data
        self.alignment = alignment
        self.addr = 0
        self.reloc = []

    @staticmethod
    def from_elfsec(elfsec, filename):
        if False:
            print('Hello World!')
        assert elfsec.header.sh_addr == 0
        return Section(elfsec.name, elfsec.data(), elfsec.data_alignment, filename)

class GOTEntry:

    def __init__(self, name, sym, link_addr=0):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.sym = sym
        self.offset = None
        self.link_addr = link_addr

    def isexternal(self):
        if False:
            print('Hello World!')
        return self.sec_name.startswith('.external')

    def istext(self):
        if False:
            return 10
        return self.sec_name.startswith('.text')

    def isrodata(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sec_name.startswith(('.rodata', '.data.rel.ro'))

    def isbss(self):
        if False:
            while True:
                i = 10
        return self.sec_name.startswith('.bss')

class LiteralEntry:

    def __init__(self, value, offset):
        if False:
            print('Hello World!')
        self.value = value
        self.offset = offset

class LinkEnv:

    def __init__(self, arch):
        if False:
            while True:
                i = 10
        self.arch = ARCH_DATA[arch]
        self.sections = []
        self.literal_sections = []
        self.known_syms = {}
        self.unresolved_syms = []
        self.mpy_relocs = []

    def check_arch(self, arch_name):
        if False:
            return 10
        if arch_name != self.arch.name:
            raise LinkError('incompatible arch')

    def print_sections(self):
        if False:
            i = 10
            return i + 15
        log(LOG_LEVEL_2, 'sections:')
        for sec in self.sections:
            log(LOG_LEVEL_2, '  {:08x} {} size={}'.format(sec.addr, sec.name, len(sec.data)))

    def find_addr(self, name):
        if False:
            i = 10
            return i + 15
        if name in self.known_syms:
            s = self.known_syms[name]
            return s.section.addr + s['st_value']
        raise LinkError('unknown symbol: {}'.format(name))

def build_got_generic(env):
    if False:
        return 10
    env.got_entries = {}
    for sec in env.sections:
        for r in sec.reloc:
            s = r.sym
            if not (s.entry['st_info']['bind'] == 'STB_GLOBAL' and r['r_info_type'] in env.arch.arch_got):
                continue
            s_type = s.entry['st_info']['type']
            assert s_type in ('STT_NOTYPE', 'STT_FUNC', 'STT_OBJECT'), s_type
            assert s.name
            if s.name in env.got_entries:
                continue
            env.got_entries[s.name] = GOTEntry(s.name, s)

def build_got_xtensa(env):
    if False:
        print('Hello World!')
    env.got_entries = {}
    env.lit_entries = {}
    env.xt_literals = {}
    for sec in env.literal_sections:
        assert len(sec.data) % env.arch.word_size == 0
        for r in sec.reloc:
            s = r.sym
            s_type = s.entry['st_info']['type']
            assert s_type in ('STT_NOTYPE', 'STT_FUNC', 'STT_OBJECT', 'STT_SECTION'), s_type
            assert r['r_info_type'] in env.arch.arch_got
            assert r['r_offset'] % env.arch.word_size == 0
            existing = struct.unpack_from('<I', sec.data, r['r_offset'])[0]
            if s_type == 'STT_SECTION':
                assert r['r_addend'] == 0
                name = '{}+0x{:x}'.format(s.section.name, existing)
            else:
                assert existing == 0
                name = s.name
                if r['r_addend'] != 0:
                    name = '{}+0x{:x}'.format(name, r['r_addend'])
            idx = '{}+0x{:x}'.format(sec.filename, r['r_offset'])
            env.xt_literals[idx] = name
            if name in env.got_entries:
                continue
            env.got_entries[name] = GOTEntry(name, s, existing)
        for i in range(0, len(sec.data), env.arch.word_size):
            idx = '{}+0x{:x}'.format(sec.filename, i)
            if idx not in env.xt_literals:
                value = struct.unpack_from('<I', sec.data, i)[0]
                env.xt_literals[idx] = value
                if value in env.lit_entries:
                    continue
                env.lit_entries[value] = LiteralEntry(value, len(env.lit_entries) * env.arch.word_size)

def populate_got(env):
    if False:
        i = 10
        return i + 15
    for got_entry in env.got_entries.values():
        sym = got_entry.sym
        if hasattr(sym, 'resolved'):
            sym = sym.resolved
        sec = sym.section
        addr = sym['st_value']
        got_entry.sec_name = sec.name
        got_entry.link_addr += sec.addr + addr
    got_list = sorted(env.got_entries.values(), key=lambda g: g.isexternal() + 2 * g.istext() + 3 * g.isrodata() + 4 * g.isbss())
    offset = 0
    for got_entry in got_list:
        got_entry.offset = offset
        offset += env.arch.word_size
        o = env.got_section.addr + got_entry.offset
        env.full_text[o:o + env.arch.word_size] = got_entry.link_addr.to_bytes(env.arch.word_size, 'little')
    for got_entry in got_list:
        if got_entry.name in ('mp_native_qstr_table', 'mp_native_obj_table', 'mp_fun_table'):
            dest = got_entry.name
        elif got_entry.name.startswith('mp_fun_table+0x'):
            dest = int(got_entry.name.split('+')[1], 16) // env.arch.word_size
        elif got_entry.sec_name == '.external.mp_fun_table':
            dest = got_entry.sym.mp_fun_table_offset
        elif got_entry.sec_name.startswith('.text'):
            dest = '.text'
        elif got_entry.sec_name.startswith('.rodata'):
            dest = '.rodata'
        elif got_entry.sec_name.startswith('.data.rel.ro'):
            dest = '.data.rel.ro'
        elif got_entry.sec_name.startswith('.bss'):
            dest = '.bss'
        else:
            assert 0, (got_entry.name, got_entry.sec_name)
        env.mpy_relocs.append(('.text', env.got_section.addr + got_entry.offset, dest))
    log(LOG_LEVEL_2, 'GOT: {:08x}'.format(env.got_section.addr))
    for g in got_list:
        log(LOG_LEVEL_2, '  {:08x} {} -> {}+{:08x}'.format(g.offset, g.name, g.sec_name, g.link_addr))

def populate_lit(env):
    if False:
        return 10
    log(LOG_LEVEL_2, 'LIT: {:08x}'.format(env.lit_section.addr))
    for lit_entry in env.lit_entries.values():
        value = lit_entry.value
        log(LOG_LEVEL_2, '  {:08x} = {:08x}'.format(lit_entry.offset, value))
        o = env.lit_section.addr + lit_entry.offset
        env.full_text[o:o + env.arch.word_size] = value.to_bytes(env.arch.word_size, 'little')

def do_relocation_text(env, text_addr, r):
    if False:
        print('Hello World!')
    s = r.sym
    s_bind = s.entry['st_info']['bind']
    s_type = s.entry['st_info']['type']
    r_offset = r['r_offset'] + text_addr
    r_info_type = r['r_info_type']
    try:
        r_addend = r['r_addend']
    except KeyError:
        r_addend = 0
    reloc_type = 'le32'
    log_name = None
    if env.arch.name == 'EM_386' and r_info_type in (R_386_PC32, R_386_PLT32) or (env.arch.name == 'EM_X86_64' and r_info_type in (R_X86_64_PC32, R_X86_64_PLT32)) or (env.arch.name == 'EM_ARM' and r_info_type in (R_ARM_REL32, R_ARM_THM_CALL, R_ARM_THM_JUMP24)) or (s_bind == 'STB_LOCAL' and env.arch.name == 'EM_XTENSA' and (r_info_type == R_XTENSA_32)):
        if hasattr(s, 'resolved'):
            s = s.resolved
        sec = s.section
        if env.arch.separate_rodata and sec.name.startswith('.rodata'):
            raise LinkError('fixed relocation to rodata with rodata referenced via GOT')
        if sec.name.startswith('.bss'):
            raise LinkError("{}: fixed relocation to bss (bss variables can't be static)".format(s.filename))
        if sec.name.startswith('.external'):
            raise LinkError('{}: fixed relocation to external symbol: {}'.format(s.filename, s.name))
        addr = sec.addr + s['st_value']
        reloc = addr - r_offset + r_addend
        if r_info_type in (R_ARM_THM_CALL, R_ARM_THM_JUMP24):
            reloc_type = 'thumb_b'
    elif env.arch.name == 'EM_386' and r_info_type == R_386_GOTPC or (env.arch.name == 'EM_ARM' and r_info_type == R_ARM_BASE_PREL):
        assert s.name == '_GLOBAL_OFFSET_TABLE_'
        addr = env.got_section.addr
        reloc = addr - r_offset + r_addend
    elif env.arch.name == 'EM_386' and r_info_type in (R_386_GOT32, R_386_GOT32X) or (env.arch.name == 'EM_ARM' and r_info_type == R_ARM_GOT_BREL):
        reloc = addr = env.got_entries[s.name].offset
    elif env.arch.name == 'EM_X86_64' and r_info_type in (R_X86_64_GOTPCREL, R_X86_64_REX_GOTPCRELX):
        got_entry = env.got_entries[s.name]
        addr = env.got_section.addr + got_entry.offset
        reloc = addr - r_offset + r_addend
    elif env.arch.name == 'EM_386' and r_info_type == R_386_GOTOFF:
        addr = s.section.addr + s['st_value']
        reloc = addr - env.got_section.addr + r_addend
    elif env.arch.name == 'EM_XTENSA' and r_info_type == R_XTENSA_SLOT0_OP:
        sec = s.section
        if sec.name.startswith('.text'):
            return
        assert sec.name.startswith('.literal'), sec.name
        lit_idx = '{}+0x{:x}'.format(sec.filename, r_addend)
        lit_ptr = env.xt_literals[lit_idx]
        if isinstance(lit_ptr, str):
            addr = env.got_section.addr + env.got_entries[lit_ptr].offset
            log_name = 'GOT {}'.format(lit_ptr)
        else:
            addr = env.lit_section.addr + env.lit_entries[lit_ptr].offset
            log_name = 'LIT'
        reloc = addr - r_offset
        reloc_type = 'xtensa_l32r'
    elif env.arch.name == 'EM_XTENSA' and r_info_type in (R_XTENSA_DIFF32, R_XTENSA_PDIFF32):
        if s.section.name.startswith('.text'):
            return
        assert 0
    else:
        assert 0, r_info_type
    if reloc_type == 'le32':
        (existing,) = struct.unpack_from('<I', env.full_text, r_offset)
        struct.pack_into('<I', env.full_text, r_offset, existing + reloc & 4294967295)
    elif reloc_type == 'thumb_b':
        (b_h, b_l) = struct.unpack_from('<HH', env.full_text, r_offset)
        existing = (b_h & 2047) << 12 | (b_l & 2047) << 1
        if existing >= 4194304:
            existing -= 8388608
        new = existing + reloc
        b_h = b_h & 63488 | new >> 12 & 2047
        b_l = b_l & 63488 | new >> 1 & 2047
        struct.pack_into('<HH', env.full_text, r_offset, b_h, b_l)
    elif reloc_type == 'xtensa_l32r':
        l32r = unpack_u24le(env.full_text, r_offset)
        assert l32r & 15 == 1
        l32r_imm16 = l32r >> 8
        l32r_imm16 = l32r_imm16 + reloc >> 2 & 65535
        l32r = l32r & 255 | l32r_imm16 << 8
        pack_u24le(env.full_text, r_offset, l32r)
    else:
        assert 0, reloc_type
    if log_name is None:
        if s_type == 'STT_SECTION':
            log_name = s.section.name
        else:
            log_name = s.name
    log(LOG_LEVEL_3, '  {:08x} {} -> {:08x}'.format(r_offset, log_name, addr))

def do_relocation_data(env, text_addr, r):
    if False:
        print('Hello World!')
    s = r.sym
    s_type = s.entry['st_info']['type']
    r_offset = r['r_offset'] + text_addr
    r_info_type = r['r_info_type']
    try:
        r_addend = r['r_addend']
    except KeyError:
        r_addend = 0
    if env.arch.name == 'EM_386' and r_info_type == R_386_32 or (env.arch.name == 'EM_X86_64' and r_info_type == R_X86_64_64) or (env.arch.name == 'EM_ARM' and r_info_type == R_ARM_ABS32) or (env.arch.name == 'EM_XTENSA' and r_info_type == R_XTENSA_32):
        if env.arch.word_size == 4:
            struct_type = '<I'
        elif env.arch.word_size == 8:
            struct_type = '<Q'
        sec = s.section
        assert r_offset % env.arch.word_size == 0
        addr = sec.addr + s['st_value'] + r_addend
        if s_type == 'STT_SECTION':
            log_name = sec.name
        else:
            log_name = s.name
        log(LOG_LEVEL_3, '  {:08x} -> {} {:08x}'.format(r_offset, log_name, addr))
        if env.arch.separate_rodata:
            data = env.full_rodata
        else:
            data = env.full_text
        (existing,) = struct.unpack_from(struct_type, data, r_offset)
        if sec.name.startswith(('.text', '.rodata', '.data.rel.ro', '.bss')):
            struct.pack_into(struct_type, data, r_offset, existing + addr)
            kind = sec.name
        elif sec.name == '.external.mp_fun_table':
            assert addr == 0
            kind = s.mp_fun_table_offset
        else:
            assert 0, sec.name
        if env.arch.separate_rodata:
            base = '.rodata'
        else:
            base = '.text'
        env.mpy_relocs.append((base, r_offset, kind))
    else:
        assert 0, r_info_type

def load_object_file(env, felf):
    if False:
        i = 10
        return i + 15
    with open(felf, 'rb') as f:
        elf = elffile.ELFFile(f)
        env.check_arch(elf['e_machine'])
        symtab = list(elf.get_section_by_name('.symtab').iter_symbols())
        sections_shndx = {}
        for (idx, s) in enumerate(elf.iter_sections()):
            if s.header.sh_type in ('SHT_PROGBITS', 'SHT_NOBITS'):
                if s.data_size == 0:
                    pass
                elif s.name.startswith(('.literal', '.text', '.rodata', '.data.rel.ro', '.bss')):
                    sec = Section.from_elfsec(s, felf)
                    sections_shndx[idx] = sec
                    if s.name.startswith('.literal'):
                        env.literal_sections.append(sec)
                    else:
                        env.sections.append(sec)
                elif s.name.startswith('.data'):
                    raise LinkError('{}: {} non-empty'.format(felf, s.name))
                else:
                    pass
            elif s.header.sh_type in ('SHT_REL', 'SHT_RELA'):
                shndx = s.header.sh_info
                if shndx in sections_shndx:
                    sec = sections_shndx[shndx]
                    sec.reloc_name = s.name
                    sec.reloc = list(s.iter_relocations())
                    for r in sec.reloc:
                        r.sym = symtab[r['r_info_sym']]
        for sym in symtab:
            sym.filename = felf
            shndx = sym.entry['st_shndx']
            if shndx in sections_shndx:
                sym.section = sections_shndx[shndx]
                if sym['st_info']['bind'] == 'STB_GLOBAL':
                    if sym.name in env.known_syms and (not sym.name.startswith('__x86.get_pc_thunk.')):
                        raise LinkError('duplicate symbol: {}'.format(sym.name))
                    env.known_syms[sym.name] = sym
            elif sym.entry['st_shndx'] == 'SHN_UNDEF' and sym['st_info']['bind'] == 'STB_GLOBAL':
                env.unresolved_syms.append(sym)

def link_objects(env, native_qstr_vals_len):
    if False:
        print('Hello World!')
    if env.arch.name == 'EM_XTENSA':
        build_got_xtensa(env)
    else:
        build_got_generic(env)
    got_size = len(env.got_entries) * env.arch.word_size
    env.got_section = Section('GOT', bytearray(got_size), env.arch.word_size)
    if env.arch.name == 'EM_XTENSA':
        env.sections.insert(0, env.got_section)
    else:
        env.sections.append(env.got_section)
    if env.arch.name == 'EM_XTENSA':
        lit_size = len(env.lit_entries) * env.arch.word_size
        env.lit_section = Section('LIT', bytearray(lit_size), env.arch.word_size)
        env.sections.insert(1, env.lit_section)
    env.qstr_table_section = Section('.external.qstr_table', bytearray(native_qstr_vals_len * env.arch.qstr_entry_size), env.arch.qstr_entry_size)
    env.obj_table_section = Section('.external.obj_table', bytearray(0 * env.arch.word_size), env.arch.word_size)
    mp_fun_table_sec = Section('.external.mp_fun_table', b'', 0)
    fun_table = {key: 67 + idx for (idx, key) in enumerate(['mp_type_type', 'mp_type_str', 'mp_type_list', 'mp_type_dict', 'mp_type_fun_builtin_0', 'mp_type_fun_builtin_1', 'mp_type_fun_builtin_2', 'mp_type_fun_builtin_3', 'mp_type_fun_builtin_var', 'mp_stream_read_obj', 'mp_stream_readinto_obj', 'mp_stream_unbuffered_readline_obj', 'mp_stream_write_obj'])}
    for sym in env.unresolved_syms:
        assert sym['st_value'] == 0
        if sym.name == '_GLOBAL_OFFSET_TABLE_':
            pass
        elif sym.name == 'mp_fun_table':
            sym.section = Section('.external', b'', 0)
        elif sym.name == 'mp_native_qstr_table':
            sym.section = env.qstr_table_section
        elif sym.name == 'mp_native_obj_table':
            sym.section = env.obj_table_section
        elif sym.name in env.known_syms:
            sym.resolved = env.known_syms[sym.name]
        elif sym.name in fun_table:
            sym.section = mp_fun_table_sec
            sym.mp_fun_table_offset = fun_table[sym.name]
        else:
            raise LinkError('{}: undefined symbol: {}'.format(sym.filename, sym.name))
    env.full_text = bytearray(env.arch.asm_jump(8))
    env.full_rodata = bytearray(0)
    env.full_bss = bytearray(0)
    for sec in env.sections:
        if env.arch.separate_rodata and sec.name.startswith(('.rodata', '.data.rel.ro')):
            data = env.full_rodata
        elif sec.name.startswith('.bss'):
            data = env.full_bss
        else:
            data = env.full_text
        sec.addr = align_to(len(data), sec.alignment)
        data.extend(b'\x00' * (sec.addr - len(data)))
        data.extend(sec.data)
    env.print_sections()
    populate_got(env)
    if env.arch.name == 'EM_XTENSA':
        populate_lit(env)
    for sec in env.sections:
        if not sec.reloc:
            continue
        log(LOG_LEVEL_3, '{}: {} relocations via {}:'.format(sec.filename, sec.name, sec.reloc_name))
        for r in sec.reloc:
            if sec.name.startswith(('.text', '.rodata')):
                do_relocation_text(env, sec.addr, r)
            elif sec.name.startswith('.data.rel.ro'):
                do_relocation_data(env, sec.addr, r)
            else:
                assert 0, sec.name

class MPYOutput:

    def open(self, fname):
        if False:
            print('Hello World!')
        self.f = open(fname, 'wb')
        self.prev_base = -1
        self.prev_offset = -1

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.f.close()

    def write_bytes(self, buf):
        if False:
            while True:
                i = 10
        self.f.write(buf)

    def write_uint(self, val):
        if False:
            return 10
        b = bytearray()
        b.insert(0, val & 127)
        val >>= 7
        while val:
            b.insert(0, 128 | val & 127)
            val >>= 7
        self.write_bytes(b)

    def write_qstr(self, s):
        if False:
            print('Hello World!')
        if s in qstrutil.static_qstr_list:
            self.write_uint(qstrutil.static_qstr_list.index(s) + 1 << 1 | 1)
        else:
            s = bytes(s, 'ascii')
            self.write_uint(len(s) << 1)
            self.write_bytes(s)
            self.write_bytes(b'\x00')

    def write_reloc(self, base, offset, dest, n):
        if False:
            return 10
        need_offset = not (base == self.prev_base and offset == self.prev_offset + 1)
        self.prev_offset = offset + n - 1
        if dest <= 2:
            dest = dest << 1 | (n > 1)
        else:
            assert 6 <= dest <= 127
            assert n == 1
        dest = dest << 1 | need_offset
        assert 0 <= dest <= 254, dest
        self.write_bytes(bytes([dest]))
        if need_offset:
            if base == '.text':
                base = 0
            elif base == '.rodata':
                base = 1
            self.write_uint(offset << 1 | base)
        if n > 1:
            self.write_uint(n)

def build_mpy(env, entry_offset, fmpy, native_qstr_vals):
    if False:
        for i in range(10):
            print('nop')
    jump = env.arch.asm_jump(entry_offset)
    env.full_text[:len(jump)] = jump
    log(LOG_LEVEL_1, 'arch:         {}'.format(env.arch.name))
    log(LOG_LEVEL_1, 'text size:    {}'.format(len(env.full_text)))
    if len(env.full_rodata):
        log(LOG_LEVEL_1, 'rodata size:  {}'.format(len(env.full_rodata)))
    log(LOG_LEVEL_1, 'bss size:     {}'.format(len(env.full_bss)))
    log(LOG_LEVEL_1, 'GOT entries:  {}'.format(len(env.got_entries)))
    out = MPYOutput()
    out.open(fmpy)
    out.write_bytes(bytearray([ord('M'), MPY_VERSION, env.arch.mpy_feature | MPY_SUB_VERSION, MP_SMALL_INT_BITS]))
    out.write_uint(1 + len(native_qstr_vals))
    out.write_uint(0)
    out.write_qstr(fmpy)
    for q in native_qstr_vals:
        out.write_qstr(q)
    out.write_uint(len(env.full_text) << 3 | MP_CODE_NATIVE_VIPER - MP_CODE_BYTECODE)
    out.write_bytes(env.full_text)
    scope_flags = MP_SCOPE_FLAG_VIPERRELOC
    if len(env.full_rodata):
        scope_flags |= MP_SCOPE_FLAG_VIPERRODATA
    if len(env.full_bss):
        scope_flags |= MP_SCOPE_FLAG_VIPERBSS
    out.write_uint(scope_flags)
    if len(env.full_rodata):
        rodata_const_table_idx = 1
        out.write_uint(len(env.full_rodata))
    if len(env.full_bss):
        bss_const_table_idx = 2
        out.write_uint(len(env.full_bss))
    if len(env.full_rodata):
        out.write_bytes(env.full_rodata)
    prev_kind = None
    prev_base = None
    prev_offset = None
    prev_n = None
    for (base, addr, kind) in env.mpy_relocs:
        if isinstance(kind, str) and kind.startswith('.text'):
            kind = 0
        elif isinstance(kind, str) and kind.startswith(('.rodata', '.data.rel.ro')):
            if env.arch.separate_rodata:
                kind = rodata_const_table_idx
            else:
                kind = 0
        elif isinstance(kind, str) and kind.startswith('.bss'):
            kind = bss_const_table_idx
        elif kind == 'mp_native_qstr_table':
            kind = 6
        elif kind == 'mp_native_obj_table':
            kind = 7
        elif kind == 'mp_fun_table':
            kind = 8
        else:
            kind = 9 + kind
        assert addr % env.arch.word_size == 0, addr
        offset = addr // env.arch.word_size
        if kind == prev_kind and base == prev_base and (offset == prev_offset + 1):
            prev_n += 1
            prev_offset += 1
        else:
            if prev_kind is not None:
                out.write_reloc(prev_base, prev_offset - prev_n + 1, prev_kind, prev_n)
            prev_kind = kind
            prev_base = base
            prev_offset = offset
            prev_n = 1
    if prev_kind is not None:
        out.write_reloc(prev_base, prev_offset - prev_n + 1, prev_kind, prev_n)
    out.write_bytes(b'\xff')
    out.close()

def do_preprocess(args):
    if False:
        for i in range(10):
            print('nop')
    if args.output is None:
        assert args.files[0].endswith('.c')
        args.output = args.files[0][:-1] + 'config.h'
    (static_qstrs, qstr_vals) = extract_qstrs(args.files)
    with open(args.output, 'w') as f:
        print('#include <stdint.h>\ntypedef uintptr_t mp_uint_t;\ntypedef intptr_t mp_int_t;\ntypedef uintptr_t mp_off_t;', file=f)
        for (i, q) in enumerate(static_qstrs):
            print('#define %s (%u)' % (q, i + 1), file=f)
        for (i, q) in enumerate(sorted(qstr_vals)):
            print('#define %s (mp_native_qstr_table[%d])' % (q, i + 1), file=f)
        print('extern const uint16_t mp_native_qstr_table[];', file=f)
        print('extern const mp_uint_t mp_native_obj_table[];', file=f)

def do_link(args):
    if False:
        while True:
            i = 10
    if args.output is None:
        assert args.files[0].endswith('.o')
        args.output = args.files[0][:-1] + 'mpy'
    native_qstr_vals = []
    if args.qstrs is not None:
        with open(args.qstrs) as f:
            for l in f:
                m = re.match('#define MP_QSTR_([A-Za-z0-9_]*) \\(mp_native_', l)
                if m:
                    native_qstr_vals.append(m.group(1))
    log(LOG_LEVEL_2, 'qstr vals: ' + ', '.join(native_qstr_vals))
    env = LinkEnv(args.arch)
    try:
        for file in args.files:
            load_object_file(env, file)
        link_objects(env, len(native_qstr_vals))
        build_mpy(env, env.find_addr('mpy_init'), args.output, native_qstr_vals)
    except LinkError as er:
        print('LinkError:', er.args[0])
        sys.exit(1)

def main():
    if False:
        return 10
    import argparse
    cmd_parser = argparse.ArgumentParser(description='Run scripts on the pyboard.')
    cmd_parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity')
    cmd_parser.add_argument('--arch', default='x64', help='architecture')
    cmd_parser.add_argument('--preprocess', action='store_true', help='preprocess source files')
    cmd_parser.add_argument('--qstrs', default=None, help='file defining additional qstrs')
    cmd_parser.add_argument('--output', '-o', default=None, help='output .mpy file (default to input with .o->.mpy)')
    cmd_parser.add_argument('files', nargs='+', help='input files')
    args = cmd_parser.parse_args()
    global log_level
    log_level = args.verbose
    if args.preprocess:
        do_preprocess(args)
    else:
        do_link(args)
if __name__ == '__main__':
    main()