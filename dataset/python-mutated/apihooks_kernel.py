"""
@author:       Cem Gurkok
@license:      GNU General Public License 2.0 or later
@contact:      cemgurkok@gmail.com
@organization: 
"""
import volatility.obj as obj
import common
import volatility.commands as commands
import distorm3
import volatility.plugins.mac.check_sysctl as check_sysctl
import volatility.plugins.mac.check_trap_table as check_trap_table
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class mac_apihooks_kernel(common.AbstractMacCommand):
    """ Checks to see if system call and kernel functions are hooked """

    def __init__(self, config, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.addr_space = None
        commands.Command.__init__(self, config, *args, **kwargs)
        self._config.add_option('CHECKKEXTS', short_option='X', default=False, cache_invalidator=False, help="Check all kext functions in the kext's symbol table for hooking, including kernel symbol table", action='store_true')
        self._config.add_option('CHECKKERNEL', short_option='K', default=False, cache_invalidator=False, help='Check only kernel symbol table functions for hooking', action='store_true')

    def getKextSymbols(self, kext_obj=None, kext_name=None, kext_addr=0, onlyFunctions=False, fmodel='64bit'):
        if False:
            while True:
                i = 10
        mach_header_struct = 'macho64_header'
        segment_command_struct = 'macho64_segment_command'
        section_struct = 'macho64_section'
        nlist_struct = 'macho64_nlist'
        LC_SEGMENT = 25
        if fmodel == '32bit':
            mach_header_struct = 'macho32_header'
            segment_command_struct = 'macho32_segment_command'
            section_struct = 'macho32_section'
            nlist_struct = 'macho32_nlist'
            LC_SEGMENT = 1
        if kext_name != None:
            if kext_name in ['kernel', '__kernel__']:
                kext_addr = self.addr_space.profile.get_symbol('_g_kernel_kmod_info')
            else:
                kmodaddr = obj.Object('Pointer', offset=self.addr_space.profile.get_symbol('_kmod'), vm=self.addr_space)
                kmod = kmodaddr.dereference_as('kmod_info')
                while kmod.is_valid():
                    if str(kmod.name) == kext_name:
                        kext_addr = kmod.address
                        break
                    kmod = kmod.next
                if kext_addr == None:
                    yield
        elif kext_obj != None:
            kext_addr = kext_obj.adddress
        mh = obj.Object(mach_header_struct, offset=kext_addr, vm=self.addr_space)
        seg_offset = mh.obj_offset + self.addr_space.profile.get_obj_size(mach_header_struct)
        linkedit_vmaddr = 0
        symtab_symoff = 0
        symtab_stroff = 0
        symbol_cnt = 0
        linkedit_fileoffset = 0
        linkedit_size = 0
        text_sect_num = 0
        sect_cnt = 0
        for i in xrange(0, mh.ncmds):
            seg = obj.Object(segment_command_struct, offset=seg_offset, vm=self.addr_space)
            if seg.cmd == 25 and seg.segname and (str(seg.segname) == '__LINKEDIT'):
                linkedit_vmaddr = seg.vmaddr
                linkedit_fileoffset = seg.fileoff
                linkedit_size = seg.filesize
            elif seg.cmd == 2:
                symtab = obj.Object('symtab_command', offset=seg_offset, vm=self.addr_space)
                symtab_symoff = symtab.symoff
                symtab_stroff = symtab.stroff
                symbol_cnt = symtab.nsyms
            if seg.cmd == LC_SEGMENT:
                for j in xrange(0, seg.nsects):
                    sect_cnt += 1
                    sect = obj.Object(section_struct, offset=seg_offset + self.addr_space.profile.get_obj_size(segment_command_struct) + self.addr_space.profile.get_obj_size(section_struct) * j, vm=self.addr_space)
                    sect_name = ''.join(map(str, str(sect.sectname))).strip(' \t\r\n\x00')
                    if seg.cmd == 25 and seg.segname and (str(seg.segname) == '__TEXT') and (sect_name == '__text'):
                        text_sect_num = sect_cnt
            seg_offset += seg.cmdsize
        symbol_offset = symtab_symoff - linkedit_fileoffset
        string_offset = symtab_stroff - linkedit_fileoffset
        for i in range(0, symbol_cnt - 1):
            sym = obj.Object(nlist_struct, offset=linkedit_vmaddr + symbol_offset + i * self.addr_space.profile.get_obj_size(nlist_struct), vm=self.addr_space)
            sym_addr = sym.n_strx + linkedit_vmaddr + string_offset
            sym_name = obj.Object('String', offset=sym_addr, vm=self.addr_space, length=256)
            if sym_name.is_valid():
                if onlyFunctions:
                    if sym.n_sect == text_sect_num:
                        yield (sym_name, sym.n_value)
                else:
                    yield (sym_name, sym.n_value)

    def findKextWithAddress(self, addr):
        if False:
            print('Hello World!')
        kexts = []
        kp = self.addr_space.profile.get_symbol('_g_kernel_kmod_info')
        kmodk = obj.Object('kmod_info', offset=kp, vm=self.addr_space)
        kexts.append(kmodk)
        p = self.addr_space.profile.get_symbol('_kmod')
        kmodaddr = obj.Object('Pointer', offset=p, vm=self.addr_space)
        kmod = kmodaddr.dereference_as('kmod_info')
        while kmod.is_valid():
            kexts.append(kmod)
            kmod = kmod.next
        for kext in kexts:
            if addr >= kext.address and addr <= kext.address + kext.m('size'):
                return kext.name
        return 'UNKNOWN'

    def isCallReferenceModified(self, model, distorm_mode, func_addr, kernel_syms, kmods):
        if False:
            print('Hello World!')
        modified = False
        data = self.addr_space.read(func_addr, 750)
        n = 0
        d = None
        push_val = None
        regs = {}
        ops = []
        for op in distorm3.Decompose(func_addr, data, distorm_mode):
            ops.append(op)
        for op in distorm3.Decompose(func_addr, data, distorm_mode):
            if not op.valid or op.mnemonic == 'NOP':
                break
            if op.flowControl == 'FC_CALL':
                if push_val:
                    push_val = None
                if op.mnemonic == 'CALL' and op.operands[0].type == 'AbsoluteMemoryAddress':
                    if model == '32bit':
                        const = op.operands[0].disp & 4294967295
                        d = obj.Object('unsigned int', offset=const, vm=self.addr_space)
                    else:
                        const = op.operands[0].disp
                        d = obj.Object('unsigned long long', offset=const, vm=self.addr_space)
                    if self.outside_module(d, kernel_syms, kmods):
                        break
                elif op.operands[0].type == 'Immediate':
                    d = op.operands[0].value
                    if self.outside_module(d, kernel_syms, kmods):
                        break
                elif op.operands[0].type == 'Register':
                    d = regs.get(op.operands[0].name)
                    if d and self.outside_module(d, kernel_syms, kmods):
                        break
            n += 1
        if d and self.outside_module(d, kernel_syms, kmods) == True and (str(ops[n + 1].mnemonic) not in ['DB 0xff', 'ADD', 'XCHG', 'OUTS']):
            modified = True
        return (modified, d)

    def isPrologInlined(self, model, distorm_mode, func_addr):
        if False:
            i = 10
            return i + 15
        inlined = False
        content = self.addr_space.read(func_addr, 24)
        op_cnt = 1
        for op in distorm3.Decompose(func_addr, content, distorm_mode):
            if op_cnt == 2:
                if model == '32bit':
                    if op.mnemonic == 'MOV' and len(op.operands) == 2 and (op.operands[0].type == 'Register') and (op.operands[1].type == 'Register') and (op.operands[0].name == 'EBP') and (op.operands[1].name == 'ESP') and (prev_op.mnemonic == 'PUSH') and (len(prev_op.operands) == 1) and (prev_op.operands[0].type == 'Register') and (prev_op.operands[0].name == 'EBP'):
                        pass
                    else:
                        inlined = True
                elif model == '64bit':
                    if op.mnemonic == 'MOV' and len(op.operands) == 2 and (op.operands[0].type == 'Register') and (op.operands[1].type == 'Register') and (op.operands[0].name == 'RBP') and (op.operands[1].name == 'RSP') and (prev_op.mnemonic == 'PUSH') and (len(prev_op.operands) == 1) and (prev_op.operands[0].type == 'Register') and (prev_op.operands[0].name == 'RBP'):
                        pass
                    elif prev_op.mnemonic == 'PUSH' and len(prev_op.operands) == 1 and (prev_op.operands[0].type == 'Register') and (prev_op.operands[0].name == 'RBP') and (op.mnemonic == 'PUSH') and (len(op.operands) == 1) and (op.operands[0].type == 'Register') and (op.operands[0].name in ['RSP', 'RBX', 'R12', 'R13', 'R14', 'R15']):
                        pass
                    else:
                        inlined = True
                break
            prev_op = op
            op_cnt += 1
        return inlined

    def outside_module(self, addr, kernel_syms, kmods):
        if False:
            for i in range(10):
                print('nop')
        (good, _) = common.is_known_address_name(addr, kernel_syms, kmods)
        return not good

    def isInlined(self, model, distorm_mode, func_addr, kernel_syms, kmods):
        if False:
            for i in range(10):
                print('nop')
        inlined = False
        data = self.addr_space.read(func_addr, 24)
        n = 0
        d = None
        push_val = None
        regs = {}
        ops = []
        for op in distorm3.Decompose(func_addr, data, distorm_mode):
            ops.append(op)
        for op in distorm3.Decompose(func_addr, data, distorm_mode):
            if not op.valid or n == 3:
                break
            if op.flowControl == 'FC_CALL':
                if push_val:
                    push_val = None
                if op.mnemonic == 'CALL' and op.operands[0].type == 'AbsoluteMemoryAddress':
                    if model == '32bit':
                        const = op.operands[0].disp & 4294967295
                        d = obj.Object('unsigned int', offset=const, vm=addr_space)
                    else:
                        const = op.operands[0].disp
                        d = obj.Object('unsigned long long', offset=const, vm=addr_space)
                    if self.outside_module(d, kernel_syms, kmods):
                        break
                elif op.operands[0].type == 'Immediate':
                    d = op.operands[0].value
                    if self.outside_module(d, kernel_syms, kmods):
                        break
                elif op.operands[0].type == 'Register':
                    d = regs.get(op.operands[0].name)
                    if d and self.outside_module(d, kernel_syms, kmods):
                        break
            elif op.flowControl == 'FC_UNC_BRANCH' and op.mnemonic == 'JMP':
                if push_val:
                    push_val = None
                if op.size > 2:
                    if op.operands[0].type == 'AbsoluteMemoryAddress':
                        if model == '32bit':
                            const = op.operands[0].disp & 4294967295
                            d = obj.Object('unsigned int', offset=const, vm=addr_space)
                        else:
                            const = op.operands[0].disp
                            d = obj.Object('long long', offset=const, vm=addr_space)
                        if self.outside_module(d, kernel_syms, kmods):
                            break
                    elif op.operands[0].type == 'Immediate':
                        d = op.operands[0].value
                        if self.outside_module(d, kernel_syms, kmods):
                            break
                elif op.size == 2 and op.operands[0].type == 'Register':
                    d = regs.get(op.operands[0].name)
                    if d and self.outside_module(d, kernel_syms, kmods):
                        break
            elif op.flowControl == 'FC_NONE':
                if op.mnemonic == 'PUSH' and op.operands[0].type == 'Immediate' and (op.size == 5):
                    push_val = op.operands[0].value
                if op.mnemonic == 'MOV' and op.operands[0].type == 'Register' and (op.operands[1].type == 'Immediate'):
                    if push_val:
                        push_val = None
                    regs[op.operands[0].name] = op.operands[1].value
            elif op.flowControl == 'FC_RET':
                if push_val:
                    d = push_val
                    if self.outside_module(d, kernel_syms, kmods):
                        break
                break
            n += 1
        if d and self.outside_module(d, kernel_syms, kmods) == True and (str(ops[n + 1].mnemonic) not in ['DB 0xff', 'ADD', 'XCHG', 'OUTS']):
            inlined = True
        return (inlined, d)

    def calculate(self):
        if False:
            i = 10
            return i + 15
        common.set_plugin_members(self)
        (kernel_symbol_addresses, kmod) = common.get_kernel_function_addrs(self)
        model = self.addr_space.profile.metadata.get('memory_model', 0)
        if model == '32bit':
            distorm_mode = distorm3.Decode32Bits
        else:
            distorm_mode = distorm3.Decode64Bits
        sym_addrs = self.profile.get_all_function_addresses()
        kp = self.addr_space.profile.get_symbol('_g_kernel_kmod_info')
        kmodk = obj.Object('kmod_info', offset=kp, vm=self.addr_space)
        k_start = kmodk.address
        k_end = k_start + kmodk.m('size')
        nsysent = obj.Object('int', offset=self.addr_space.profile.get_symbol('_nsysent'), vm=self.addr_space)
        sysents = obj.Object(theType='Array', offset=self.addr_space.profile.get_symbol('_sysent'), vm=self.addr_space, count=nsysent, targetType='sysent')
        dict_syscall_funcs = {}
        list_syscall_names = []
        for (i, sysent) in enumerate(sysents):
            ent_addr = sysent.sy_call.v()
            hooked = ent_addr not in sym_addrs
            (inlined, dst_addr) = self.isInlined(model, distorm_mode, ent_addr, kernel_symbol_addresses, [kmodk])
            prolog_inlined = self.isPrologInlined(model, distorm_mode, ent_addr)
            if hooked == True or inlined == True or prolog_inlined == True:
                if dst_addr != None:
                    kext = self.findKextWithAddress(dst_addr)
                else:
                    kext = self.findKextWithAddress(ent_addr)
                yield ('SyscallTable1', i, ent_addr, hooked, inlined or prolog_inlined, False, '-', kext)
            else:
                ent_name = self.profile.get_symbol_by_address_type('kernel', ent_addr, 'N_FUN')
                if ent_name != '_nosys' and ent_name in dict_syscall_funcs:
                    prev_ent = dict_syscall_funcs[ent_name]
                    kext = self.findKextWithAddress(ent_addr)
                    yield ('SyscallTable', list_syscall_names.index(ent_name), prev_ent.sy_call.v(), False, False, False, '-', kext)
                    yield ('DuplicateSyscall -> {0}'.format(ent_name), i, ent_addr, True, False, False, '-', kext)
                elif ent_name.find('dtrace') > -1:
                    kext = self.findKextWithAddress(ent_addr)
                    yield ('SyscallTable', i, ent_addr, False, False, False, '-', kext)
                else:
                    list_syscall_names.append(ent_name)
                    dict_syscall_funcs[ent_name] = sysent
        kext_addr_list = []
        kmod = obj.Object('kmod_info', offset=self.addr_space.profile.get_symbol('_g_kernel_kmod_info'), vm=self.addr_space)
        kext_addr_list.append((kmod.address.v(), kmod.address + kmod.m('size'), '__kernel__'))
        p = self.addr_space.profile.get_symbol('_kmod')
        kmodaddr = obj.Object('Pointer', offset=p, vm=self.addr_space)
        kmod = kmodaddr.dereference_as('kmod_info')
        while kmod.is_valid():
            kext_addr_list.append((kmod.address.v(), kmod.address + kmod.m('size'), kmod.name))
            kmod = kmod.next
        for (kext_address, kext_end, kext_name) in kext_addr_list:
            for (func_name, func_addr) in self.getKextSymbols(kext_addr=kext_address, onlyFunctions=True, fmodel=model):
                inlined = False
                if func_name in ['pthreads_dummy_symbol']:
                    continue
                (modified, dst_addr) = self.isCallReferenceModified(model, distorm_mode, func_addr, kernel_symbol_addresses, kext_addr_list)
                if modified:
                    if dst_addr != None:
                        hook_kext = self.findKextWithAddress(dst_addr)
                    else:
                        hook_kext = kext_name
                    yield ('SymbolsTable', '-', func_addr, False, modified, False, '-', hook_kext)
                (inlined, dst_addr) = self.isInlined(model, distorm_mode, func_addr, kernel_symbol_addresses, kext_addr_list)
                if inlined:
                    if dst_addr != None:
                        hook_kext = self.findKextWithAddress(dst_addr)
                    else:
                        hook_kext = kext_name
                    yield ('SymbolsTable', '-', func_addr, False, inlined, False, '-', hook_kext)
        args = ()
        trap = check_trap_table.mac_check_trap_table(self._config, args)
        for (table_addr, table_name, i, call_addr, sym_name, hooked) in trap.calculate():
            if hooked == True or 'dtrace' in sym_name:
                kext = self.findKextWithAddress(call_addr)
                yield ('TrapTable', i, call_addr, hooked, False, False, '-', kext)
            else:
                (inlined, dst_addr) = self.isInlined(model, distorm_mode, call_addr, kernel_symbol_addresses, [kmodk])
                if inlined:
                    if dst_addr != None:
                        hook_kext = self.findKextWithAddress(dst_addr)
                    else:
                        hook_kext = kext_name
                    yield ('TrapTable', '-', func_addr, False, inlined, False, '-', hook_kext)
                else:
                    (modified, dst_addr) = self.isCallReferenceModified(model, distorm_mode, call_addr, kernel_symbol_addresses, [kmodk])
                    if modified:
                        if dst_addr != None:
                            hook_kext = self.findKextWithAddress(dst_addr)
                        else:
                            hook_kext = kext_name
                        yield ('TrapTable', '-', func_addr, False, modified, False, '-', hook_kext)

    def unified_output(self, data):
        if False:
            print('Hello World!')
        return TreeGrid([('Table Name', str), ('Index', int), ('Address', Address), ('Symbol', str), ('Inlined', str), ('Shadowed', str), ('Perms', str), ('Hook In', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for (table_name, i, call_addr, hooked, inlined, syscall_shadowed, perms, kext) in data:
            if hooked == False:
                sym_name = self.profile.get_symbol_by_address_type('kernel', call_addr, 'N_FUN')
                if sym_name.find('dtrace') > -1:
                    sym_name = '[HOOKED] {0}'.format(sym_name)
            elif hooked == True:
                sym_name = 'HOOKED'
            else:
                sym_name = hooked
            if inlined == False:
                txt_inlined = 'No'
            elif inlined == True:
                txt_inlined = 'Yes'
            else:
                txt_inlined = '-'
            if syscall_shadowed == False:
                txt_shadowed = 'No'
            elif syscall_shadowed == True:
                txt_shadowed = 'Yes'
            else:
                txt_shadowed = '-'
            yield (0, [str(table_name), int(i), Address(call_addr), str(sym_name), str(txt_inlined), str(txt_shadowed), str(perms), str(kext)])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        self.table_header(outfd, [('Table Name', '<30'), ('Index', '<6'), ('Address', '[addrpad]'), ('Symbol', '<30'), ('Inlined', '<5'), ('Shadowed', '<5'), ('Perms', '<6'), ('Hook In', '')])
        for (table_name, i, call_addr, hooked, inlined, syscall_shadowed, perms, kext) in data:
            if hooked == False:
                sym_name = self.profile.get_symbol_by_address_type('kernel', call_addr, 'N_FUN')
                if sym_name.find('dtrace') > -1:
                    sym_name = '[HOOKED] {0}'.format(sym_name)
            elif hooked == True:
                sym_name = 'HOOKED'
            else:
                sym_name = hooked
            if inlined == False:
                txt_inlined = 'No'
            elif inlined == True:
                txt_inlined = 'Yes'
            else:
                txt_inlined = '-'
            if syscall_shadowed == False:
                txt_shadowed = 'No'
            elif syscall_shadowed == True:
                txt_shadowed = 'Yes'
            else:
                txt_shadowed = '-'
            self.table_row(outfd, table_name, i, call_addr, sym_name, txt_inlined, txt_shadowed, perms, kext)