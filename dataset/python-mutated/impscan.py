import volatility.utils as utils
import volatility.obj as obj
import volatility.plugins.common as common
import volatility.plugins.taskmods as taskmods
import volatility.debug as debug
import volatility.win32.tasks as tasks
import volatility.win32.modules as modules
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address
try:
    import distorm3
    has_distorm = True
except ImportError:
    has_distorm = False

class ImpScan(common.AbstractWindowsCommand):
    """Scan for calls to imported functions"""

    def __init__(self, config, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        config.remove_option('PID')
        config.add_option('PID', short_option='p', default=None, help='Process ID (leave off to scan kernel memory)', action='store', type='int')
        config.add_option('OFFSET', short_option='o', default=None, help='EPROCESS offset (in hex) in the physical address space', action='store', type='int')
        config.add_option('BASE', short_option='b', default=None, help='Base address in process memory if --pid ' + 'is supplied, otherwise an address in kernel space', action='store', type='int')
        config.add_option('SIZE', short_option='s', default=None, help='Size of memory to scan', action='store', type='int')
        self.forwarded_imports = {'RtlGetLastWin32Error': 'kernel32.dll!GetLastError', 'RtlSetLastWin32Error': 'kernel32.dll!SetLastError', 'RtlRestoreLastWin32Error': 'kernel32.dll!SetLastError', 'RtlAllocateHeap': 'kernel32.dll!HeapAlloc', 'RtlReAllocateHeap': 'kernel32.dll!HeapReAlloc', 'RtlFreeHeap': 'kernel32.dll!HeapFree', 'RtlEnterCriticalSection': 'kernel32.dll!EnterCriticalSection', 'RtlLeaveCriticalSection': 'kernel32.dll!LeaveCriticalSection', 'RtlDeleteCriticalSection': 'kernel32.dll!DeleteCriticalSection', 'RtlZeroMemory': 'kernel32.dll!ZeroMemory', 'RtlSizeHeap': 'kernel32.dll!HeapSize', 'RtlUnwind': 'kernel32.dll!RtlUnwind'}

    @staticmethod
    def enum_apis(all_mods):
        if False:
            i = 10
            return i + 15
        'Enumerate all exported functions from kernel \n        or process space. \n\n        @param all_mods: list of _LDR_DATA_TABLE_ENTRY \n\n        To enum kernel APIs, all_mods is a list of drivers. \n        To enum process APIs, all_mods is a list of DLLs. \n\n        The function name is used if available, otherwise \n        we take the ordinal value. \n        '
        exports = {}
        for mod in all_mods:
            for (ordinal, func_addr, func_name) in mod.exports():
                if func_addr != None:
                    name = func_name or ordinal or ''
                    exports[int(mod.DllBase + func_addr)] = (mod, str(name))
        return exports

    def _call_or_unc_jmp(self, op):
        if False:
            for i in range(10):
                print('nop')
        'Determine if an instruction is a call or an\n        unconditional jump\n\n        @param op: a distorm3 Op object\n        '
        return op.flowControl == 'FC_CALL' and op.mnemonic == 'CALL' or (op.flowControl == 'FC_UNC_BRANCH' and op.mnemonic == 'JMP')

    def _vicinity_scan(self, addr_space, calls_imported, apis, base_address, data_len, is_wow64=False, forward=True):
        if False:
            print('Hello World!')
        'Scan forward from the lowest IAT entry found or\n        backward from the highest IAT entry found. We do this \n        because not every imported function will be called \n        from the code section and sometimes page(s) with the \n        calls are unavailable. \n\n        @param addr_space: an AS\n        @param calls_imported: dictionary of confirmed imports\n        @param apis: dictionary of exported functions in the AS \n        @param base_address: memory base address \n        @param data_len: size in bytes to check from base_address\n        @param is_wow64: True if its a Wow64 process\n        @param forward: the direction for the vicinity scan\n        '
        sortedlist = calls_imported.keys()
        sortedlist.sort()
        if not sortedlist:
            return
        if is_wow64:
            addr_type = 'int'
        else:
            addr_type = 'address'
        size_of_address = addr_space.profile.get_obj_size(addr_type)
        if forward:
            start_addr = sortedlist[0]
        else:
            start_addr = sortedlist[len(sortedlist) - 1]
        threshold = 5
        i = 0
        while threshold and i < 8192:
            if forward:
                next_addr = start_addr + i * size_of_address
            else:
                next_addr = start_addr - i * size_of_address
            call_dest = obj.Object(addr_type, offset=next_addr, vm=addr_space).v()
            if not call_dest or (call_dest > base_address and call_dest < base_address + data_len):
                threshold -= 1
                i += 1
                continue
            if call_dest in apis and call_dest not in calls_imported:
                calls_imported[next_addr] = call_dest
                threshold = 5
            else:
                threshold -= 1
            i += 1

    def _original_import(self, mod_name, func_name):
        if False:
            return 10
        'Revert a forwarded import to the original module \n        and function name. \n\n        @param mod_name: current module name \n        @param func_name: current function name \n        '
        if func_name in self.forwarded_imports:
            return self.forwarded_imports[func_name].split('!')
        else:
            return (mod_name, func_name)

    def call_scan(self, addr_space, base_address, data, is_wow64=False):
        if False:
            for i in range(10):
                print('nop')
        "Disassemble a block of data and yield possible \n        calls to imported functions. We're looking for \n        instructions such as these:\n\n        x86:\n        CALL DWORD [0x1000400]\n        JMP  DWORD [0x1000400]\n        \n        x64:\n        CALL QWORD [RIP+0x989d]\n\n        On x86, the 0x1000400 address is an entry in the \n        IAT or call table. It stores a DWORD which is the \n        location of the API function being called. \n\n        On x64, the 0x989d is a relative offset from the\n        current instruction (RIP). \n\n        @param addr_space: an AS to scan with\n        @param base_address: memory base address\n        @param data: buffer of data found at base_address\n        @param is_wow64: True if its a Wow64 process\n        "
        end_address = base_address + len(data)
        memory_model = addr_space.profile.metadata.get('memory_model', '32bit')
        if memory_model == '32bit' or is_wow64:
            mode = distorm3.Decode32Bits
            addr_type = 'int'
        else:
            mode = distorm3.Decode64Bits
            addr_type = 'address'
        for op in distorm3.DecomposeGenerator(base_address, data, mode):
            if not op.valid:
                continue
            iat_loc = None
            if memory_model == '32bit' or is_wow64:
                if self._call_or_unc_jmp(op) and op.operands[0].type == 'AbsoluteMemoryAddress':
                    iat_loc = op.operands[0].disp & 4294967295
            elif self._call_or_unc_jmp(op) and 'FLAG_RIP_RELATIVE' in op.flags and (op.operands[0].type == 'AbsoluteMemory'):
                iat_loc = op.address + op.size + op.operands[0].disp
            if not iat_loc or iat_loc < base_address or iat_loc > end_address:
                continue
            call_dest = obj.Object(addr_type, offset=iat_loc, vm=addr_space)
            if call_dest == None:
                continue
            yield (op.address, iat_loc, int(call_dest))

    def calculate(self):
        if False:
            return 10
        if not has_distorm:
            debug.error('You must install distorm3')
        addr_space = utils.load_as(self._config)
        all_mods = []
        if self._config.OFFSET != None:
            all_tasks = [taskmods.DllList.virtual_process_from_physical_offset(addr_space, self._config.OFFSET)]
        else:
            all_tasks = list(tasks.pslist(addr_space))
            all_mods = list(modules.lsmod(addr_space))
        if not self._config.PID and (not self._config.OFFSET):
            if not self._config.BASE:
                debug.error('You must specify --BASE')
            base_address = self._config.BASE
            size_to_read = self._config.SIZE
            if not size_to_read:
                for module in all_mods:
                    if module.DllBase == base_address:
                        size_to_read = module.SizeOfImage
                        break
                if not size_to_read:
                    pefile = obj.Object('_IMAGE_DOS_HEADER', offset=base_address, vm=addr_space)
                    try:
                        nt_header = pefile.get_nt_header()
                        size_to_read = nt_header.OptionalHeader.SizeOfImage
                    except ValueError:
                        pass
                    if not size_to_read:
                        debug.error('You must specify --SIZE')
            kernel_space = tasks.find_space(addr_space, all_tasks, base_address)
            if not kernel_space:
                debug.error('Cannot read supplied address')
            data = kernel_space.zread(base_address, size_to_read)
            apis = self.enum_apis(all_mods)
            addr_space = kernel_space
            is_wow64 = False
        else:
            task = None
            for atask in all_tasks:
                if self._config.OFFSET or atask.UniqueProcessId == self._config.PID:
                    task = atask
                    break
            if not task:
                debug.error('You must supply an active PID')
            task_space = task.get_process_address_space()
            if not task_space:
                debug.error('Cannot acquire process AS')
            all_mods = list(task.get_load_modules())
            if not all_mods:
                debug.error('Cannot load DLLs in process AS')
            if self._config.BASE:
                base_address = self._config.BASE
                size_to_read = self._config.SIZE
                if not size_to_read:
                    for vad in task.VadRoot.traverse():
                        if base_address >= vad.Start and base_address <= vad.End:
                            size_to_read = vad.Length
                    if not size_to_read:
                        debug.error('You must specify --SIZE')
            else:
                base_address = all_mods[0].DllBase
                size_to_read = all_mods[0].SizeOfImage
            is_wow64 = task.IsWow64
            data = task_space.zread(base_address, size_to_read)
            apis = self.enum_apis(all_mods)
            addr_space = task_space
        calls_imported = dict(((iat, call) for (_, iat, call) in self.call_scan(addr_space, base_address, data, is_wow64) if call in apis))
        self._vicinity_scan(addr_space, calls_imported, apis, base_address, len(data), is_wow64, forward=True)
        self._vicinity_scan(addr_space, calls_imported, apis, base_address, len(data), is_wow64, forward=False)
        for (iat, call) in sorted(calls_imported.items()):
            yield (iat, call, apis[call][0], apis[call][1])

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('IAT', Address), ('Call', Address), ('Module', str), ('Function', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for (iat, call, mod, func) in data:
            (mod_name, func_name) = self._original_import(str(mod.BaseDllName or ''), func)
            yield (0, [Address(iat), Address(call), str(mod_name), str(func_name)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        'Render as text'
        self.table_header(outfd, [('IAT', '[addrpad]'), ('Call', '[addrpad]'), ('Module', '20'), ('Function', '')])
        for (iat, call, mod, func) in data:
            (mod_name, func_name) = self._original_import(str(mod.BaseDllName or ''), func)
            self.table_row(outfd, iat, call, mod_name, func_name)

    def render_idc(self, outfd, data):
        if False:
            i = 10
            return i + 15
        'Render as IDC'
        bits = None
        for (iat, _, mod, func) in data:
            if bits == None:
                bits = mod.obj_vm.profile.metadata.get('memory_model', '32bit')
            (_, func_name) = self._original_import(str(mod.BaseDllName or ''), func)
            if bits == '32bit':
                outfd.write('MakeDword(0x{0:08X});\n'.format(iat))
            else:
                outfd.write('MakeQword(0x{0:08X});\n'.format(iat))
            outfd.write('MakeName(0x{0:08X}, "{1}");\n'.format(iat, func_name))