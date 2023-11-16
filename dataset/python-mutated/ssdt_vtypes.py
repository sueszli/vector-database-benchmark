import sys
import volatility.debug as debug
import volatility.obj as obj
ssdt_vtypes = {'_SERVICE_DESCRIPTOR_TABLE': [64, {'Descriptors': [0, ['array', 4, ['_SERVICE_DESCRIPTOR_ENTRY']]]}], '_SERVICE_DESCRIPTOR_ENTRY': [16, {'KiServiceTable': [0, ['pointer', ['void']]], 'CounterBaseTable': [4, ['pointer', ['unsigned long']]], 'ServiceLimit': [8, ['unsigned long']], 'ArgumentTable': [12, ['pointer', ['unsigned char']]]}]}
ssdt_vtypes_2003 = {'_SERVICE_DESCRIPTOR_TABLE': [32, {'Descriptors': [0, ['array', 2, ['_SERVICE_DESCRIPTOR_ENTRY']]]}]}
ssdt_vtypes_64 = {'_SERVICE_DESCRIPTOR_TABLE': [64, {'Descriptors': [0, ['array', 2, ['_SERVICE_DESCRIPTOR_ENTRY']]]}], '_SERVICE_DESCRIPTOR_ENTRY': [32, {'KiServiceTable': [0, ['pointer64', ['void']]], 'CounterBaseTable': [8, ['pointer64', ['unsigned long']]], 'ServiceLimit': [16, ['unsigned long long']], 'ArgumentTable': [24, ['pointer64', ['unsigned char']]]}]}

def syscalls_property(x):
    if False:
        print('Hello World!')
    debug.debug("Deprecation warning: Please use profile.additional['syscalls'] over profile.syscalls")
    return x.additional.get('syscalls', [[], []])

class WinSyscallsAttribute(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.__class__.syscalls = property(syscalls_property)

class AbstractSyscalls(obj.ProfileModification):
    syscall_module = 'No default'

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        module = sys.modules.get(self.syscall_module, None)
        profile.additional['syscalls'] = module.syscalls

class WinXPSyscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.xp_sp2_x86_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 5, 'minor': lambda x: x == 1}

class Win64SyscallVTypes(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update(ssdt_vtypes_64)

class Win2003SyscallVTypes(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 5, 'minor': lambda x: x == 2}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update(ssdt_vtypes_2003)

class Win2003SP0Syscalls(AbstractSyscalls):
    before = ['Win2003SP12Syscalls']
    syscall_module = 'volatility.plugins.overlays.windows.win2003_sp0_x86_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 5, 'minor': lambda x: x == 2, 'build': lambda x: x == 3789}

class Win2003SP12Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win2003_sp12_x86_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 5, 'minor': lambda x: x == 2}

class Win2003SP12x64Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win2003_sp12_x64_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 5, 'minor': lambda x: x == 2}

class VistaSP0Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.vista_sp0_x86_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'build': lambda x: x == 6000}

class VistaSP0x64Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.vista_sp0_x64_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'build': lambda x: x == 6000}

class VistaSP12Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.vista_sp12_x86_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'build': lambda x: x >= 6001}

class VistaSP12x64Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.vista_sp12_x64_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'build': lambda x: x >= 6001}

class Win7SP01Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win7_sp01_x86_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 1}

class Win7SP01x64Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win7_sp01_x64_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 1}

class Win8SP0x64Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win8_sp0_x64_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 2}

class Win8SP0x86Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win8_sp0_x86_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 2}

class Win8SP1x86Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win8_sp1_x86_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 3}

class Win8SP1x64Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win8_sp1_x64_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 3}

class Win10x64_10586_Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win10_x64_10586_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x == 10586}

class Win10x86_10586_Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win10_x86_10586_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x == 10586}

class Win10x64_14393_Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win10_x64_14393_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x == 14393}

class Win10x86_14393_Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win10_x86_14393_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x == 14393}

class Win10x64_15063_Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win10_x64_15063_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x == 15063}

class Win10x86_15063_Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win10_x86_15063_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x == 15063}

class Win10x64_16299_Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win10_x64_16299_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x == 16299}

class Win10x86_16299_Syscalls(AbstractSyscalls):
    syscall_module = 'volatility.plugins.overlays.windows.win10_x86_16299_syscalls'
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x == 16299}