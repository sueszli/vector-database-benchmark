import volatility.utils as utils
import volatility.obj as obj
import volatility.poolscan as poolscan
import volatility.debug as debug
import volatility.plugins.common as common
import volatility.win32.modules as modules
import volatility.win32.tasks as tasks
import volatility.plugins.malware.devicetree as devicetree
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address
try:
    import distorm3
    has_distorm3 = True
except ImportError:
    has_distorm3 = False
callback_types = {'_NOTIFICATION_PACKET': [16, {'ListEntry': [0, ['_LIST_ENTRY']], 'DriverObject': [8, ['pointer', ['_DRIVER_OBJECT']]], 'NotificationRoutine': [12, ['unsigned int']]}], '_KBUGCHECK_CALLBACK_RECORD': [32, {'Entry': [0, ['_LIST_ENTRY']], 'CallbackRoutine': [8, ['unsigned int']], 'Buffer': [12, ['pointer', ['void']]], 'Length': [16, ['unsigned int']], 'Component': [20, ['pointer', ['String', dict(length=64)]]], 'Checksum': [24, ['pointer', ['unsigned int']]], 'State': [28, ['unsigned char']]}], '_KBUGCHECK_REASON_CALLBACK_RECORD': [28, {'Entry': [0, ['_LIST_ENTRY']], 'CallbackRoutine': [8, ['unsigned int']], 'Component': [12, ['pointer', ['String', dict(length=8)]]], 'Checksum': [16, ['pointer', ['unsigned int']]], 'Reason': [20, ['unsigned int']], 'State': [24, ['unsigned char']]}], '_SHUTDOWN_PACKET': [12, {'Entry': [0, ['_LIST_ENTRY']], 'DeviceObject': [8, ['pointer', ['_DEVICE_OBJECT']]]}], '_EX_CALLBACK_ROUTINE_BLOCK': [8, {'RundownProtect': [0, ['unsigned int']], 'Function': [4, ['unsigned int']], 'Context': [8, ['unsigned int']]}], '_GENERIC_CALLBACK': [12, {'Callback': [4, ['pointer', ['void']]], 'Associated': [8, ['pointer', ['void']]]}], '_REGISTRY_CALLBACK_LEGACY': [56, {'CreateTime': [0, ['WinTimeStamp', dict(is_utc=True)]]}], '_REGISTRY_CALLBACK': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'Function': [28, ['pointer', ['void']]]}], '_DBGPRINT_CALLBACK': [20, {'Function': [8, ['pointer', ['void']]]}], '_NOTIFY_ENTRY_HEADER': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'EventCategory': [8, ['Enumeration', dict(target='long', choices={0: 'EventCategoryReserved', 1: 'EventCategoryHardwareProfileChange', 2: 'EventCategoryDeviceInterfaceChange', 3: 'EventCategoryTargetDeviceChange'})]], 'CallbackRoutine': [20, ['unsigned int']], 'DriverObject': [28, ['pointer', ['_DRIVER_OBJECT']]]}]}
callback_types_x64 = {'_GENERIC_CALLBACK': [24, {'Callback': [8, ['pointer', ['void']]], 'Associated': [16, ['pointer', ['void']]]}], '_NOTIFICATION_PACKET': [48, {'ListEntry': [0, ['_LIST_ENTRY']], 'DriverObject': [16, ['pointer', ['_DRIVER_OBJECT']]], 'NotificationRoutine': [24, ['address']]}], '_SHUTDOWN_PACKET': [12, {'Entry': [0, ['_LIST_ENTRY']], 'DeviceObject': [16, ['pointer', ['_DEVICE_OBJECT']]]}], '_DBGPRINT_CALLBACK': [20, {'Function': [16, ['pointer', ['void']]]}], '_NOTIFY_ENTRY_HEADER': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'EventCategory': [16, ['Enumeration', dict(target='long', choices={0: 'EventCategoryReserved', 1: 'EventCategoryHardwareProfileChange', 2: 'EventCategoryDeviceInterfaceChange', 3: 'EventCategoryTargetDeviceChange'})]], 'CallbackRoutine': [32, ['address']], 'DriverObject': [48, ['pointer', ['_DRIVER_OBJECT']]]}], '_REGISTRY_CALLBACK': [80, {'ListEntry': [0, ['_LIST_ENTRY']], 'Function': [32, ['pointer', ['void']]]}], '_KBUGCHECK_CALLBACK_RECORD': [None, {'Entry': [0, ['_LIST_ENTRY']], 'CallbackRoutine': [16, ['address']], 'Component': [40, ['pointer', ['String', dict(length=8)]]]}], '_KBUGCHECK_REASON_CALLBACK_RECORD': [None, {'Entry': [0, ['_LIST_ENTRY']], 'CallbackRoutine': [16, ['unsigned int']], 'Component': [40, ['pointer', ['String', dict(length=8)]]]}]}

class _SHUTDOWN_PACKET(obj.CType):
    """Class for shutdown notification callbacks"""

    def is_valid(self):
        if False:
            i = 10
            return i + 15
        '\n        Perform some checks. \n        Note: obj_native_vm is kernel space.\n        '
        if not obj.CType.is_valid(self):
            return False
        if not self.obj_native_vm.is_valid_address(self.Entry.Flink) or not self.obj_native_vm.is_valid_address(self.Entry.Blink) or (not self.obj_native_vm.is_valid_address(self.DeviceObject)):
            return False
        device = self.DeviceObject.dereference()
        object_header = obj.Object('_OBJECT_HEADER', offset=device.obj_offset - self.obj_native_vm.profile.get_obj_offset('_OBJECT_HEADER', 'Body'), vm=device.obj_vm, native_vm=device.obj_native_vm)
        return object_header.get_object_type() == 'Device'

class CallbackMods(obj.ProfileModification):
    before = ['WindowsOverlay']
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        if profile.metadata.get('memory_model', '32bit') == '32bit':
            profile.vtypes.update(callback_types)
            profile.object_classes.update({'_SHUTDOWN_PACKET': _SHUTDOWN_PACKET})
        else:
            profile.vtypes.update(callback_types_x64)

class AbstractCallbackScanner(poolscan.PoolScanner):
    """Return the offset of the callback, no object headers"""

class PoolScanFSCallback(AbstractCallbackScanner):
    """PoolScanner for File System Callbacks"""

    def __init__(self, address_space):
        if False:
            print('Hello World!')
        AbstractCallbackScanner.__init__(self, address_space)
        self.pooltag = 'IoFs'
        self.struct_name = '_NOTIFICATION_PACKET'
        if address_space.profile.metadata.get('memory_model', '32bit') == '32bit':
            size = 24
        else:
            size = 48
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x == size)), ('CheckPoolType', dict(non_paged=True, paged=True, free=True))]

class PoolScanShutdownCallback(AbstractCallbackScanner):
    """PoolScanner for Shutdown Callbacks"""

    def __init__(self, address_space):
        if False:
            return 10
        AbstractCallbackScanner.__init__(self, address_space)
        self.pooltag = 'IoSh'
        self.struct_name = '_SHUTDOWN_PACKET'
        if address_space.profile.metadata.get('memory_model', '32bit') == '32bit':
            size = 24
        else:
            size = 48
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x == size)), ('CheckPoolType', dict(non_paged=True, paged=True, free=True)), ('CheckPoolIndex', dict(value=0))]

class PoolScanGenericCallback(AbstractCallbackScanner):
    """PoolScanner for Generic Callbacks"""

    def __init__(self, address_space):
        if False:
            for i in range(10):
                print('nop')
        AbstractCallbackScanner.__init__(self, address_space)
        self.pooltag = 'Cbrb'
        self.struct_name = '_GENERIC_CALLBACK'
        if address_space.profile.metadata.get('memory_model', '32bit') == '32bit':
            size = 24
        else:
            size = 48
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x == size)), ('CheckPoolType', dict(non_paged=True, paged=True, free=True))]

class PoolScanDbgPrintCallback(AbstractCallbackScanner):
    """PoolScanner for DebugPrint Callbacks on Vista and 7"""

    def __init__(self, address_space):
        if False:
            print('Hello World!')
        AbstractCallbackScanner.__init__(self, address_space)
        self.pooltag = 'DbCb'
        self.struct_name = '_DBGPRINT_CALLBACK'
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= 32 and x <= 64)), ('CheckPoolType', dict(non_paged=True, paged=True, free=True))]

class PoolScanRegistryCallback(AbstractCallbackScanner):
    """PoolScanner for DebugPrint Callbacks on Vista and 7"""

    def __init__(self, address_space):
        if False:
            for i in range(10):
                print('nop')
        AbstractCallbackScanner.__init__(self, address_space)
        self.pooltag = 'CMcb'
        self.struct_name = '_REGISTRY_CALLBACK'
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= 56)), ('CheckPoolType', dict(non_paged=True, paged=True, free=True)), ('CheckPoolIndex', dict(value=4))]

class PoolScanPnp9(AbstractCallbackScanner):
    """PoolScanner for Pnp9 (EventCategoryHardwareProfileChange)"""

    def __init__(self, address_space):
        if False:
            print('Hello World!')
        AbstractCallbackScanner.__init__(self, address_space)
        self.pooltag = 'Pnp9'
        self.struct_name = '_NOTIFY_ENTRY_HEADER'
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= 48)), ('CheckPoolType', dict(non_paged=True, paged=True, free=True)), ('CheckPoolIndex', dict(value=1))]

class PoolScanPnpD(AbstractCallbackScanner):
    """PoolScanner for PnpD (EventCategoryDeviceInterfaceChange)"""

    def __init__(self, address_space):
        if False:
            while True:
                i = 10
        AbstractCallbackScanner.__init__(self, address_space)
        self.pooltag = 'PnpD'
        self.struct_name = '_NOTIFY_ENTRY_HEADER'
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= 64)), ('CheckPoolType', dict(non_paged=True, paged=True, free=True)), ('CheckPoolIndex', dict(value=1))]

class PoolScanPnpC(AbstractCallbackScanner):
    """PoolScanner for PnpC (EventCategoryTargetDeviceChange)"""

    def __init__(self, address_space):
        if False:
            while True:
                i = 10
        AbstractCallbackScanner.__init__(self, address_space)
        self.pooltag = 'PnpC'
        self.struct_name = '_NOTIFY_ENTRY_HEADER'
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= 56)), ('CheckPoolType', dict(non_paged=True, paged=True, free=True)), ('CheckPoolIndex', dict(value=1))]

class Callbacks(common.AbstractScanCommand):
    """Print system-wide notification routines"""
    scanners = [PoolScanFSCallback, PoolScanShutdownCallback, PoolScanGenericCallback]

    @staticmethod
    def get_kernel_callbacks(nt_mod):
        if False:
            return 10
        '\n        Enumerate the Create Process, Create Thread, and Image Load callbacks.\n\n        On some systems, the byte sequences will be inaccurate or the exported \n        function will not be found. In these cases, the PoolScanGenericCallback\n        scanner will pick up the pool associated with the callbacks.\n        '
        bits32 = nt_mod.obj_vm.profile.metadata.get('memory_model', '32bit') == '32bit'
        vista_or_later = nt_mod.obj_vm.profile.metadata.get('major', 0) >= 6
        if bits32:
            routines = [('PsSetLoadImageNotifyRoutine', 'V¾'), ('PsSetCreateThreadNotifyRoutine', 'V¾'), ('PsSetCreateProcessNotifyRoutine', '¿')]
        else:
            routines = [('PsRemoveLoadImageNotifyRoutine', 'H\x8d\r'), ('PsRemoveCreateThreadNotifyRoutine', 'H\x8d\r')]
        for (symbol, hexbytes) in routines:
            symbol_rva = nt_mod.getprocaddress(symbol)
            if symbol_rva == None:
                continue
            symbol_address = symbol_rva + nt_mod.DllBase
            data = nt_mod.obj_vm.zread(symbol_address, 100)
            offset = data.find(hexbytes)
            if offset == -1:
                continue
            if bits32:
                p = obj.Object('Pointer', offset=symbol_address + offset + len(hexbytes), vm=nt_mod.obj_vm)
            else:
                v = obj.Object('int', offset=symbol_address + offset + len(hexbytes), vm=nt_mod.obj_vm)
                p = symbol_address + offset + 7 + v
            if vista_or_later and ('CreateProcess' in symbol or 'CreateThread' in symbol):
                count = 64
            else:
                count = 8
            addrs = obj.Object('Array', count=8, targetType='_EX_FAST_REF', offset=p, vm=nt_mod.obj_vm)
            for addr in addrs:
                callback = addr.dereference_as('_GENERIC_CALLBACK')
                if callback:
                    yield (symbol, callback.Callback, None)

    @staticmethod
    def get_bugcheck_callbacks(addr_space):
        if False:
            i = 10
            return i + 15
        "\n        Enumerate generic Bugcheck callbacks.\n\n        Note: These structures don't exist in tagged pools, but you can find \n        them via KDDEBUGGER_DATA64 on all versions of Windows.\n        "
        kdbg = tasks.get_kdbg(addr_space)
        list_head = kdbg.KeBugCheckCallbackListHead.dereference_as('_KBUGCHECK_CALLBACK_RECORD')
        for l in list_head.Entry.list_of_type('_KBUGCHECK_CALLBACK_RECORD', 'Entry'):
            yield ('KeBugCheckCallbackListHead', l.CallbackRoutine, l.Component.dereference())

    @staticmethod
    def get_registry_callbacks_legacy(nt_mod):
        if False:
            print('Hello World!')
        '\n        Enumerate registry change callbacks.\n\n        This method of finding a global variable via disassembly of the \n        CmRegisterCallback function is only for XP systems. If it fails on \n        XP you can still find the callbacks using PoolScanGenericCallback. \n\n        On Vista and Windows 7, these callbacks are registered using the \n        CmRegisterCallbackEx function. \n        '
        if not has_distorm3:
            return
        symbol = 'CmRegisterCallback'
        symbol_rva = nt_mod.getprocaddress(symbol)
        if symbol_rva == None:
            return
        symbol_address = symbol_rva + nt_mod.DllBase
        data = nt_mod.obj_vm.zread(symbol_address, 200)
        c = 0
        vector = None
        for op in distorm3.Decompose(symbol_address, data, distorm3.Decode32Bits):
            if op.valid and op.mnemonic == 'MOV' and (len(op.operands) == 2) and (op.operands[0].name == 'EBX'):
                vector = op.operands[1].value
                if c == 1:
                    break
                else:
                    c += 1
        if vector == None:
            return
        addrs = obj.Object('Array', count=100, offset=vector, vm=nt_mod.obj_vm, targetType='_EX_FAST_REF')
        for addr in addrs:
            callback = addr.dereference_as('_EX_CALLBACK_ROUTINE_BLOCK')
            if callback:
                yield (symbol, callback.Function, None)

    @staticmethod
    def get_bugcheck_reason_callbacks(nt_mod):
        if False:
            while True:
                i = 10
        "\n        Enumerate Bugcheck Reason callbacks.\n\n        Note: These structures don't exist in tagged pools, so we \n        find them by locating the list head which is a non-exported \n        NT symbol. The method works on all x86 versions of Windows. \n\n        mov [eax+KBUGCHECK_REASON_CALLBACK_RECORD.Entry.Blink],                 offset _KeBugCheckReasonCallbackListHead\n        "
        symbol = 'KeRegisterBugCheckReasonCallback'
        bits32 = nt_mod.obj_vm.profile.metadata.get('memory_model', '32bit') == '32bit'
        if bits32:
            hexbytes = 'Ç@\x04'
        else:
            hexbytes = 'H\x8d\r'
        symbol_rva = nt_mod.getprocaddress(symbol)
        if symbol_rva == None:
            return
        symbol_address = symbol_rva + nt_mod.DllBase
        data = nt_mod.obj_vm.zread(symbol_address, 200)
        offset = data.find(hexbytes)
        if offset == -1:
            return
        if bits32:
            p = obj.Object('Pointer', offset=symbol_address + offset + len(hexbytes), vm=nt_mod.obj_vm)
            bugs = p.dereference_as('_KBUGCHECK_REASON_CALLBACK_RECORD')
        else:
            v = obj.Object('int', offset=symbol_address + offset + len(hexbytes), vm=nt_mod.obj_vm)
            p = symbol_address + offset + 7 + v
            bugs = obj.Object('_KBUGCHECK_REASON_CALLBACK_RECORD', offset=p, vm=nt_mod.obj_vm)
        for l in bugs.Entry.list_of_type('_KBUGCHECK_REASON_CALLBACK_RECORD', 'Entry'):
            if nt_mod.obj_vm.is_valid_address(l.CallbackRoutine):
                yield (symbol, l.CallbackRoutine, l.Component.dereference())

    def calculate(self):
        if False:
            return 10
        addr_space = utils.load_as(self._config)
        bits32 = addr_space.profile.metadata.get('memory_model', '32bit') == '32bit'
        version = (addr_space.profile.metadata.get('major', 0), addr_space.profile.metadata.get('minor', 0))
        modlist = list(modules.lsmod(addr_space))
        mods = dict(((addr_space.address_mask(mod.DllBase), mod) for mod in modlist))
        mod_addrs = sorted(mods.keys())
        if version >= (6, 0):
            self.scanners.append(PoolScanDbgPrintCallback)
            self.scanners.append(PoolScanRegistryCallback)
            self.scanners.append(PoolScanPnp9)
            self.scanners.append(PoolScanPnpD)
            self.scanners.append(PoolScanPnpC)
        for objct in self.scan_results(addr_space):
            name = objct.obj_name
            if name == '_REGISTRY_CALLBACK':
                info = ('CmRegisterCallback', objct.Function, None)
                yield (info, mods, mod_addrs)
            elif name == '_DBGPRINT_CALLBACK':
                info = ('DbgSetDebugPrintCallback', objct.Function, None)
                yield (info, mods, mod_addrs)
            elif name == '_SHUTDOWN_PACKET':
                driver = objct.DeviceObject.dereference().DriverObject
                if not driver:
                    continue
                index = devicetree.MAJOR_FUNCTIONS.index('IRP_MJ_SHUTDOWN')
                address = driver.MajorFunction[index]
                details = str(driver.DriverName or '-')
                info = ('IoRegisterShutdownNotification', address, details)
                yield (info, mods, mod_addrs)
            elif name == '_GENERIC_CALLBACK':
                info = ('GenericKernelCallback', objct.Callback, None)
                yield (info, mods, mod_addrs)
            elif name == '_NOTIFY_ENTRY_HEADER':
                driver = objct.DriverObject.dereference()
                driver_name = ''
                if driver:
                    header = driver.get_object_header()
                    if header.get_object_type() == 'Driver':
                        driver_name = header.NameInfo.Name.v()
                info = (objct.EventCategory, objct.CallbackRoutine, driver_name)
                yield (info, mods, mod_addrs)
            elif name == '_NOTIFICATION_PACKET':
                info = ('IoRegisterFsRegistrationChange', objct.NotificationRoutine, None)
                yield (info, mods, mod_addrs)
        for info in self.get_kernel_callbacks(modlist[0]):
            yield (info, mods, mod_addrs)
        for info in self.get_bugcheck_callbacks(addr_space):
            yield (info, mods, mod_addrs)
        for info in self.get_bugcheck_reason_callbacks(modlist[0]):
            yield (info, mods, mod_addrs)
        if bits32 and version == (5, 1):
            for info in self.get_registry_callbacks_legacy(modlist[0]):
                yield (info, mods, mod_addrs)

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('Type', str), ('Callback', Address), ('Module', str), ('Details', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for ((sym, cb, detail), mods, mod_addrs) in data:
            module = tasks.find_module(mods, mod_addrs, mods.values()[0].obj_vm.address_mask(cb))
            if module:
                module_name = module.BaseDllName or module.FullDllName
            else:
                module_name = 'UNKNOWN'
            yield (0, [str(sym), Address(cb), str(module_name), str(detail or '-')])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Type', '36'), ('Callback', '[addrpad]'), ('Module', '20'), ('Details', '')])
        for ((sym, cb, detail), mods, mod_addrs) in data:
            module = tasks.find_module(mods, mod_addrs, mods.values()[0].obj_vm.address_mask(cb))
            if module:
                module_name = module.BaseDllName or module.FullDllName
            else:
                module_name = 'UNKNOWN'
            self.table_row(outfd, sym, cb, module_name, detail or '-')