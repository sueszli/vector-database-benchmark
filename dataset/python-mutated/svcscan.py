import volatility.utils as utils
import volatility.obj as obj
import volatility.plugins.common as common
import volatility.win32.tasks as tasks
import volatility.debug as debug
import volatility.plugins.registry.registryapi as registryapi
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address
SERVICE_TYPE_FLAGS = {'SERVICE_KERNEL_DRIVER': 0, 'SERVICE_FILE_SYSTEM_DRIVER': 1, 'SERVICE_WIN32_OWN_PROCESS': 4, 'SERVICE_WIN32_SHARE_PROCESS': 5, 'SERVICE_INTERACTIVE_PROCESS': 8}
SERVICE_STATE_ENUM = {1: 'SERVICE_STOPPED', 2: 'SERVICE_START_PENDING', 3: 'SERVICE_STOP_PENDING', 4: 'SERVICE_RUNNING', 5: 'SERVICE_CONTINUE_PENDING', 6: 'SERVICE_PAUSE_PENDING', 7: 'SERVICE_PAUSED'}
SERVICE_START_ENUM = {0: 'SERVICE_BOOT_START', 1: 'SERVICE_SYSTEM_START', 2: 'SERVICE_AUTO_START', 3: 'SERVICE_DEMAND_START', 4: 'SERVICE_DISABLED'}
svcscan_base_x86 = {'_SERVICE_HEADER': [None, {'Tag': [0, ['array', 4, ['unsigned char']]], 'ServiceRecord': [12, ['pointer', ['_SERVICE_RECORD']]]}], '_SERVICE_LIST_ENTRY': [8, {'Blink': [0, ['pointer', ['_SERVICE_RECORD']]], 'Flink': [4, ['pointer', ['_SERVICE_RECORD']]]}], '_SERVICE_RECORD': [None, {'ServiceList': [0, ['_SERVICE_LIST_ENTRY']], 'ServiceName': [8, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [12, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [16, ['unsigned int']], 'Tag': [24, ['array', 4, ['unsigned char']]], 'DriverName': [36, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [36, ['pointer', ['_SERVICE_PROCESS']]], 'Type': [40, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [44, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [68, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]]}], '_SERVICE_PROCESS': [None, {'BinaryPath': [8, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ProcessId': [12, ['unsigned int']]}]}
svcscan_base_x64 = {'_SERVICE_HEADER': [None, {'Tag': [0, ['array', 4, ['unsigned char']]], 'ServiceRecord': [16, ['pointer', ['_SERVICE_RECORD']]]}], '_SERVICE_LIST_ENTRY': [8, {'Blink': [0, ['pointer', ['_SERVICE_RECORD']]], 'Flink': [16, ['pointer', ['_SERVICE_RECORD']]]}], '_SERVICE_RECORD': [None, {'ServiceList': [0, ['_SERVICE_LIST_ENTRY']], 'ServiceName': [8, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [16, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [24, ['unsigned int']], 'Tag': [32, ['array', 4, ['unsigned char']]], 'DriverName': [48, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [48, ['pointer', ['_SERVICE_PROCESS']]], 'Type': [56, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [60, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [84, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]]}], '_SERVICE_PROCESS': [None, {'BinaryPath': [16, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ProcessId': [24, ['unsigned int']]}]}

class _SERVICE_RECORD_LEGACY(obj.CType):
    """Service records for XP/2003 x86 and x64"""

    @property
    def Binary(self):
        if False:
            print('Hello World!')
        'Return the binary path for a service'
        if str(self.State) != 'SERVICE_RUNNING':
            return obj.NoneObject("No path, service isn't running")
        if 'PROCESS' in str(self.Type):
            return self.ServiceProcess.BinaryPath.dereference()
        else:
            return self.DriverName.dereference()

    @property
    def Pid(self):
        if False:
            return 10
        'Return the process ID for a service'
        if str(self.State) == 'SERVICE_RUNNING':
            if 'PROCESS' in str(self.Type):
                return self.ServiceProcess.ProcessId
        return obj.NoneObject('Cannot get process ID')

    def is_valid(self):
        if False:
            for i in range(10):
                print('nop')
        'Check some fields for validity'
        type_flags_max = sum([1 << v for v in SERVICE_TYPE_FLAGS.values()])
        return obj.CType.is_valid(self) and self.Order > 0 and (self.Order < 65535) and (self.State.v() in SERVICE_STATE_ENUM) and (self.Start.v() in SERVICE_START_ENUM) and (self.Type.v() < type_flags_max)

    def traverse(self):
        if False:
            while True:
                i = 10
        rec = self
        while rec and rec.is_valid():
            yield rec
            rec = rec.ServiceList.Blink.dereference()

class _SERVICE_RECORD_RECENT(_SERVICE_RECORD_LEGACY):
    """Service records for 2008, Vista, 7 x86 and x64"""

    def traverse(self):
        if False:
            return 10
        'Generator that walks the singly-linked list'
        if self.is_valid():
            yield self
        rec = self.PrevEntry.dereference()
        while rec and rec.is_valid():
            yield rec
            rec = rec.PrevEntry.dereference()

class _SERVICE_HEADER(obj.CType):
    """Service headers for 2008, Vista, 7 x86 and x64"""

    def is_valid(self):
        if False:
            for i in range(10):
                print('nop')
        'Check some fields for validity'
        return obj.CType.is_valid(self) and self.ServiceRecord.is_valid() and (self.ServiceRecord.Order < 65535)

class ServiceBase(obj.ProfileModification):
    """The base applies to XP and 2003 SP0-SP1"""
    before = ['WindowsOverlay', 'WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.object_classes.update({'_SERVICE_RECORD': _SERVICE_RECORD_LEGACY, '_SERVICE_HEADER': _SERVICE_HEADER})
        profile.merge_overlay({'VOLATILITY_MAGIC': [None, {'ServiceTag': [0, ['VolatilityMagic', dict(value='sErv')]]}]})
        profile.vtypes.update(svcscan_base_x86)

class ServiceBasex64(obj.ProfileModification):
    """This overrides the base x86 vtypes with x64 vtypes"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            return 10
        profile.vtypes.update(svcscan_base_x64)

class ServiceVista(obj.ProfileModification):
    """Override the base with OC's for Vista, 2008, and 7"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x >= 6}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.object_classes.update({'_SERVICE_RECORD': _SERVICE_RECORD_RECENT})
        profile.merge_overlay({'VOLATILITY_MAGIC': [None, {'ServiceTag': [0, ['VolatilityMagic', dict(value='serH')]]}]})

class ServiceVistax86(obj.ProfileModification):
    """Override the base with vtypes for x86 Vista, 2008, and 7"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x < 2, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.merge_overlay({'_SERVICE_RECORD': [None, {'PrevEntry': [0, ['pointer', ['_SERVICE_RECORD']]], 'ServiceName': [4, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [8, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [12, ['unsigned int']], 'ServiceProcess': [28, ['pointer', ['_SERVICE_PROCESS']]], 'DriverName': [28, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'Type': [32, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [36, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [60, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]], 'ServiceList': [92, ['_SERVICE_LIST_ENTRY']]}]})

class ServiceVistax64(obj.ProfileModification):
    """Override the base with vtypes for x64 Vista, 2008, and 7"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x < 2, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            return 10
        profile.merge_overlay({'_SERVICE_RECORD': [None, {'PrevEntry': [0, ['pointer', ['_SERVICE_RECORD']]], 'ServiceName': [8, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [16, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [24, ['unsigned int']], 'ServiceProcess': [40, ['pointer', ['_SERVICE_PROCESS']]], 'DriverName': [40, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'Type': [48, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [52, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [76, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]], 'ServiceList': [120, ['_SERVICE_LIST_ENTRY']]}]})

class Service8x64(obj.ProfileModification):
    """Service structures for Win8/8.1 and Server2012/R2 64-bit"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x >= 2, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.merge_overlay({'_SERVICE_RECORD': [None, {'Tag': [0, ['String', dict(length=4)]], 'PrevEntry': [8, ['pointer', ['_SERVICE_RECORD']]], 'ServiceName': [16, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [24, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [32, ['unsigned int']], 'DriverName': [56, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [56, ['pointer', ['_SERVICE_PROCESS']]], 'Type': [64, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [68, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [92, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]]}], '_SERVICE_PROCESS': [None, {'BinaryPath': [24, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ProcessId': [32, ['unsigned int']]}]})

class Service10_15063x64(obj.ProfileModification):
    """Service structures for Win10 15063 (Creators)"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista', 'Service8x64']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 15063, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            return 10
        profile.merge_overlay({'_SERVICE_RECORD': [None, {'PrevEntry': [16, ['pointer', ['_SERVICE_RECORD']]], 'ServiceName': [56, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [64, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [32, ['unsigned int']], 'DriverName': [232, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [232, ['pointer', ['_SERVICE_PROCESS']]], 'Type': [72, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [76, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [36, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]]}]})

class Service10_16299x64(obj.ProfileModification):
    """Service structures for Win10 16299 (Fall Creators)"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista', 'Service8x64', 'Service10_15063x64']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 16299, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.merge_overlay({'_SERVICE_PROCESS': [None, {'BinaryPath': [24, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ProcessId': [40, ['unsigned int']]}]})

class Service10_18362x64(obj.ProfileModification):
    """Service structures for Win10 18362 (May 2019)"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista', 'Service8x64', 'Service10_15063x64', 'Service10_16299x64']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 18362, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.merge_overlay({'_SERVICE_RECORD': [None, {'PrevEntry': [16, ['pointer', ['_SERVICE_RECORD']]], 'ServiceName': [56, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [64, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [32, ['unsigned int']], 'DriverName': [240, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [240, ['pointer', ['_SERVICE_PROCESS']]], 'Type': [72, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [76, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [36, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]]}]})

class Service10_19041x64(obj.ProfileModification):
    """Service structures for Win10 19041 (May 2020)"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista', 'Service8x64', 'Service10_15063x64', 'Service10_16299x64', 'Service10_18362x64']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 19041, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.merge_overlay({'_SERVICE_RECORD': [None, {'DriverName': [296, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [296, ['pointer', ['_SERVICE_PROCESS']]]}]})

class Service8x86(obj.ProfileModification):
    """Service structures for Win8/8.1 32-bit"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x >= 2, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.vtypes.update({'_SERVICE_RECORD': [None, {'Tag': [0, ['String', dict(length=4)]], 'PrevEntry': [4, ['pointer', ['_SERVICE_RECORD']]], 'ServiceName': [8, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [12, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [16, ['unsigned int']], 'DriverName': [36, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [36, ['pointer', ['_SERVICE_PROCESS']]], 'Type': [40, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [44, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [68, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]]}], '_SERVICE_PROCESS': [None, {'BinaryPath': [12, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ProcessId': [16, ['unsigned int']]}]})

class Service10_15063x86(obj.ProfileModification):
    """Service structures for Win10 15063 (Creators)"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista', 'Service8x86']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 15063, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update({'_SERVICE_RECORD': [None, {'PrevEntry': [12, ['pointer', ['_SERVICE_RECORD']]], 'ServiceName': [44, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [48, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [20, ['unsigned int']], 'DriverName': [156, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [156, ['pointer', ['_SERVICE_PROCESS']]], 'Type': [52, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [56, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [24, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]]}]})

class Service10_16299x86(obj.ProfileModification):
    """Service structures for Win10 16299 (Fall Creators)"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista', 'Service8x86', 'Service10_15063x86']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 16299, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.merge_overlay({'_SERVICE_PROCESS': [None, {'BinaryPath': [12, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ProcessId': [20, ['unsigned int']]}]})

class Service10_17763x86(obj.ProfileModification):
    """Service structures for Win10 17763 (October 2018)"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista', 'Service8x86', 'Service10_15063x86', 'Service10_16299x86']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 17763, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update({'_SERVICE_RECORD': [None, {'PrevEntry': [12, ['pointer', ['_SERVICE_RECORD']]], 'ServiceName': [44, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [48, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [20, ['unsigned int']], 'DriverName': [160, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [160, ['pointer', ['_SERVICE_PROCESS']]], 'Type': [52, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [56, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [24, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]]}]})

class Service10_18362x86(obj.ProfileModification):
    """Service structures for Win10 18362 (May 2019)"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista', 'Service8x86', 'Service10_15063x86', 'Service10_16299x86', 'Service10_17763x86']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 18362, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            return 10
        profile.vtypes.update({'_SERVICE_RECORD': [None, {'PrevEntry': [12, ['pointer', ['_SERVICE_RECORD']]], 'ServiceName': [44, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'DisplayName': [48, ['pointer', ['String', dict(encoding='utf16', length=512)]]], 'Order': [20, ['unsigned int']], 'DriverName': [164, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [164, ['pointer', ['_SERVICE_PROCESS']]], 'Type': [52, ['Flags', {'bitmap': SERVICE_TYPE_FLAGS}]], 'State': [56, ['Enumeration', dict(target='long', choices=SERVICE_STATE_ENUM)]], 'Start': [24, ['Enumeration', dict(target='long', choices=SERVICE_START_ENUM)]]}]})

class Service10_19041x86(obj.ProfileModification):
    """Service structures for Win10 19041 (May 2020)"""
    before = ['WindowsOverlay', 'WindowsObjectClasses', 'ServiceBase', 'ServiceVista', 'Service8x86', 'Service10_15063x86', 'Service10_16299x86', 'Service10_17763x86', 'Service10_18362x86']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 19041, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.merge_overlay({'_SERVICE_HEADER': [None, {'ServiceRecord': [16, ['pointer', ['_SERVICE_RECORD']]]}], '_SERVICE_RECORD': [None, {'DriverName': [192, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'ServiceProcess': [192, ['pointer', ['_SERVICE_PROCESS']]]}]})

class SvcScan(common.AbstractWindowsCommand):
    """Scan for Windows services"""

    def calculate(self):
        if False:
            i = 10
            return i + 15
        addr_space = utils.load_as(self._config)
        version = (addr_space.profile.metadata.get('major', 0), addr_space.profile.metadata.get('minor', 0))
        tag = obj.VolMagic(addr_space).ServiceTag.v()
        records = []
        for task in tasks.pslist(addr_space):
            if str(task.ImageFileName).lower() != 'services.exe':
                continue
            process_space = task.get_process_address_space()
            if process_space == None:
                continue
            for address in task.search_process_memory([tag], vad_filter=lambda x: x.Length < 1073741824):
                if version <= (5, 2):
                    rec = obj.Object('_SERVICE_RECORD', offset=address - addr_space.profile.get_obj_offset('_SERVICE_RECORD', 'Tag'), vm=process_space)
                    if rec.is_valid():
                        yield rec
                else:
                    svc_hdr = obj.Object('_SERVICE_HEADER', offset=address, vm=process_space)
                    if svc_hdr.is_valid():
                        for rec in svc_hdr.ServiceRecord.traverse():
                            if rec in records:
                                break
                            records.append(rec)
                            yield rec

    def render_dot(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        'Generate a dot graph of service relationships. \n\n        This currently only works for XP/2003 profiles, \n        because the linked list was removed after that.\n        '
        all_services = [d for d in data]
        if all_services[0].obj_vm.profile.metadata.get('major', 0) != 5:
            debug.error('This profile does not support --output=dot format')
        objects = set()
        links = set()
        for svc in all_services:
            label = '{{ {0:#x} \\n {1} \\n {2} \\n F:{3:#x} B:{4:#x} }}'.format(svc.obj_offset, svc.ServiceName.dereference(), str(svc.State), svc.ServiceList.Flink.v(), svc.ServiceList.Blink.v())
            objects.add('"{0:#x}" [label="{1}" shape="record"];\n'.format(svc.obj_offset, label))
            flink = svc.ServiceList.Flink.dereference()
            blink = svc.ServiceList.Blink.dereference()
            if flink.is_valid():
                links.add('"{0:#x}" -> "{1:#x}" [];\n'.format(svc.obj_offset, flink.obj_offset))
            if blink.is_valid():
                links.add('"{0:#x}" -> "{1:#x}" [];\n'.format(svc.obj_offset, blink.obj_offset))
        outfd.write('digraph svctree { \ngraph [rankdir = "TB"];\n')
        for item in objects:
            outfd.write(item)
        for link in links:
            outfd.write(link)
        outfd.write('}\n')

    @staticmethod
    def get_service_info(regapi):
        if False:
            for i in range(10):
                print('nop')
        ccs = regapi.reg_get_currentcontrolset()
        key_name = '{0}\\services'.format(ccs)
        info = {}
        for subkey in regapi.reg_get_all_subkeys(hive_name='system', key=key_name):
            path_value = ''
            dll_value = ''
            failure_value = ''
            image_path = regapi.reg_get_value(hive_name='system', key='', value='ImagePath', given_root=subkey)
            if image_path:
                if isinstance(image_path, list):
                    image_path = image_path[0]
                path_value = utils.remove_unprintable(image_path)
            failure_path = regapi.reg_get_value(hive_name='system', key='', value='FailureCommand', given_root=subkey)
            if failure_path:
                failure_value = utils.remove_unprintable(failure_path)
            for rootkey in regapi.reg_get_all_subkeys(hive_name='system', key='', given_root=subkey):
                if rootkey.Name == 'Parameters':
                    service_dll = regapi.reg_get_value(hive_name='system', key='', value='ServiceDll', given_root=rootkey)
                    if service_dll != None:
                        dll_value = utils.remove_unprintable(service_dll)
                    break
            last_write = int(subkey.LastWriteTime)
            info[utils.remove_unprintable(str(subkey.Name))] = (dll_value, path_value, failure_value, last_write)
        return info

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        if self._config.VERBOSE:
            return TreeGrid([('Offset', Address), ('Order', int), ('Start', str), ('PID', int), ('ServiceName', str), ('DisplayName', str), ('ServiceType', str), ('State', str), ('BinaryPath', str), ('ServiceDll', str), ('ImagePath', str), ('FailureCommand', str)], self.generator(data))
        return TreeGrid([('Offset', Address), ('Order', int), ('Start', str), ('PID', int), ('ServiceName', str), ('DisplayName', str), ('ServiceType', str), ('State', str), ('BinaryPath', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        if self._config.VERBOSE:
            regapi = registryapi.RegistryApi(self._config)
            info = self.get_service_info(regapi)
        for rec in data:
            if self._config.VERBOSE:
                vals = info.get('{0}'.format(rec.ServiceName.dereference()), None)
                yield (0, [Address(rec.obj_offset), int(rec.Order), str(rec.Start), int(rec.Pid), str(rec.ServiceName.dereference() or ''), str(rec.DisplayName.dereference() or ''), str(rec.Type), str(rec.State), str(rec.Binary or ''), str(vals[0] if vals else ''), str(vals[1] if vals else ''), str(vals[2] if vals else '')])
            else:
                yield (0, [Address(rec.obj_offset), int(rec.Order), str(rec.Start), int(rec.Pid), str(rec.ServiceName.dereference() or ''), str(rec.DisplayName.dereference() or ''), str(rec.Type), str(rec.State), str(rec.Binary or '')])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        if self._config.VERBOSE:
            regapi = registryapi.RegistryApi(self._config)
            info = self.get_service_info(regapi)
        for rec in data:
            outfd.write('Offset: {0:#x}\n'.format(rec.obj_offset))
            outfd.write('Order: {0}\n'.format(rec.Order))
            outfd.write('Start: {0}\n'.format(rec.Start))
            outfd.write('Process ID: {0}\n'.format(rec.Pid))
            outfd.write('Service Name: {0}\n'.format(rec.ServiceName.dereference()))
            outfd.write('Display Name: {0}\n'.format(rec.DisplayName.dereference()))
            outfd.write('Service Type: {0}\n'.format(rec.Type))
            outfd.write('Service State: {0}\n'.format(rec.State))
            outfd.write('Binary Path: {0}\n'.format(rec.Binary))
            if self._config.VERBOSE:
                vals = info.get('{0}'.format(rec.ServiceName.dereference()), None)
                if vals:
                    outfd.write('ServiceDll: {0}\n'.format(vals[0]))
                    outfd.write('ImagePath: {0}\n'.format(vals[1]))
                    outfd.write('FailureCommand: {0}\n'.format(vals[2]))
            outfd.write('\n')