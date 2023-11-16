import re
import volatility.obj as obj
import volatility.plugins.filescan as filescan
import volatility.win32.modules as modules
import volatility.win32.tasks as tasks
import volatility.utils as utils
import volatility.plugins.malware.malfind as malfind
import volatility.plugins.overlays.windows.windows as windows
MAJOR_FUNCTIONS = ['IRP_MJ_CREATE', 'IRP_MJ_CREATE_NAMED_PIPE', 'IRP_MJ_CLOSE', 'IRP_MJ_READ', 'IRP_MJ_WRITE', 'IRP_MJ_QUERY_INFORMATION', 'IRP_MJ_SET_INFORMATION', 'IRP_MJ_QUERY_EA', 'IRP_MJ_SET_EA', 'IRP_MJ_FLUSH_BUFFERS', 'IRP_MJ_QUERY_VOLUME_INFORMATION', 'IRP_MJ_SET_VOLUME_INFORMATION', 'IRP_MJ_DIRECTORY_CONTROL', 'IRP_MJ_FILE_SYSTEM_CONTROL', 'IRP_MJ_DEVICE_CONTROL', 'IRP_MJ_INTERNAL_DEVICE_CONTROL', 'IRP_MJ_SHUTDOWN', 'IRP_MJ_LOCK_CONTROL', 'IRP_MJ_CLEANUP', 'IRP_MJ_CREATE_MAILSLOT', 'IRP_MJ_QUERY_SECURITY', 'IRP_MJ_SET_SECURITY', 'IRP_MJ_POWER', 'IRP_MJ_SYSTEM_CONTROL', 'IRP_MJ_DEVICE_CHANGE', 'IRP_MJ_QUERY_QUOTA', 'IRP_MJ_SET_QUOTA', 'IRP_MJ_PNP']
DEVICE_CODES = {39: 'FILE_DEVICE_8042_PORT', 50: 'FILE_DEVICE_ACPI', 41: 'FILE_DEVICE_BATTERY', 1: 'FILE_DEVICE_BEEP', 42: 'FILE_DEVICE_BUS_EXTENDER', 2: 'FILE_DEVICE_CD_ROM', 3: 'FILE_DEVICE_CD_ROM_FILE_SYSTEM', 48: 'FILE_DEVICE_CHANGER', 4: 'FILE_DEVICE_CONTROLLER', 5: 'FILE_DEVICE_DATALINK', 6: 'FILE_DEVICE_DFS', 53: 'FILE_DEVICE_DFS_FILE_SYSTEM', 54: 'FILE_DEVICE_DFS_VOLUME', 7: 'FILE_DEVICE_DISK', 8: 'FILE_DEVICE_DISK_FILE_SYSTEM', 51: 'FILE_DEVICE_DVD', 9: 'FILE_DEVICE_FILE_SYSTEM', 58: 'FILE_DEVICE_FIPS', 52: 'FILE_DEVICE_FULLSCREEN_VIDEO', 10: 'FILE_DEVICE_INPORT_PORT', 11: 'FILE_DEVICE_KEYBOARD', 47: 'FILE_DEVICE_KS', 57: 'FILE_DEVICE_KSEC', 12: 'FILE_DEVICE_MAILSLOT', 45: 'FILE_DEVICE_MASS_STORAGE', 13: 'FILE_DEVICE_MIDI_IN', 14: 'FILE_DEVICE_MIDI_OUT', 43: 'FILE_DEVICE_MODEM', 15: 'FILE_DEVICE_MOUSE', 16: 'FILE_DEVICE_MULTI_UNC_PROVIDER', 17: 'FILE_DEVICE_NAMED_PIPE', 18: 'FILE_DEVICE_NETWORK', 19: 'FILE_DEVICE_NETWORK_BROWSER', 20: 'FILE_DEVICE_NETWORK_FILE_SYSTEM', 40: 'FILE_DEVICE_NETWORK_REDIRECTOR', 21: 'FILE_DEVICE_NULL', 22: 'FILE_DEVICE_PARALLEL_PORT', 23: 'FILE_DEVICE_PHYSICAL_NETCARD', 24: 'FILE_DEVICE_PRINTER', 25: 'FILE_DEVICE_SCANNER', 28: 'FILE_DEVICE_SCREEN', 55: 'FILE_DEVICE_SERENUM', 26: 'FILE_DEVICE_SERIAL_MOUSE_PORT', 27: 'FILE_DEVICE_SERIAL_PORT', 49: 'FILE_DEVICE_SMARTCARD', 46: 'FILE_DEVICE_SMB', 29: 'FILE_DEVICE_SOUND', 30: 'FILE_DEVICE_STREAMS', 31: 'FILE_DEVICE_TAPE', 32: 'FILE_DEVICE_TAPE_FILE_SYSTEM', 56: 'FILE_DEVICE_TERMSRV', 33: 'FILE_DEVICE_TRANSPORT', 34: 'FILE_DEVICE_UNKNOWN', 44: 'FILE_DEVICE_VDM', 35: 'FILE_DEVICE_VIDEO', 36: 'FILE_DEVICE_VIRTUAL_DISK', 37: 'FILE_DEVICE_WAVE_IN', 38: 'FILE_DEVICE_WAVE_OUT'}

class _DRIVER_OBJECT(obj.CType, windows.ExecutiveObjectMixin):
    """Class for driver objects"""

    def devices(self):
        if False:
            while True:
                i = 10
        "Enumerate the driver's device objects"
        device = self.DeviceObject.dereference()
        while device:
            yield device
            device = device.NextDevice.dereference()

    def is_valid(self):
        if False:
            return 10
        return obj.CType.is_valid(self) and self.DriverStart % 4096 == 0

class _DEVICE_OBJECT(obj.CType, windows.ExecutiveObjectMixin):
    """Class for device objects"""

    def attached_devices(self):
        if False:
            for i in range(10):
                print('nop')
        "Enumerate the device's attachees"
        device = self.AttachedDevice.dereference()
        while device:
            yield device
            device = device.AttachedDevice.dereference()

class MalwareDrivers(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.object_classes.update({'_DRIVER_OBJECT': _DRIVER_OBJECT, '_DEVICE_OBJECT': _DEVICE_OBJECT})

class DeviceTree(filescan.DriverScan):
    """Show device tree"""

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        for driver in data:
            header = driver.get_object_header()
            outfd.write('DRV 0x{0:08x} {1}\n'.format(driver.obj_offset, str(driver.DriverName or header.NameInfo.Name or '')))
            for device in driver.devices():
                device_header = obj.Object('_OBJECT_HEADER', offset=device.obj_offset - device.obj_vm.profile.get_obj_offset('_OBJECT_HEADER', 'Body'), vm=device.obj_vm, native_vm=device.obj_native_vm)
                device_name = str(device_header.NameInfo.Name or '')
                outfd.write('---| DEV {0:#x} {1} {2}\n'.format(device.obj_offset, device_name, DEVICE_CODES.get(device.DeviceType.v(), 'UNKNOWN')))
                level = 0
                for att_device in device.attached_devices():
                    device_header = obj.Object('_OBJECT_HEADER', offset=att_device.obj_offset - att_device.obj_vm.profile.get_obj_offset('_OBJECT_HEADER', 'Body'), vm=att_device.obj_vm, native_vm=att_device.obj_native_vm)
                    device_name = str(device_header.NameInfo.Name or '')
                    name = device_name + ' - ' + str(att_device.DriverObject.DriverName or '')
                    outfd.write('------{0}| ATT {1:#x} {2} {3}\n'.format('---' * level, att_device.obj_offset, name, DEVICE_CODES.get(att_device.DeviceType.v(), 'UNKNOWN')))
                    level += 1

class DriverIrp(filescan.DriverScan):
    """Driver IRP hook detection"""

    def __init__(self, config, *args, **kwargs):
        if False:
            while True:
                i = 10
        filescan.DriverScan.__init__(self, config, *args, **kwargs)
        config.add_option('REGEX', short_option='r', type='str', action='store', help='Analyze drivers matching REGEX')

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        addr_space = utils.load_as(self._config)
        if self._config.regex != None:
            mod_re = re.compile(self._config.regex, re.I)
        else:
            mod_re = None
        mods = dict(((addr_space.address_mask(mod.DllBase), mod) for mod in modules.lsmod(addr_space)))
        mod_addrs = sorted(mods.keys())
        bits = addr_space.profile.metadata.get('memory_model', '32bit')
        self.table_header(None, [('i', '>4'), ('Funcs', '36'), ('addr', '[addrpad]'), ('name', '')])
        for driver in data:
            header = driver.get_object_header()
            driver_name = str(header.NameInfo.Name or '')
            if mod_re != None:
                if not (mod_re.search(driver_name) or mod_re.search(driver_name)):
                    continue
            outfd.write('{0}\n'.format('-' * 50))
            outfd.write('DriverName: {0}\n'.format(driver_name))
            outfd.write('DriverStart: {0:#x}\n'.format(driver.DriverStart))
            outfd.write('DriverSize: {0:#x}\n'.format(driver.DriverSize))
            outfd.write('DriverStartIo: {0:#x}\n'.format(driver.DriverStartIo))
            for (i, function) in enumerate(driver.MajorFunction):
                function = driver.MajorFunction[i]
                module = tasks.find_module(mods, mod_addrs, addr_space.address_mask(function))
                if module:
                    module_name = str(module.BaseDllName or '')
                else:
                    module_name = 'Unknown'
                self.table_row(outfd, i, MAJOR_FUNCTIONS[i], function, module_name)
                if self._config.verbose:
                    data = addr_space.zread(function, 64)
                    outfd.write('\n'.join(['{0:#x} {1:<16} {2}'.format(o, h, i) for (o, i, h) in malfind.Disassemble(data=data, start=function, bits=bits, stoponret=True)]))
                    outfd.write('\n')