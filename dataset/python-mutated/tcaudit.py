import os
import volatility.obj as obj
import volatility.debug as debug
import volatility.utils as utils
import volatility.plugins.common as common
import volatility.win32.modules as modules
import volatility.win32.tasks as tasks
import volatility.plugins.filescan as filescan
import volatility.plugins.malware.devicetree as devicetree
import volatility.plugins.malware.svcscan as svcscan
import volatility.plugins.registry.registryapi as registryapi
tc_70a_vtypes_x86 = {'UINT64_STRUCT': [8, {'LowPart': [0, ['unsigned long']], 'HighPart': [4, ['unsigned long']], 'Value': [0, ['unsigned long long']]}], 'CRYPTO_INFO_t': [17512, {'ea': [0, ['long']], 'mode': [4, ['long']], 'ks': [8, ['array', 5324, ['unsigned char']]], 'ks2': [5332, ['array', 5324, ['unsigned char']]], 'hiddenVolume': [10656, ['long']], 'HeaderVersion': [10660, ['unsigned short']], 'gf_ctx': [10664, ['GfCtx']], 'master_keydata': [16808, ['array', 256, ['unsigned char']]], 'k2': [17064, ['array', 256, ['unsigned char']]], 'salt': [17320, ['array', 64, ['unsigned char']]], 'noIterations': [17384, ['long']], 'pkcs5': [17388, ['long']], 'volume_creation_time': [17392, ['unsigned long long']], 'header_creation_time': [17400, ['unsigned long long']], 'bProtectHiddenVolume': [17408, ['long']], 'bHiddenVolProtectionAction': [17412, ['long']], 'volDataAreaOffset': [17416, ['unsigned long long']], 'hiddenVolumeSize': [17424, ['unsigned long long']], 'hiddenVolumeOffset': [17432, ['unsigned long long']], 'hiddenVolumeProtectedSize': [17440, ['unsigned long long']], 'bPartitionInInactiveSysEncScope': [17448, ['long']], 'FirstDataUnitNo': [17456, ['UINT64_STRUCT']], 'RequiredProgramVersion': [17464, ['unsigned short']], 'LegacyVolume': [17468, ['long']], 'SectorSize': [17472, ['unsigned long']], 'VolumeSize': [17480, ['UINT64_STRUCT']], 'EncryptedAreaStart': [17488, ['UINT64_STRUCT']], 'EncryptedAreaLength': [17496, ['UINT64_STRUCT']], 'HeaderFlags': [17504, ['unsigned long']]}], 'Password': [72, {'Length': [0, ['unsigned long']], 'Text': [4, ['array', 65, ['unsigned char']]], 'Pad': [69, ['array', 3, ['unsigned char']]]}], 'EXTENSION': [1296, {'bRootDevice': [0, ['long']], 'IsVolumeDevice': [4, ['long']], 'IsDriveFilterDevice': [8, ['long']], 'IsVolumeFilterDevice': [12, ['long']], 'lMagicNumber': [16, ['unsigned long']], 'UniqueVolumeId': [20, ['long']], 'nDosDriveNo': [24, ['long']], 'bShuttingDown': [28, ['long']], 'bThreadShouldQuit': [32, ['long']], 'peThread': [36, ['pointer', ['_KTHREAD']]], 'keCreateEvent': [40, ['_KEVENT']], 'ListSpinLock': [56, ['unsigned long']], 'ListEntry': [60, ['_LIST_ENTRY']], 'RequestSemaphore': [68, ['_KSEMAPHORE']], 'hDeviceFile': [88, ['pointer', ['void']]], 'pfoDeviceFile': [92, ['pointer', ['_FILE_OBJECT']]], 'pFsdDevice': [96, ['pointer', ['_DEVICE_OBJECT']]], 'cryptoInfo': [100, ['pointer', ['CRYPTO_INFO_t']]], 'HostLength': [104, ['long long']], 'DiskLength': [112, ['long long']], 'NumberOfCylinders': [120, ['long long']], 'TracksPerCylinder': [128, ['unsigned long']], 'SectorsPerTrack': [132, ['unsigned long']], 'BytesPerSector': [136, ['unsigned long']], 'PartitionType': [140, ['unsigned char']], 'HostBytesPerSector': [144, ['unsigned long']], 'keVolumeEvent': [148, ['_KEVENT']], 'Queue': [168, ['EncryptedIoQueue']], 'bReadOnly': [648, ['long']], 'bRemovable': [652, ['long']], 'PartitionInInactiveSysEncScope': [656, ['long']], 'bRawDevice': [660, ['long']], 'bMountManager': [664, ['long']], 'SystemFavorite': [668, ['long']], 'wszVolume': [672, ['array', 260, ['wchar']]], 'fileCreationTime': [1192, ['_LARGE_INTEGER']], 'fileLastAccessTime': [1200, ['_LARGE_INTEGER']], 'fileLastWriteTime': [1208, ['_LARGE_INTEGER']], 'fileLastChangeTime': [1216, ['_LARGE_INTEGER']], 'bTimeStampValid': [1224, ['long']], 'UserSid': [1228, ['pointer', ['void']]], 'SecurityClientContextValid': [1232, ['long']], 'SecurityClientContext': [1236, ['_SECURITY_CLIENT_CONTEXT']]}]}
tc_71a_vtypes_x86 = {'UINT64_STRUCT': [8, {'LowPart': [0, ['unsigned long']], 'HighPart': [4, ['unsigned long']], 'Value': [0, ['unsigned long long']]}], 'CRYPTO_INFO_t': [17512, {'ea': [0, ['long']], 'mode': [4, ['long']], 'ks': [8, ['array', 5324, ['unsigned char']]], 'ks2': [5332, ['array', 5324, ['unsigned char']]], 'hiddenVolume': [10656, ['long']], 'HeaderVersion': [10660, ['unsigned short']], 'gf_ctx': [10664, ['GfCtx']], 'master_keydata': [16808, ['array', 256, ['unsigned char']]], 'k2': [17064, ['array', 256, ['unsigned char']]], 'salt': [17320, ['array', 64, ['unsigned char']]], 'noIterations': [17384, ['long']], 'pkcs5': [17388, ['long']], 'volume_creation_time': [17392, ['unsigned long long']], 'header_creation_time': [17400, ['unsigned long long']], 'bProtectHiddenVolume': [17408, ['long']], 'bHiddenVolProtectionAction': [17412, ['long']], 'volDataAreaOffset': [17416, ['unsigned long long']], 'hiddenVolumeSize': [17424, ['unsigned long long']], 'hiddenVolumeOffset': [17432, ['unsigned long long']], 'hiddenVolumeProtectedSize': [17440, ['unsigned long long']], 'bPartitionInInactiveSysEncScope': [17448, ['long']], 'FirstDataUnitNo': [17456, ['UINT64_STRUCT']], 'RequiredProgramVersion': [17464, ['unsigned short']], 'LegacyVolume': [17468, ['long']], 'SectorSize': [17472, ['unsigned long']], 'VolumeSize': [17480, ['UINT64_STRUCT']], 'EncryptedAreaStart': [17488, ['UINT64_STRUCT']], 'EncryptedAreaLength': [17496, ['UINT64_STRUCT']], 'HeaderFlags': [17504, ['unsigned long']]}], 'EXTENSION': [1232, {'bRootDevice': [0, ['long']], 'IsVolumeDevice': [4, ['long']], 'IsDriveFilterDevice': [8, ['long']], 'IsVolumeFilterDevice': [12, ['long']], 'UniqueVolumeId': [16, ['long']], 'nDosDriveNo': [20, ['long']], 'bShuttingDown': [24, ['long']], 'bThreadShouldQuit': [28, ['long']], 'peThread': [32, ['pointer', ['_KTHREAD']]], 'keCreateEvent': [36, ['_KEVENT']], 'ListSpinLock': [52, ['unsigned long']], 'ListEntry': [56, ['_LIST_ENTRY']], 'RequestSemaphore': [64, ['_KSEMAPHORE']], 'hDeviceFile': [84, ['pointer', ['void']]], 'pfoDeviceFile': [88, ['pointer', ['_FILE_OBJECT']]], 'pFsdDevice': [92, ['pointer', ['_DEVICE_OBJECT']]], 'cryptoInfo': [96, ['pointer', ['CRYPTO_INFO_t']]], 'HostLength': [104, ['long long']], 'DiskLength': [112, ['long long']], 'NumberOfCylinders': [120, ['long long']], 'TracksPerCylinder': [128, ['unsigned long']], 'SectorsPerTrack': [132, ['unsigned long']], 'BytesPerSector': [136, ['unsigned long']], 'PartitionType': [140, ['unsigned char']], 'HostBytesPerSector': [144, ['unsigned long']], 'keVolumeEvent': [148, ['_KEVENT']], 'Queue': [168, ['EncryptedIoQueue']], 'bReadOnly': [584, ['long']], 'bRemovable': [588, ['long']], 'PartitionInInactiveSysEncScope': [592, ['long']], 'bRawDevice': [596, ['long']], 'bMountManager': [600, ['long']], 'SystemFavorite': [604, ['long']], 'wszVolume': [608, ['array', 260, ['wchar']]], 'fileCreationTime': [1128, ['_LARGE_INTEGER']], 'fileLastAccessTime': [1136, ['_LARGE_INTEGER']], 'fileLastWriteTime': [1144, ['_LARGE_INTEGER']], 'fileLastChangeTime': [1152, ['_LARGE_INTEGER']], 'bTimeStampValid': [1160, ['long']], 'UserSid': [1164, ['pointer', ['void']]], 'SecurityClientContextValid': [1168, ['long']], 'SecurityClientContext': [1172, ['_SECURITY_CLIENT_CONTEXT']]}], 'Password': [72, {'Length': [0, ['unsigned long']], 'Text': [4, ['array', 65, ['unsigned char']]], 'Pad': [69, ['array', 3, ['unsigned char']]]}]}
tc_70a_vtypes_x64 = {'UINT64_STRUCT': [8, {'LowPart': [0, ['unsigned long']], 'HighPart': [4, ['unsigned long']], 'Value': [0, ['unsigned long long']]}], 'CRYPTO_INFO_t': [17512, {'ea': [0, ['long']], 'mode': [4, ['long']], 'ks': [8, ['array', 5324, ['unsigned char']]], 'ks2': [5332, ['array', 5324, ['unsigned char']]], 'hiddenVolume': [10656, ['long']], 'HeaderVersion': [10660, ['unsigned short']], 'gf_ctx': [10664, ['GfCtx']], 'master_keydata': [16808, ['array', 256, ['unsigned char']]], 'k2': [17064, ['array', 256, ['unsigned char']]], 'salt': [17320, ['array', 64, ['unsigned char']]], 'noIterations': [17384, ['long']], 'pkcs5': [17388, ['long']], 'volume_creation_time': [17392, ['unsigned long long']], 'header_creation_time': [17400, ['unsigned long long']], 'bProtectHiddenVolume': [17408, ['long']], 'bHiddenVolProtectionAction': [17412, ['long']], 'volDataAreaOffset': [17416, ['unsigned long long']], 'hiddenVolumeSize': [17424, ['unsigned long long']], 'hiddenVolumeOffset': [17432, ['unsigned long long']], 'hiddenVolumeProtectedSize': [17440, ['unsigned long long']], 'bPartitionInInactiveSysEncScope': [17448, ['long']], 'FirstDataUnitNo': [17456, ['UINT64_STRUCT']], 'RequiredProgramVersion': [17464, ['unsigned short']], 'LegacyVolume': [17468, ['long']], 'SectorSize': [17472, ['unsigned long']], 'VolumeSize': [17480, ['UINT64_STRUCT']], 'EncryptedAreaStart': [17488, ['UINT64_STRUCT']], 'EncryptedAreaLength': [17496, ['UINT64_STRUCT']], 'HeaderFlags': [17504, ['unsigned long']]}], 'EXTENSION': [1600, {'bRootDevice': [0, ['long']], 'IsVolumeDevice': [4, ['long']], 'IsDriveFilterDevice': [8, ['long']], 'IsVolumeFilterDevice': [12, ['long']], 'lMagicNumber': [16, ['unsigned long']], 'UniqueVolumeId': [20, ['long']], 'nDosDriveNo': [24, ['long']], 'bShuttingDown': [28, ['long']], 'bThreadShouldQuit': [32, ['long']], 'peThread': [40, ['pointer64', ['_KTHREAD']]], 'keCreateEvent': [48, ['_KEVENT']], 'ListSpinLock': [72, ['unsigned long long']], 'ListEntry': [80, ['_LIST_ENTRY']], 'RequestSemaphore': [96, ['_KSEMAPHORE']], 'hDeviceFile': [128, ['pointer64', ['void']]], 'pfoDeviceFile': [136, ['pointer64', ['_FILE_OBJECT']]], 'pFsdDevice': [144, ['pointer64', ['_DEVICE_OBJECT']]], 'cryptoInfo': [152, ['pointer64', ['CRYPTO_INFO_t']]], 'HostLength': [160, ['long long']], 'DiskLength': [168, ['long long']], 'NumberOfCylinders': [176, ['long long']], 'TracksPerCylinder': [184, ['unsigned long']], 'SectorsPerTrack': [188, ['unsigned long']], 'BytesPerSector': [192, ['unsigned long']], 'PartitionType': [196, ['unsigned char']], 'HostBytesPerSector': [200, ['unsigned long']], 'keVolumeEvent': [208, ['_KEVENT']], 'Queue': [232, ['EncryptedIoQueue']], 'bReadOnly': [928, ['long']], 'bRemovable': [932, ['long']], 'PartitionInInactiveSysEncScope': [936, ['long']], 'bRawDevice': [940, ['long']], 'bMountManager': [944, ['long']], 'SystemFavorite': [948, ['long']], 'wszVolume': [952, ['array', 260, ['wchar']]], 'fileCreationTime': [1472, ['_LARGE_INTEGER']], 'fileLastAccessTime': [1480, ['_LARGE_INTEGER']], 'fileLastWriteTime': [1488, ['_LARGE_INTEGER']], 'fileLastChangeTime': [1496, ['_LARGE_INTEGER']], 'bTimeStampValid': [1504, ['long']], 'UserSid': [1512, ['pointer64', ['void']]], 'SecurityClientContextValid': [1520, ['long']], 'SecurityClientContext': [1528, ['_SECURITY_CLIENT_CONTEXT']]}], 'Password': [72, {'Length': [0, ['unsigned long']], 'Text': [4, ['array', 65, ['unsigned char']]], 'Pad': [69, ['array', 3, ['unsigned char']]]}]}
tc_71a_vtypes_x64 = {'UINT64_STRUCT': [8, {'LowPart': [0, ['unsigned long']], 'HighPart': [4, ['unsigned long']], 'Value': [0, ['unsigned long long']]}], 'CRYPTO_INFO_t': [17512, {'ea': [0, ['long']], 'mode': [4, ['long']], 'ks': [8, ['array', 5324, ['unsigned char']]], 'ks2': [5332, ['array', 5324, ['unsigned char']]], 'hiddenVolume': [10656, ['long']], 'HeaderVersion': [10660, ['unsigned short']], 'gf_ctx': [10664, ['GfCtx']], 'master_keydata': [16808, ['array', 256, ['unsigned char']]], 'k2': [17064, ['array', 256, ['unsigned char']]], 'salt': [17320, ['array', 64, ['unsigned char']]], 'noIterations': [17384, ['long']], 'pkcs5': [17388, ['long']], 'volume_creation_time': [17392, ['unsigned long long']], 'header_creation_time': [17400, ['unsigned long long']], 'bProtectHiddenVolume': [17408, ['long']], 'bHiddenVolProtectionAction': [17412, ['long']], 'volDataAreaOffset': [17416, ['unsigned long long']], 'hiddenVolumeSize': [17424, ['unsigned long long']], 'hiddenVolumeOffset': [17432, ['unsigned long long']], 'hiddenVolumeProtectedSize': [17440, ['unsigned long long']], 'bPartitionInInactiveSysEncScope': [17448, ['long']], 'FirstDataUnitNo': [17456, ['UINT64_STRUCT']], 'RequiredProgramVersion': [17464, ['unsigned short']], 'LegacyVolume': [17468, ['long']], 'SectorSize': [17472, ['unsigned long']], 'VolumeSize': [17480, ['UINT64_STRUCT']], 'EncryptedAreaStart': [17488, ['UINT64_STRUCT']], 'EncryptedAreaLength': [17496, ['UINT64_STRUCT']], 'HeaderFlags': [17504, ['unsigned long']]}], 'Password': [72, {'Length': [0, ['unsigned long']], 'Text': [4, ['array', 65, ['unsigned char']]], 'Pad': [69, ['array', 3, ['unsigned char']]]}], 'EXTENSION': [1504, {'bRootDevice': [0, ['long']], 'IsVolumeDevice': [4, ['long']], 'IsDriveFilterDevice': [8, ['long']], 'IsVolumeFilterDevice': [12, ['long']], 'UniqueVolumeId': [16, ['long']], 'nDosDriveNo': [20, ['long']], 'bShuttingDown': [24, ['long']], 'bThreadShouldQuit': [28, ['long']], 'peThread': [32, ['pointer64', ['_KTHREAD']]], 'keCreateEvent': [40, ['_KEVENT']], 'ListSpinLock': [64, ['unsigned long long']], 'ListEntry': [72, ['_LIST_ENTRY']], 'RequestSemaphore': [88, ['_KSEMAPHORE']], 'hDeviceFile': [120, ['pointer64', ['void']]], 'pfoDeviceFile': [128, ['pointer64', ['_FILE_OBJECT']]], 'pFsdDevice': [136, ['pointer64', ['_DEVICE_OBJECT']]], 'cryptoInfo': [144, ['pointer64', ['CRYPTO_INFO_t']]], 'HostLength': [152, ['long long']], 'DiskLength': [160, ['long long']], 'NumberOfCylinders': [168, ['long long']], 'TracksPerCylinder': [176, ['unsigned long']], 'SectorsPerTrack': [180, ['unsigned long']], 'BytesPerSector': [184, ['unsigned long']], 'PartitionType': [188, ['unsigned char']], 'HostBytesPerSector': [192, ['unsigned long']], 'keVolumeEvent': [200, ['_KEVENT']], 'Queue': [224, ['EncryptedIoQueue']], 'bReadOnly': [832, ['long']], 'bRemovable': [836, ['long']], 'PartitionInInactiveSysEncScope': [840, ['long']], 'bRawDevice': [844, ['long']], 'bMountManager': [848, ['long']], 'SystemFavorite': [852, ['long']], 'wszVolume': [856, ['array', 260, ['wchar']]], 'fileCreationTime': [1376, ['_LARGE_INTEGER']], 'fileLastAccessTime': [1384, ['_LARGE_INTEGER']], 'fileLastWriteTime': [1392, ['_LARGE_INTEGER']], 'fileLastChangeTime': [1400, ['_LARGE_INTEGER']], 'bTimeStampValid': [1408, ['long']], 'UserSid': [1416, ['pointer64', ['void']]], 'SecurityClientContextValid': [1424, ['long']], 'SecurityClientContext': [1432, ['_SECURITY_CLIENT_CONTEXT']]}]}

class TrueCryptPassphrase(common.AbstractWindowsCommand):
    """TrueCrypt Cached Passphrase Finder"""

    def __init__(self, config, *args, **kwargs):
        if False:
            return 10
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        config.add_option('MIN-LENGTH', short_option='M', default=5, help='Mimumim length of passphrases to identify', action='store', type='int')

    @staticmethod
    def scan_module(addr_space, module_base, min_length):
        if False:
            return 10
        dos_header = obj.Object('_IMAGE_DOS_HEADER', offset=module_base, vm=addr_space)
        nt_header = dos_header.get_nt_header()
        data_section = None
        for sec in nt_header.get_sections():
            if str(sec.Name) == '.data':
                data_section = sec
                break
        if not data_section:
            raise StopIteration
        base = sec.VirtualAddress + module_base
        size = sec.Misc.VirtualSize
        ints = obj.Object('Array', targetType='int', offset=base, count=size / 4, vm=addr_space)
        for length in ints:
            if length >= min_length and length <= 64:
                offset = length.obj_offset + 4
                passphrase = addr_space.read(offset, length)
                if not passphrase:
                    continue
                chars = [c for c in passphrase if ord(c) >= 32 and ord(c) <= 127]
                if len(chars) != length:
                    continue
                if addr_space.read(offset + length, 3) != '\x00' * 3:
                    continue
                yield (offset, passphrase)

    def calculate(self):
        if False:
            while True:
                i = 10
        addr_space = utils.load_as(self._config)
        for mod in modules.lsmod(addr_space):
            if str(mod.BaseDllName).lower() != 'truecrypt.sys':
                continue
            for (offset, password) in self.scan_module(addr_space, mod.DllBase, self._config.MIN_LENGTH):
                yield (offset, password)

    def render_text(self, outfd, data):
        if False:
            return 10
        for (offset, passphrase) in data:
            outfd.write('Found at {0:#x} length {1}: {2}\n'.format(offset, len(passphrase), passphrase))

class TrueCryptSummary(common.AbstractWindowsCommand):
    """TrueCrypt Summary"""

    def calculate(self):
        if False:
            return 10
        addr_space = utils.load_as(self._config)
        memory_model = addr_space.profile.metadata.get('memory_model')
        if memory_model == '32bit':
            regapi = registryapi.RegistryApi(self._config)
            regapi.reset_current()
            regapi.set_current(hive_name='software')
            x86key = 'Microsoft\\Windows\\CurrentVersion\\Uninstall'
            x64key = 'Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall'
            for subkey in regapi.reg_get_all_subkeys(None, key=x86key):
                if str(subkey.Name) == 'TrueCrypt':
                    subpath = x86key + '\\' + subkey.Name
                    version = regapi.reg_get_value('software', key=subpath, value='DisplayVersion')
                    if version:
                        yield ('Registry Version', '{0} Version {1}'.format(str(subkey.Name), version))
        scanner = TrueCryptPassphrase(self._config)
        for (offset, passphrase) in scanner.calculate():
            yield ('Password', '{0} at offset {1:#x}'.format(passphrase, offset))
        for proc in tasks.pslist(addr_space):
            if str(proc.ImageFileName).lower() == 'truecrypt.exe':
                yield ('Process', '{0} at {1:#x} pid {2}'.format(proc.ImageFileName, proc.obj_offset, proc.UniqueProcessId))
        scanner = svcscan.SvcScan(self._config)
        for service in scanner.calculate():
            name = str(service.ServiceName.dereference())
            if name == 'truecrypt':
                yield ('Service', '{0} state {1}'.format(name, service.State))
        for mod in modules.lsmod(addr_space):
            basename = str(mod.BaseDllName or '').lower()
            fullname = str(mod.FullDllName or '').lower()
            if basename.endswith('truecrypt.sys') or fullname.endswith('truecrypt.sys'):
                yield ('Kernel Module', '{0} at {1:#x} - {2:#x}'.format(mod.BaseDllName, mod.DllBase, mod.DllBase + mod.SizeOfImage))
        scanner = filescan.SymLinkScan(self._config)
        for symlink in scanner.calculate():
            object_header = symlink.get_object_header()
            if 'TrueCryptVolume' in str(symlink.LinkTarget or ''):
                yield ('Symbolic Link', '{0} -> {1} mounted {2}'.format(str(object_header.NameInfo.Name or ''), str(symlink.LinkTarget or ''), str(symlink.CreationTime or '')))
        scanner = filescan.FileScan(self._config)
        for fileobj in scanner.calculate():
            filename = str(fileobj.file_name_with_device() or '')
            if 'TrueCryptVolume' in filename:
                yield ('File Object', '{0} at {1:#x}'.format(filename, fileobj.obj_offset))
        scanner = filescan.DriverScan(self._config)
        for driver in scanner.calculate():
            object_header = driver.get_object_header()
            driverext = driver.DriverExtension
            drivername = str(driver.DriverName or '')
            servicekey = str(driverext.ServiceKeyName or '')
            if drivername.endswith('truecrypt') or servicekey.endswith('truecrypt'):
                yield ('Driver', '{0} at {1:#x} range {2:#x} - {3:#x}'.format(drivername, driver.obj_offset, driver.DriverStart, driver.DriverStart + driver.DriverSize))
                for device in driver.devices():
                    header = device.get_object_header()
                    devname = str(header.NameInfo.Name or '')
                    type = devicetree.DEVICE_CODES.get(device.DeviceType.v())
                    yield ('Device', '{0} at {1:#x} type {2}'.format(devname or '<HIDDEN>', device.obj_offset, type or 'UNKNOWN'))
                    if type == 'FILE_DEVICE_DISK':
                        data = addr_space.read(device.DeviceExtension, 2000)
                        offset = data.find('\\\x00?\x00?\x00\\\x00')
                        if offset == -1:
                            container = '<HIDDEN>'
                        else:
                            container = obj.Object('String', length=255, offset=device.DeviceExtension + offset, encoding='utf16', vm=addr_space)
                        yield ('Container', 'Path: {0}'.format(container))

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        for (field, info) in data:
            outfd.write('{0:20} {1}\n'.format(field, info))

class TrueCryptMaster(common.AbstractWindowsCommand):
    """Recover TrueCrypt 7.1a Master Keys"""
    version_map = {'7.1a': {'32bit': tc_71a_vtypes_x86, '64bit': tc_71a_vtypes_x64}, '7.0a': {'32bit': tc_70a_vtypes_x86, '64bit': tc_70a_vtypes_x64}}

    def __init__(self, config, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        config.add_option('DUMP-DIR', short_option='D', default=None, help='Directory in which to dump the keys')
        config.add_option('VERSION', short_option='T', default='7.1a', help='Truecrypt version string (default: 7.1a)')

    @staticmethod
    def apply_types(addr_space, ver):
        if False:
            return 10
        'Apply the TrueCrypt types for a specific version of TC. \n\n        @param addr_space: <volatility.BaseAddressSpace>\n        @param ver: <string> version \n        '
        mm_model = addr_space.profile.metadata.get('memory_model', '32bit')
        try:
            vtypes = TrueCryptMaster.version_map[ver][mm_model]
            addr_space.profile.vtypes.update(vtypes)
            addr_space.profile.merge_overlay({'EXTENSION': [None, {'wszVolume': [None, ['String', dict(length=260, encoding='utf16')]]}], 'CRYPTO_INFO_t': [None, {'mode': [None, ['Enumeration', dict(target='long', choices={1: 'XTS', 2: 'LWR', 3: 'CBC', 4: 'OUTER_CBC', 5: 'INNER_CBC'})]], 'ea': [None, ['Enumeration', dict(target='long', choices={1: 'AES', 2: 'SERPENT', 3: 'TWOFISH', 4: 'BLOWFISH', 5: 'CAST', 6: 'TRIPLEDES'})]]}]})
            addr_space.profile.compile()
        except KeyError:
            debug.error('Truecrypt version {0} is not supported'.format(ver))

    def calculate(self):
        if False:
            i = 10
            return i + 15
        addr_space = utils.load_as(self._config)
        self.apply_types(addr_space, self._config.VERSION)
        scanner = filescan.DriverScan(self._config)
        for driver in scanner.calculate():
            drivername = str(driver.DriverName or '')
            if drivername.endswith('truecrypt'):
                for device in driver.devices():
                    code = device.DeviceType.v()
                    type = devicetree.DEVICE_CODES.get(code)
                    if type == 'FILE_DEVICE_DISK':
                        yield device

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        for device in data:
            ext = device.DeviceExtension.dereference_as('EXTENSION')
            if not ext.is_valid():
                continue
            outfd.write('Container: {0}\n'.format(ext.wszVolume))
            outfd.write('Hidden Volume: {0}\n'.format('Yes' if ext.cryptoInfo.hiddenVolume == 1 else 'No'))
            outfd.write('Removable: {0}\n'.format('Yes' if ext.bRemovable == 1 else 'No'))
            outfd.write('Read Only: {0}\n'.format('Yes' if ext.bReadOnly == 1 else 'No'))
            outfd.write('Disk Length: {0} (bytes)\n'.format(ext.DiskLength))
            outfd.write('Host Length: {0} (bytes)\n'.format(ext.HostLength))
            outfd.write('Encryption Algorithm: {0}\n'.format(ext.cryptoInfo.ea))
            outfd.write('Mode: {0}\n'.format(ext.cryptoInfo.mode))
            outfd.write('Master Key\n')
            key = device.obj_vm.read(ext.cryptoInfo.master_keydata.obj_offset, 64)
            addr = ext.cryptoInfo.master_keydata.obj_offset
            outfd.write('{0}\n'.format('\n'.join(['{0:#010x}  {1:<48}  {2}'.format(addr + o, h, ''.join(c)) for (o, h, c) in utils.Hexdump(key)])))
            if self._config.DUMP_DIR:
                if not os.path.isdir(self._config.DUMP_DIR):
                    debug.error('The path {0} is not a valid directory'.format(self._config.DUMP_DIR))
                name = '{0:#x}_master.key'.format(addr)
                keyfile = os.path.join(self._config.DUMP_DIR, name)
                with open(keyfile, 'wb') as handle:
                    handle.write(key)
                outfd.write('Dumped {0} bytes to {1}\n'.format(len(key), keyfile))
            outfd.write('\n')