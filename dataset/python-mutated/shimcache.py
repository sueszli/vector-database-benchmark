"""
@author:       Jamie Levy (gleeda)
@license:      GNU General Public License 2.0
@contact:      jamie@memoryanalysis.net
@organization: Volatility Foundation
"""
import volatility.plugins.registry.registryapi as registryapi
from volatility.renderers import TreeGrid
import volatility.debug as debug
import volatility.utils as utils
import volatility.obj as obj
import volatility.plugins.common as common
import volatility.addrspace as addrspace
shimrecs_type_xp = {'ShimRecords': [None, {'Magic': [0, ['unsigned int']], 'NumRecords': [8, ['short']], 'Entries': [400, ['array', lambda x: x.NumRecords, ['AppCompatCacheEntry']]]}]}
shimrecs_type_2003vista = {'ShimRecords': [None, {'Magic': [0, ['unsigned int']], 'NumRecords': [4, ['int']], 'Entries': [8, ['array', lambda x: x.NumRecords, ['AppCompatCacheEntry']]]}]}
shimrecs_type_win7 = {'ShimRecords': [None, {'Magic': [0, ['unsigned int']], 'NumRecords': [4, ['int']], 'Entries': [128, ['array', lambda x: x.NumRecords, ['AppCompatCacheEntry']]]}]}
appcompat_type_xp_x86 = {'AppCompatCacheEntry': [552, {'Path': [0, ['NullString', dict(length=520, encoding='utf8')]], 'LastModified': [528, ['WinTimeStamp', dict(is_utc=True)]], 'FileSize': [536, ['long long']], 'LastUpdate': [544, ['WinTimeStamp', dict(is_utc=True)]]}]}
appcompat_type_2003_x86 = {'AppCompatCacheEntry': [24, {'Length': [0, ['unsigned short']], 'MaximumLength': [2, ['unsigned short']], 'PathOffset': [4, ['unsigned int']], 'LastModified': [8, ['WinTimeStamp', dict(is_utc=True)]], 'FileSize': [16, ['_LARGE_INTEGER']]}]}
appcompat_type_vista_x86 = {'AppCompatCacheEntry': [24, {'Length': [0, ['unsigned short']], 'MaximumLength': [2, ['unsigned short']], 'PathOffset': [4, ['unsigned int']], 'LastModified': [8, ['WinTimeStamp', dict(is_utc=True)]], 'InsertFlags': [16, ['unsigned int']], 'Flags': [20, ['unsigned int']]}]}
appcompat_type_win7_x86 = {'AppCompatCacheEntry': [32, {'Length': [0, ['unsigned short']], 'MaximumLength': [2, ['unsigned short']], 'PathOffset': [4, ['unsigned int']], 'LastModified': [8, ['WinTimeStamp', dict(is_utc=True)]], 'InsertFlags': [16, ['unsigned int']], 'ShimFlags': [20, ['unsigned int']], 'BlobSize': [24, ['unsigned int']], 'BlobOffset': [28, ['unsigned int']]}]}
appcompat_type_2003_x64 = {'AppCompatCacheEntry': [32, {'Length': [0, ['unsigned short']], 'MaximumLength': [2, ['unsigned short']], 'PathOffset': [8, ['unsigned long long']], 'LastModified': [16, ['WinTimeStamp', dict(is_utc=True)]], 'FileSize': [24, ['_LARGE_INTEGER']]}]}
appcompat_type_vista_x64 = {'AppCompatCacheEntry': [32, {'Length': [0, ['unsigned short']], 'MaximumLength': [2, ['unsigned short']], 'PathOffset': [8, ['unsigned int']], 'LastModified': [16, ['WinTimeStamp', dict(is_utc=True)]], 'InsertFlags': [24, ['unsigned int']], 'Flags': [28, ['unsigned int']]}]}
appcompat_type_win7_x64 = {'AppCompatCacheEntry': [48, {'Length': [0, ['unsigned short']], 'MaximumLength': [2, ['unsigned short']], 'PathOffset': [8, ['unsigned long long']], 'LastModified': [16, ['WinTimeStamp', dict(is_utc=True)]], 'InsertFlags': [24, ['unsigned int']], 'ShimFlags': [28, ['unsigned int']], 'BlobSize': [32, ['unsigned long long']], 'BlobOffset': [40, ['unsigned long long']]}]}

class ShimCacheTypesXPx86(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 5, 'minor': lambda x: x == 1, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update(shimrecs_type_xp)
        profile.vtypes.update(appcompat_type_xp_x86)

class ShimCacheTypes2003x86(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 5, 'minor': lambda x: x == 2, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.vtypes.update(shimrecs_type_2003vista)
        profile.vtypes.update(appcompat_type_2003_x86)

class ShimCacheTypesVistax86(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update(shimrecs_type_2003vista)
        profile.vtypes.update(appcompat_type_vista_x86)

class ShimCacheTypesWin7x86(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 1, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.vtypes.update(shimrecs_type_win7)
        profile.vtypes.update(appcompat_type_win7_x86)

class ShimCacheTypes2003x64(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 5, 'minor': lambda x: x == 2, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.vtypes.update(shimrecs_type_2003vista)
        profile.vtypes.update(appcompat_type_2003_x64)

class ShimCacheTypesVistax64(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update(shimrecs_type_2003vista)
        profile.vtypes.update(appcompat_type_vista_x64)

class ShimCacheTypesWin7x64(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 1, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update(shimrecs_type_win7)
        profile.vtypes.update(appcompat_type_win7_x64)

class ShimCache(common.AbstractWindowsCommand):
    """Parses the Application Compatibility Shim Cache registry key"""

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        self._addrspace = None
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)

    @staticmethod
    def is_valid_profile(profile):
        if False:
            for i in range(10):
                print('nop')
        return profile.metadata.get('os', 'unknown').lower() == 'windows'

    @staticmethod
    def remove_unprintable(item):
        if False:
            for i in range(10):
                print('nop')
        return ''.join([str(c) for c in item if (ord(c) > 31 or ord(c) == 9) and ord(c) <= 126])

    @staticmethod
    def get_entries(addr_space, regapi):
        if False:
            print('Hello World!')
        regapi.reset_current()
        currentcs = regapi.reg_get_currentcontrolset()
        if currentcs == None:
            currentcs = 'ControlSet001'
        version = (addr_space.profile.metadata.get('major', 0), addr_space.profile.metadata.get('minor', 0))
        xp = False
        if version <= (5, 1):
            key = currentcs + '\\Control\\Session Manager\\AppCompatibility'
            xp = True
        else:
            key = currentcs + '\\Control\\Session Manager\\AppCompatCache'
        data_raw = regapi.reg_get_value('system', key, 'AppCompatCache')
        if data_raw == None or len(data_raw) < 28:
            debug.warning('No ShimCache data found')
            raise StopIteration
        bufferas = addrspace.BufferAddressSpace(addr_space.get_config(), data=data_raw)
        shimdata = obj.Object('ShimRecords', offset=0, vm=bufferas)
        if shimdata == None:
            debug.warning('No ShimCache data found')
            raise StopIteration
        if shimdata.Magic not in [3735928559, 3134984190, 3134984174]:
            debug.warning('ShimRecords.Magic value {0:X} is not valid'.format(shimdata.Magic))
            raise StopIteration
        for e in shimdata.Entries:
            if xp:
                yield (e.Path, e.LastModified, e.LastUpdate)
            else:
                yield (ShimCache.remove_unprintable(bufferas.read(int(e.PathOffset), int(e.Length))), e.LastModified, None)

    def calculate(self):
        if False:
            return 10
        addr_space = utils.load_as(self._config)
        regapi = registryapi.RegistryApi(self._config)
        for entry in self.get_entries(addr_space, regapi):
            yield entry

    def unified_output(self, data):
        if False:
            print('Hello World!')
        return TreeGrid([('Last Modified', str), ('Last Update', str), ('Path', str)], self.generator(data))

    def generator(self, data):
        if False:
            i = 10
            return i + 15
        for (path, lm, lu) in data:
            if lu:
                yield (0, [str(lm), str(lu), str(path).strip()])
            else:
                yield (0, [str(lm), '-', str(path).strip()])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        first = True
        for (path, lm, lu) in data:
            if lu:
                if first:
                    self.table_header(outfd, [('Last Modified', '30'), ('Last Update', '30'), ('Path', '')])
                    first = False
                outfd.write('{0:30} {1:30} {2}\n'.format(lm, lu, path))
            else:
                if first:
                    self.table_header(outfd, [('Last Modified', '30'), ('Path', '')])
                    first = False
                outfd.write('{0:30} {1}\n'.format(lm, path))