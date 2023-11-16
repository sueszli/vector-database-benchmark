import copy
import volatility.obj as obj
import volatility.plugins.overlays.windows.windows as windows
import volatility.plugins.overlays.windows.pe_vtypes as pe_vtypes
import volatility.registry as registry
import volatility.debug as debug

class Pointer64Decorator(object):

    def __init__(self, f):
        if False:
            print('Hello World!')
        self.f = f

    def __call__(self, name, typeList, typeDict=None):
        if False:
            for i in range(10):
                print('nop')
        if len(typeList) and typeList[0] == 'pointer64':
            typeList = copy.deepcopy(typeList)
            typeList[0] = 'pointer'
        return self.f(name, typeList, typeDict)

class _EX_FAST_REF(windows._EX_FAST_REF):
    MAX_FAST_REF = 15

class LIST_ENTRY32(windows._LIST_ENTRY):
    """the LDR member is an unsigned long not a Pointer as regular LIST_ENTRY"""

    def get_next_entry(self, member):
        if False:
            print('Hello World!')
        return obj.Object('LIST_ENTRY32', offset=self.m(member).v(), vm=self.obj_vm)

class ExFastRefx64(obj.ProfileModification):
    before = ['WindowsOverlay', 'WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.object_classes.update({'_EX_FAST_REF': _EX_FAST_REF})

class Windows64Overlay(obj.ProfileModification):
    before = ['WindowsOverlay', 'WindowsObjectClasses']
    conditions = {'memory_model': lambda x: x == '64bit', 'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.merge_overlay({'VOLATILITY_MAGIC': [0, {'PoolAlignment': [0, ['VolatilityMagic', dict(value=16)]], 'KUSER_SHARED_DATA': [0, ['VolatilityMagic', dict(value=18446734727860715520)]]}]})
        profile.vtypes['_IMAGE_NT_HEADERS'] = profile.vtypes['_IMAGE_NT_HEADERS64']
        profile.merge_overlay({'_DBGKD_GET_VERSION64': [None, {'DebuggerDataList': [None, ['pointer', ['unsigned long long']]]}]})
        profile.merge_overlay({'_KPROCESS': [None, {'DirectoryTableBase': [None, ['unsigned long long']]}]})
        profile._list_to_type = Pointer64Decorator(profile._list_to_type)

class WinPeb32(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit'}
    before = ['WinPEVTypes', 'WinPEx64VTypes', 'WinPEObjectClasses', 'WindowsObjectClasses']

    def cast_as_32bit(self, source_vtype):
        if False:
            while True:
                i = 10
        vtype = copy.copy(source_vtype)
        members = vtype[1]
        mapping = {'pointer': 'pointer32', '_UNICODE_STRING': '_UNICODE32_STRING', '_LIST_ENTRY': 'LIST_ENTRY32'}
        for (name, member) in members.items():
            datatype = member[1][0]
            if datatype in mapping:
                member[1][0] = mapping[datatype]
        return vtype

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profiles = registry.get_plugin_classes(obj.Profile)
        meta = profile.metadata
        profile_32bit = None
        for prof in profiles.values():
            if prof._md_os == 'windows' and prof._md_major == meta.get('major') and (prof._md_minor == meta.get('minor')) and (prof._md_build == meta.get('build') or prof._md_build + 1 == meta.get('build')) and (prof._md_memory_model == '32bit'):
                profile_32bit = prof()
                break
        if profile_32bit == None:
            debug.warning('Cannot find a 32-bit equivalent profile. The WoW64 plugins (dlllist, ldrmodules, etc) may not work.')
            return
        profile.vtypes.update({'_PEB32_LDR_DATA': self.cast_as_32bit(profile_32bit.vtypes['_PEB_LDR_DATA']), '_LDR32_DATA_TABLE_ENTRY': self.cast_as_32bit(profile_32bit.vtypes['_LDR_DATA_TABLE_ENTRY']), '_UNICODE32_STRING': self.cast_as_32bit(profile_32bit.vtypes['_UNICODE_STRING'])})
        profile.object_classes.update({'_LDR32_DATA_TABLE_ENTRY': pe_vtypes._LDR_DATA_TABLE_ENTRY, '_UNICODE32_STRING': windows._UNICODE_STRING, 'LIST_ENTRY32': LIST_ENTRY32})
        profile.merge_overlay({'_PEB32': [None, {'Ldr': [None, ['pointer32', ['_PEB32_LDR_DATA']]]}]})