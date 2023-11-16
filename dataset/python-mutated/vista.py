import volatility.obj as obj
import volatility.plugins.gui.vtypes.win7_sp0_x64_vtypes_gui as win7_sp0_x64_vtypes_gui
import volatility.plugins.gui.constants as consts

class Vista2008x64GuiVTypes(obj.ProfileModification):
    before = ['XP2003x64BaseVTypes', 'Win32Kx64VTypes']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 0}

    def modification(self, profile):
        if False:
            return 10
        profile.vtypes.update(win7_sp0_x64_vtypes_gui.win32k_types)
        profile.vtypes.update({'tagSHAREDINFO': [568, {'psi': [0, ['pointer64', ['tagSERVERINFO']]], 'aheList': [8, ['pointer64', ['_HANDLEENTRY']]], 'ulSharedDelta': [24, ['unsigned long long']]}]})
        profile.merge_overlay({'tagDESKTOP': [None, {'pheapDesktop': [120, ['pointer64', ['tagWIN32HEAP']]], 'ulHeapSize': [128, ['unsigned long']]}], 'tagTHREADINFO': [None, {'ppi': [104, ['pointer64', ['tagPROCESSINFO']]], 'PtiLink': [352, ['_LIST_ENTRY']]}], 'tagHOOK': [None, {'flags': [None, ['Flags', {'bitmap': consts.HOOK_FLAGS}]]}], '_HANDLEENTRY': [None, {'bType': [None, ['Enumeration', dict(target='unsigned char', choices=consts.HANDLE_TYPE_ENUM)]]}], 'tagWINDOWSTATION': [None, {'pClipBase': [None, ['pointer', ['array', lambda x: x.cNumClipFormats, ['tagCLIP']]]]}], 'tagCLIP': [None, {'fmt': [0, ['Enumeration', dict(target='unsigned long', choices=consts.CLIPBOARD_FORMAT_ENUM)]]}]})

class Vista2008x86GuiVTypes(obj.ProfileModification):
    before = ['XP2003x86BaseVTypes', 'Win32Kx86VTypes']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 0}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.merge_overlay({'tagWINDOWSTATION': [84, {'pClipBase': [None, ['pointer', ['array', lambda x: x.cNumClipFormats, ['tagCLIP']]]]}], 'tagDESKTOP': [None, {'PtiList': [100, ['_LIST_ENTRY']], 'hsectionDesktop': [60, ['pointer', ['void']]], 'pheapDesktop': [64, ['pointer', ['tagWIN32HEAP']]], 'ulHeapSize': [68, ['unsigned long']]}], 'tagTHREADINFO': [None, {'PtiLink': [176, ['_LIST_ENTRY']], 'fsHooks': [156, ['unsigned long']], 'aphkStart': [248, ['array', 16, ['pointer', ['tagHOOK']]]]}], 'tagSERVERINFO': [None, {'cHandleEntries': [4, ['unsigned long']], 'cbHandleTable': [456, ['unsigned long']]}], 'tagSHAREDINFO': [284, {'psi': [0, ['pointer', ['tagSERVERINFO']]], 'aheList': [4, ['pointer', ['_HANDLEENTRY']]], 'ulSharedDelta': [12, ['unsigned long']]}], 'tagCLIP': [16, {}]})