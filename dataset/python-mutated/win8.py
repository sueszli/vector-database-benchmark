import volatility.obj as obj
import volatility.plugins.gui.constants as consts
import volatility.plugins.gui.win32k_core as win32k_core
import volatility.plugins.gui.vtypes.win7_sp0_x86_vtypes_gui as win7_sp0_x86_vtypes_gui
import volatility.plugins.gui.vtypes.win7_sp0_x64_vtypes_gui as win7_sp0_x64_vtypes_gui

class _RTL_ATOM_TABLE_ENTRY(win32k_core._RTL_ATOM_TABLE_ENTRY):
    """A class for atom table entries"""

    @property
    def Flags(self):
        if False:
            print('Hello World!')
        return self.Reference.Flags

    @property
    def ReferenceCount(self):
        if False:
            print('Hello World!')
        return self.Reference.ReferenceCount

class Win8x86Gui(obj.ProfileModification):
    before = ['XP2003x86BaseVTypes', 'Win32Kx86VTypes', 'AtomTablex86Overlay', 'Win32KCoreClasses']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x > 1}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.vtypes.update(win7_sp0_x86_vtypes_gui.win32k_types)
        profile.object_classes.update({'_RTL_ATOM_TABLE_ENTRY': _RTL_ATOM_TABLE_ENTRY})
        profile.merge_overlay({'tagWINDOWSTATION': [None, {'spwndClipOwner': [40, ['pointer', ['tagWND']]], 'spwndClipViewer': [36, ['pointer', ['tagWND']]], 'pClipBase': [48, ['pointer', ['array', lambda x: x.cNumClipFormats, ['tagCLIP']]]], 'cNumClipFormats': [52, ['unsigned long']], 'iClipSerialNumber': [56, ['unsigned long']], 'pGlobalAtomTable': [72, ['pointer', ['void']]]}], '_HANDLEENTRY': [None, {'bType': [None, ['Enumeration', dict(target='unsigned char', choices=consts.HANDLE_TYPE_ENUM_SEVEN)]]}], 'tagCLIP': [20, {'fmt': [None, ['Enumeration', dict(target='unsigned long', choices=consts.CLIPBOARD_FORMAT_ENUM)]]}], 'tagTHREADINFO': [None, {'ppi': [196, ['pointer', ['tagPROCESSINFO']]], 'PtiLink': [344, ['_LIST_ENTRY']]}], 'tagDESKTOP': [None, {'rpdeskNext': [8, ['pointer', ['tagDESKTOP']]], 'rpwinstaParent': [12, ['pointer', ['tagWINDOWSTATION']]], 'pheapDesktop': [60, ['pointer', ['tagWIN32HEAP']]], 'PtiList': [88, ['_LIST_ENTRY']]}], '_RTL_ATOM_TABLE': [None, {'NumBuckets': [20, ['unsigned long']], 'Buckets': [24, ['array', lambda x: x.NumBuckets, ['pointer', ['_RTL_ATOM_TABLE_ENTRY']]]]}]})

class Win8x64Gui(obj.ProfileModification):
    before = ['Win32KCoreClasses']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x > 1}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.vtypes.update(win7_sp0_x64_vtypes_gui.win32k_types)
        profile.object_classes.update({'_RTL_ATOM_TABLE_ENTRY': _RTL_ATOM_TABLE_ENTRY})
        profile.merge_overlay({'tagWINDOWSTATION': [None, {'pClipBase': [96, ['pointer', ['array', lambda x: x.cNumClipFormats, ['tagCLIP']]]], 'cNumClipFormats': [104, ['unsigned long']], 'iClipSerialNumber': [108, ['unsigned long']], 'pGlobalAtomTable': [136, ['pointer', ['void']]]}], 'tagDESKTOP': [None, {'rpdeskNext': [16, ['pointer', ['tagDESKTOP']]], 'rpwinstaParent': [24, ['pointer', ['tagWINDOWSTATION']]], 'pheapDesktop': [120, ['pointer', ['tagWIN32HEAP']]], 'PtiList': [160, ['_LIST_ENTRY']]}], 'tagTHREADINFO': [None, {'ppi': [368, ['pointer', ['tagPROCESSINFO']]], 'PtiLink': [640, ['_LIST_ENTRY']]}], 'tagCLIP': [None, {'fmt': [None, ['Enumeration', dict(target='unsigned long', choices=consts.CLIPBOARD_FORMAT_ENUM)]]}], '_RTL_ATOM_TABLE': [None, {'NumBuckets': [28, ['unsigned long']], 'Buckets': [32, ['array', lambda x: x.NumBuckets, ['pointer', ['_RTL_ATOM_TABLE_ENTRY']]]]}], '_HANDLEENTRY': [None, {'bType': [None, ['Enumeration', dict(target='unsigned char', choices=consts.HANDLE_TYPE_ENUM_SEVEN)]]}]})