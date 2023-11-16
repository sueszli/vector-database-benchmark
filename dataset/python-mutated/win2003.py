import volatility.obj as obj

class Win2003x86GuiVTypes(obj.ProfileModification):
    """Apply the overlays for Windows 2003 x86 (builds on Windows XP x86)"""
    before = ['XP2003x86BaseVTypes']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 5, 'minor': lambda x: x == 2}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.merge_overlay({'tagWINDOWSTATION': [84, {'spwndClipOwner': [24, ['pointer', ['tagWND']]], 'pGlobalAtomTable': [60, ['pointer', ['void']]]}], 'tagTHREADINFO': [None, {'PtiLink': [176, ['_LIST_ENTRY']], 'fsHooks': [156, ['unsigned long']], 'aphkStart': [248, ['array', 16, ['pointer', ['tagHOOK']]]]}], 'tagDESKTOP': [None, {'hsectionDesktop': [60, ['pointer', ['void']]], 'pheapDesktop': [64, ['pointer', ['tagWIN32HEAP']]], 'ulHeapSize': [68, ['unsigned long']], 'PtiList': [96, ['_LIST_ENTRY']]}], 'tagSERVERINFO': [None, {'cHandleEntries': [4, ['unsigned long']], 'cbHandleTable': [440, ['unsigned long']]}]})