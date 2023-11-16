import volatility.obj as obj

class Win10x86_Gui(obj.ProfileModification):
    before = ['XP2003x86BaseVTypes', 'Win32Kx86VTypes', 'AtomTablex86Overlay', 'Win32KCoreClasses', 'Win8x86Gui']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        build = profile.metadata.get('build', 0)
        if build >= 15063:
            profile.merge_overlay({'tagDESKTOP': [None, {'rpdeskNext': [16, ['pointer', ['tagDESKTOP']]], 'rpwinstaParent': [20, ['pointer', ['tagWINDOWSTATION']]], 'pheapDesktop': [64, ['pointer', ['tagWIN32HEAP']]], 'PtiList': [92, ['_LIST_ENTRY']]}], 'tagTHREADINFO': [None, {'ppi': [224, ['pointer', ['tagPROCESSINFO']]], 'PtiLink': [392, ['_LIST_ENTRY']]}], 'tagWND': [None, {'spwndNext': [52, ['pointer', ['tagWND']]], 'spwndPrev': [56, ['pointer', ['tagWND']]], 'spwndParent': [60, ['pointer', ['tagWND']]], 'spwndChild': [64, ['pointer', ['tagWND']]], 'lpfnWndProc': [104, ['pointer', ['void']]], 'pcls': [108, ['pointer', ['tagCLS']]], 'strName': [140, ['_LARGE_UNICODE_STRING']]}]})

class Win10x64_Gui(obj.ProfileModification):
    before = ['Win32KCoreClasses', 'Win8x64Gui']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        build = profile.metadata.get('build', 0)
        if build >= 15063:
            profile.merge_overlay({'tagDESKTOP': [None, {'rpdeskNext': [32, ['pointer64', ['tagDESKTOP']]], 'rpwinstaParent': [40, ['pointer64', ['tagWINDOWSTATION']]], 'pheapDesktop': [128, ['pointer', ['tagWIN32HEAP']]], 'PtiList': [168, ['_LIST_ENTRY']]}], 'tagTHREADINFO': [None, {'ppi': [400, ['pointer', ['tagPROCESSINFO']]], 'PtiLink': [712, ['_LIST_ENTRY']]}], 'tagWND': [None, {'spwndNext': [88, ['pointer64', ['tagWND']]], 'spwndPrev': [96, ['pointer64', ['tagWND']]], 'spwndParent': [104, ['pointer64', ['tagWND']]], 'spwndChild': [112, ['pointer64', ['tagWND']]], 'lpfnWndProc': [160, ['pointer64', ['void']]], 'pcls': [168, ['pointer64', ['tagCLS']]], 'strName': [232, ['_LARGE_UNICODE_STRING']]}]})