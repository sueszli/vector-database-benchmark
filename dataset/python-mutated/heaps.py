import volatility.obj as obj

class HeapModification(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            return 10
        profile.merge_overlay({'_PEB': [None, {'ProcessHeaps': [None, ['pointer', ['array', lambda x: x.NumberOfHeaps, ['pointer', ['_HEAP']]]]]}]})