import volatility.obj as obj
hibernate_vtypes = {'_PO_MEMORY_RANGE_ARRAY_LINK': [16, {'NextTable': [4, ['unsigned long']], 'EntryCount': [12, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY_RANGE': [16, {'StartPage': [4, ['unsigned long']], 'EndPage': [8, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY': [32, {'MemArrayLink': [0, ['_PO_MEMORY_RANGE_ARRAY_LINK']], 'RangeTable': [16, ['array', lambda x: x.MemArrayLink.EntryCount, ['_PO_MEMORY_RANGE_ARRAY_RANGE']]]}], '_IMAGE_XPRESS_HEADER': [32, {'u09': [9, ['unsigned char']], 'u0A': [10, ['unsigned char']], 'u0B': [11, ['unsigned char']]}]}
hibernate_vistasp01_vtypes = {'_PO_MEMORY_RANGE_ARRAY_LINK': [16, {'NextTable': [4, ['unsigned long']], 'EntryCount': [12, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY': [32, {'MemArrayLink': [0, ['_PO_MEMORY_RANGE_ARRAY_LINK']], 'RangeTable': [16, ['array', lambda x: x.MemArrayLink.EntryCount, ['_PO_MEMORY_RANGE_ARRAY_RANGE']]]}]}

class HiberVistaSP01x86(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'build': lambda x: x <= 6001, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update(hibernate_vistasp01_vtypes)
hibernate_vistasp2_vtypes = {'_PO_MEMORY_RANGE_ARRAY_LINK': [16, {'NextTable': [4, ['unsigned long']], 'EntryCount': [8, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY_RANGE': [8, {'StartPage': [0, ['unsigned long']], 'EndPage': [4, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY': [32, {'MemArrayLink': [0, ['_PO_MEMORY_RANGE_ARRAY_LINK']], 'RangeTable': [12, ['array', lambda x: x.MemArrayLink.EntryCount, ['_PO_MEMORY_RANGE_ARRAY_RANGE']]]}]}

class HiberVistaSP2x86(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'build': lambda x: x == 6002, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            return 10
        profile.vtypes.update(hibernate_vistasp2_vtypes)
hibernate_win7_vtypes = {'_PO_MEMORY_RANGE_ARRAY_LINK': [16, {'NextTable': [0, ['unsigned long']], 'EntryCount': [4, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY_RANGE': [8, {'StartPage': [0, ['unsigned long']], 'EndPage': [4, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY': [32, {'MemArrayLink': [0, ['_PO_MEMORY_RANGE_ARRAY_LINK']], 'RangeTable': [8, ['array', lambda x: x.MemArrayLink.EntryCount, ['_PO_MEMORY_RANGE_ARRAY_RANGE']]]}]}

class HiberWin7SP01x86(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 1, 'build': lambda x: x <= 7601, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update(hibernate_win7_vtypes)
hibernate_win7_x64_vtypes = {'_PO_MEMORY_RANGE_ARRAY_LINK': [16, {'NextTable': [0, ['unsigned long long']], 'EntryCount': [8, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY_RANGE': [16, {'StartPage': [0, ['unsigned long long']], 'EndPage': [8, ['unsigned long long']]}], '_PO_MEMORY_RANGE_ARRAY': [32, {'MemArrayLink': [0, ['_PO_MEMORY_RANGE_ARRAY_LINK']], 'RangeTable': [16, ['array', lambda x: x.MemArrayLink.EntryCount, ['_PO_MEMORY_RANGE_ARRAY_RANGE']]]}]}

class HiberWin7SP01x64(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 1, 'build': lambda x: x <= 7601, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update(hibernate_win7_x64_vtypes)
hibernate_x64_vtypes = {'_PO_MEMORY_RANGE_ARRAY_LINK': [32, {'NextTable': [8, ['unsigned long long']], 'EntryCount': [20, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY_RANGE': [32, {'StartPage': [8, ['unsigned long long']], 'EndPage': [16, ['unsigned long long']]}], '_PO_MEMORY_RANGE_ARRAY': [64, {'MemArrayLink': [0, ['_PO_MEMORY_RANGE_ARRAY_LINK']], 'RangeTable': [32, ['array', lambda x: x.MemArrayLink.EntryCount, ['_PO_MEMORY_RANGE_ARRAY_RANGE']]]}]}

class HiberWin2003x64(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 5, 'minor': lambda x: x == 2, 'build': lambda x: x <= 3791, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.vtypes.update(hibernate_x64_vtypes)

class HiberVistaSP01x64(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'build': lambda x: x <= 6001, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update(hibernate_x64_vtypes)
hibernate_vistaSP2_x64_vtypes = {'_PO_MEMORY_RANGE_ARRAY_LINK': [24, {'NextTable': [8, ['unsigned long long']], 'EntryCount': [16, ['unsigned long']]}], '_PO_MEMORY_RANGE_ARRAY_RANGE': [16, {'StartPage': [0, ['unsigned long long']], 'EndPage': [8, ['unsigned long long']]}], '_PO_MEMORY_RANGE_ARRAY': [40, {'MemArrayLink': [0, ['_PO_MEMORY_RANGE_ARRAY_LINK']], 'RangeTable': [24, ['array', lambda x: x.MemArrayLink.EntryCount, ['_PO_MEMORY_RANGE_ARRAY_RANGE']]]}]}

class HiberVistaSP2x64(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'build': lambda x: x == 6002, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update(hibernate_vistaSP2_x64_vtypes)