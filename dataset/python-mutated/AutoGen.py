from __future__ import print_function
from __future__ import absolute_import
from Common.DataType import TAB_STAR

class AutoGen(object):
    __ObjectCache = {}

    def __new__(cls, Workspace, MetaFile, Target, Toolchain, Arch, *args, **kwargs):
        if False:
            print('Hello World!')
        Key = (Target, Toolchain, Arch, MetaFile)
        if Key in cls.__ObjectCache:
            return cls.__ObjectCache[Key]
        RetVal = cls.__ObjectCache[Key] = super(AutoGen, cls).__new__(cls)
        return RetVal

    def __hash__(self):
        if False:
            return 10
        return hash(self.MetaFile)

    def __str__(self):
        if False:
            return 10
        return str(self.MetaFile)

    def __eq__(self, Other):
        if False:
            i = 10
            return i + 15
        return Other and self.MetaFile == Other

    @classmethod
    def Cache(cls):
        if False:
            print('Hello World!')
        return cls.__ObjectCache
PrioList = {'0x11111': 16, '0x01111': 15, '0x10111': 14, '0x00111': 13, '0x11011': 12, '0x01011': 11, '0x10011': 10, '0x00011': 9, '0x11101': 8, '0x01101': 7, '0x10101': 6, '0x00101': 5, '0x11001': 4, '0x01001': 3, '0x10001': 2, '0x00001': 1}

def CalculatePriorityValue(Key):
    if False:
        print('Hello World!')
    (Target, ToolChain, Arch, CommandType, Attr) = Key.split('_')
    PriorityValue = 69905
    if Target == TAB_STAR:
        PriorityValue &= 4369
    if ToolChain == TAB_STAR:
        PriorityValue &= 65809
    if Arch == TAB_STAR:
        PriorityValue &= 69649
    if CommandType == TAB_STAR:
        PriorityValue &= 69889
    if Attr == TAB_STAR:
        PriorityValue &= 69904
    return PrioList['0x%0.5x' % PriorityValue]