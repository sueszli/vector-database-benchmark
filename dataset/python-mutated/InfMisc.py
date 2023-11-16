"""
InfMisc
"""
import Logger.Log as Logger
from Logger import ToolError
from Library import DataType as DT
from Object.Parser.InfCommonObject import InfSectionCommonDef
from Library.Misc import Sdict

class InfBootModeObject:

    def __init__(self):
        if False:
            print('Hello World!')
        self.SupportedBootModes = ''
        self.HelpString = ''
        self.Usage = ''

    def SetSupportedBootModes(self, SupportedBootModes):
        if False:
            print('Hello World!')
        self.SupportedBootModes = SupportedBootModes

    def GetSupportedBootModes(self):
        if False:
            print('Hello World!')
        return self.SupportedBootModes

    def SetHelpString(self, HelpString):
        if False:
            for i in range(10):
                print('nop')
        self.HelpString = HelpString

    def GetHelpString(self):
        if False:
            return 10
        return self.HelpString

    def SetUsage(self, Usage):
        if False:
            for i in range(10):
                print('nop')
        self.Usage = Usage

    def GetUsage(self):
        if False:
            while True:
                i = 10
        return self.Usage

class InfEventObject:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.EventType = ''
        self.HelpString = ''
        self.Usage = ''

    def SetEventType(self, EventType):
        if False:
            while True:
                i = 10
        self.EventType = EventType

    def GetEventType(self):
        if False:
            print('Hello World!')
        return self.EventType

    def SetHelpString(self, HelpString):
        if False:
            while True:
                i = 10
        self.HelpString = HelpString

    def GetHelpString(self):
        if False:
            print('Hello World!')
        return self.HelpString

    def SetUsage(self, Usage):
        if False:
            for i in range(10):
                print('nop')
        self.Usage = Usage

    def GetUsage(self):
        if False:
            while True:
                i = 10
        return self.Usage

class InfHobObject:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.HobType = ''
        self.Usage = ''
        self.SupArchList = []
        self.HelpString = ''

    def SetHobType(self, HobType):
        if False:
            for i in range(10):
                print('nop')
        self.HobType = HobType

    def GetHobType(self):
        if False:
            i = 10
            return i + 15
        return self.HobType

    def SetUsage(self, Usage):
        if False:
            i = 10
            return i + 15
        self.Usage = Usage

    def GetUsage(self):
        if False:
            print('Hello World!')
        return self.Usage

    def SetSupArchList(self, ArchList):
        if False:
            for i in range(10):
                print('nop')
        self.SupArchList = ArchList

    def GetSupArchList(self):
        if False:
            for i in range(10):
                print('nop')
        return self.SupArchList

    def SetHelpString(self, HelpString):
        if False:
            return 10
        self.HelpString = HelpString

    def GetHelpString(self):
        if False:
            i = 10
            return i + 15
        return self.HelpString

class InfSpecialCommentObject(InfSectionCommonDef):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.SpecialComments = Sdict()
        InfSectionCommonDef.__init__(self)

    def SetSpecialComments(self, SepcialSectionList=None, Type=''):
        if False:
            print('Hello World!')
        if Type == DT.TYPE_HOB_SECTION or Type == DT.TYPE_EVENT_SECTION or Type == DT.TYPE_BOOTMODE_SECTION:
            for Item in SepcialSectionList:
                if Type in self.SpecialComments:
                    ObjList = self.SpecialComments[Type]
                    ObjList.append(Item)
                    self.SpecialComments[Type] = ObjList
                else:
                    ObjList = []
                    ObjList.append(Item)
                    self.SpecialComments[Type] = ObjList
        return True

    def GetSpecialComments(self):
        if False:
            return 10
        return self.SpecialComments

def ErrorInInf(Message=None, ErrorCode=None, LineInfo=None, RaiseError=True):
    if False:
        return 10
    if ErrorCode is None:
        ErrorCode = ToolError.FORMAT_INVALID
    if LineInfo is None:
        LineInfo = ['', -1, '']
    Logger.Error('InfParser', ErrorCode, Message=Message, File=LineInfo[0], Line=LineInfo[1], ExtraData=LineInfo[2], RaiseError=RaiseError)