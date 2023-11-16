"""
InfUserExtensionsObject
"""
from Logger import StringTable as ST
from Logger import ToolError
import Logger.Log as Logger
from Library import GlobalData
from Library.Misc import Sdict

class InfUserExtensionItem:

    def __init__(self, Content='', UserId='', IdString=''):
        if False:
            i = 10
            return i + 15
        self.Content = Content
        self.UserId = UserId
        self.IdString = IdString
        self.SupArchList = []

    def SetContent(self, Content):
        if False:
            print('Hello World!')
        self.Content = Content

    def GetContent(self):
        if False:
            return 10
        return self.Content

    def SetUserId(self, UserId):
        if False:
            for i in range(10):
                print('nop')
        self.UserId = UserId

    def GetUserId(self):
        if False:
            return 10
        return self.UserId

    def SetIdString(self, IdString):
        if False:
            return 10
        self.IdString = IdString

    def GetIdString(self):
        if False:
            return 10
        return self.IdString

    def SetSupArchList(self, SupArchList):
        if False:
            while True:
                i = 10
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            print('Hello World!')
        return self.SupArchList

class InfUserExtensionObject:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.UserExtension = Sdict()

    def SetUserExtension(self, UserExtensionCont, IdContent=None, LineNo=None):
        if False:
            while True:
                i = 10
        if not UserExtensionCont or UserExtensionCont == '':
            return True
        for IdContentItem in IdContent:
            InfUserExtensionItemObj = InfUserExtensionItem()
            if IdContentItem[0] == 'COMMON':
                UserId = ''
            else:
                UserId = IdContentItem[0]
            if IdContentItem[1] == 'COMMON':
                IdString = ''
            else:
                IdString = IdContentItem[1]
            InfUserExtensionItemObj.SetUserId(UserId)
            InfUserExtensionItemObj.SetIdString(IdString)
            InfUserExtensionItemObj.SetContent(UserExtensionCont)
            InfUserExtensionItemObj.SetSupArchList(IdContentItem[2])
            if IdContentItem in self.UserExtension:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_UE_SECTION_DUPLICATE_ERROR % (IdContentItem[0] + '.' + IdContentItem[1] + '.' + IdContentItem[2]), File=GlobalData.gINF_MODULE_NAME, Line=LineNo, ExtraData=None)
            else:
                UserExtensionList = []
                UserExtensionList.append(InfUserExtensionItemObj)
                self.UserExtension[IdContentItem] = UserExtensionList
        return True

    def GetUserExtension(self):
        if False:
            i = 10
            return i + 15
        return self.UserExtension