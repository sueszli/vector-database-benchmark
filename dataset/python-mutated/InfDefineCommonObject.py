"""
InfDefineCommonObject
"""
from Object.Parser.InfCommonObject import InfLineCommentObject

class InfDefineImageExeParamItem:

    def __init__(self):
        if False:
            return 10
        self.CName = ''
        self.FeatureFlagExp = ''
        self.Comments = InfLineCommentObject()

    def SetCName(self, CName):
        if False:
            return 10
        self.CName = CName

    def GetCName(self):
        if False:
            return 10
        return self.CName

    def SetFeatureFlagExp(self, FeatureFlagExp):
        if False:
            while True:
                i = 10
        self.FeatureFlagExp = FeatureFlagExp

    def GetFeatureFlagExp(self):
        if False:
            for i in range(10):
                print('nop')
        return self.FeatureFlagExp

class InfDefineEntryPointItem(InfDefineImageExeParamItem):

    def __init__(self):
        if False:
            while True:
                i = 10
        InfDefineImageExeParamItem.__init__(self)

class InfDefineUnloadImageItem(InfDefineImageExeParamItem):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        InfDefineImageExeParamItem.__init__(self)

class InfDefineConstructorItem(InfDefineImageExeParamItem):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        InfDefineImageExeParamItem.__init__(self)
        self.SupModList = []

    def SetSupModList(self, SupModList):
        if False:
            for i in range(10):
                print('nop')
        self.SupModList = SupModList

    def GetSupModList(self):
        if False:
            while True:
                i = 10
        return self.SupModList

class InfDefineDestructorItem(InfDefineImageExeParamItem):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        InfDefineImageExeParamItem.__init__(self)
        self.SupModList = []

    def SetSupModList(self, SupModList):
        if False:
            return 10
        self.SupModList = SupModList

    def GetSupModList(self):
        if False:
            for i in range(10):
                print('nop')
        return self.SupModList

class InfDefineLibraryItem:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.LibraryName = ''
        self.Types = []
        self.Comments = InfLineCommentObject()

    def SetLibraryName(self, Name):
        if False:
            for i in range(10):
                print('nop')
        self.LibraryName = Name

    def GetLibraryName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.LibraryName

    def SetTypes(self, Type):
        if False:
            i = 10
            return i + 15
        self.Types = Type

    def GetTypes(self):
        if False:
            return 10
        return self.Types