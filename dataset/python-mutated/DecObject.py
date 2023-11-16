"""
DecObject
"""
import os.path
from Library.Misc import Sdict
from Library.DataType import TAB_GUIDS
from Library.DataType import TAB_PPIS
from Library.DataType import TAB_PROTOCOLS
from Library.DataType import TAB_DEC_DEFINES
from Library.DataType import TAB_INCLUDES
from Library.DataType import TAB_LIBRARY_CLASSES
from Library.DataType import TAB_USER_EXTENSIONS
from Library.DataType import TAB_PCDS
from Library.DataType import TAB_ARCH_COMMON

class _DecComments:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._HeadComment = []
        self._TailComment = []

    def GetComments(self):
        if False:
            for i in range(10):
                print('nop')
        return (self._HeadComment, self._TailComment)

    def GetHeadComment(self):
        if False:
            for i in range(10):
                print('nop')
        return self._HeadComment

    def SetHeadComment(self, Comment):
        if False:
            print('Hello World!')
        self._HeadComment = Comment

    def GetTailComment(self):
        if False:
            return 10
        return self._TailComment

    def SetTailComment(self, Comment):
        if False:
            i = 10
            return i + 15
        self._TailComment = Comment

class _DecBaseObject(_DecComments):

    def __init__(self, PkgFullName):
        if False:
            i = 10
            return i + 15
        _DecComments.__init__(self)
        self.ValueDict = Sdict()
        self._PkgFullName = PkgFullName
        (self._PackagePath, self._FileName) = os.path.split(PkgFullName)
        self._SecName = ''

    def GetSectionName(self):
        if False:
            for i in range(10):
                print('nop')
        return self._SecName

    def GetPackagePath(self):
        if False:
            print('Hello World!')
        return self._PackagePath

    def GetPackageFile(self):
        if False:
            i = 10
            return i + 15
        return self._FileName

    def GetPackageFullName(self):
        if False:
            for i in range(10):
                print('nop')
        return self._PkgFullName

    def AddItem(self, Item, Scope):
        if False:
            return 10
        if not Scope:
            return
        if not Item:
            return
        ArchModule = []
        for Ele in Scope:
            if Ele[1] in self.ValueDict:
                self.ValueDict[Ele[1]].append(Item)
            else:
                self.ValueDict[Ele[1]] = [Item]
            ArchModule.append(Ele[1])
        Item.ArchAndModuleType = ArchModule

    def _GetItemByArch(self, Arch):
        if False:
            print('Hello World!')
        Arch = Arch.upper()
        if Arch not in self.ValueDict:
            return []
        return self.ValueDict[Arch]

    def _GetAllItems(self):
        if False:
            return 10
        Retlst = []
        for Arch in self.ValueDict:
            for Item in self.ValueDict[Arch]:
                if Item not in Retlst:
                    Retlst.append(Item)
        return Retlst

class _DecItemBaseObject(_DecComments):

    def __init__(self):
        if False:
            while True:
                i = 10
        _DecComments.__init__(self)
        self.ArchAndModuleType = []

    def GetArchList(self):
        if False:
            return 10
        ArchSet = set()
        for Arch in self.ArchAndModuleType:
            ArchSet.add(Arch)
        return list(ArchSet)

class DecDefineObject(_DecBaseObject):

    def __init__(self, PkgFullName):
        if False:
            return 10
        _DecBaseObject.__init__(self, PkgFullName)
        self._SecName = TAB_DEC_DEFINES.upper()
        self._DecSpec = ''
        self._PkgName = ''
        self._PkgGuid = ''
        self._PkgVersion = ''
        self._PkgUniFile = ''

    def GetPackageSpecification(self):
        if False:
            i = 10
            return i + 15
        return self._DecSpec

    def SetPackageSpecification(self, DecSpec):
        if False:
            for i in range(10):
                print('nop')
        self._DecSpec = DecSpec

    def GetPackageName(self):
        if False:
            i = 10
            return i + 15
        return self._PkgName

    def SetPackageName(self, PkgName):
        if False:
            return 10
        self._PkgName = PkgName

    def GetPackageGuid(self):
        if False:
            return 10
        return self._PkgGuid

    def SetPackageGuid(self, PkgGuid):
        if False:
            i = 10
            return i + 15
        self._PkgGuid = PkgGuid

    def GetPackageVersion(self):
        if False:
            i = 10
            return i + 15
        return self._PkgVersion

    def SetPackageVersion(self, PkgVersion):
        if False:
            print('Hello World!')
        self._PkgVersion = PkgVersion

    def GetPackageUniFile(self):
        if False:
            return 10
        return self._PkgUniFile

    def SetPackageUniFile(self, PkgUniFile):
        if False:
            return 10
        self._PkgUniFile = PkgUniFile

    def GetDefines(self):
        if False:
            return 10
        return self._GetItemByArch(TAB_ARCH_COMMON)

    def GetAllDefines(self):
        if False:
            while True:
                i = 10
        return self._GetAllItems()

class DecDefineItemObject(_DecItemBaseObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        _DecItemBaseObject.__init__(self)
        self.Key = ''
        self.Value = ''

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.Key + self.Value)

    def __eq__(self, Other):
        if False:
            print('Hello World!')
        return id(self) == id(Other)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.ArchAndModuleType) + '\n' + self.Key + ' = ' + self.Value

class DecIncludeObject(_DecBaseObject):

    def __init__(self, PkgFullName):
        if False:
            while True:
                i = 10
        _DecBaseObject.__init__(self, PkgFullName)
        self._SecName = TAB_INCLUDES.upper()

    def GetIncludes(self, Arch=TAB_ARCH_COMMON):
        if False:
            while True:
                i = 10
        return self._GetItemByArch(Arch)

    def GetAllIncludes(self):
        if False:
            while True:
                i = 10
        return self._GetAllItems()

class DecIncludeItemObject(_DecItemBaseObject):

    def __init__(self, File, Root):
        if False:
            print('Hello World!')
        self.File = File
        self.Root = Root
        _DecItemBaseObject.__init__(self)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.File)

    def __eq__(self, Other):
        if False:
            return 10
        return id(self) == id(Other)

    def __str__(self):
        if False:
            return 10
        return self.File

class DecLibraryclassObject(_DecBaseObject):

    def __init__(self, PkgFullName):
        if False:
            i = 10
            return i + 15
        _DecBaseObject.__init__(self, PkgFullName)
        (self._PackagePath, self._FileName) = os.path.split(PkgFullName)
        self._SecName = TAB_LIBRARY_CLASSES.upper()

    def GetLibraryclasses(self, Arch=TAB_ARCH_COMMON):
        if False:
            for i in range(10):
                print('nop')
        return self._GetItemByArch(Arch)

    def GetAllLibraryclasses(self):
        if False:
            i = 10
            return i + 15
        return self._GetAllItems()

class DecLibraryclassItemObject(_DecItemBaseObject):

    def __init__(self, Libraryclass, File, Root):
        if False:
            for i in range(10):
                print('nop')
        _DecItemBaseObject.__init__(self)
        self.File = File
        self.Root = Root
        self.Libraryclass = Libraryclass

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.Libraryclass + self.File)

    def __eq__(self, Other):
        if False:
            while True:
                i = 10
        return id(self) == id(Other)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.Libraryclass + '|' + self.File

class DecPcdObject(_DecBaseObject):

    def __init__(self, PkgFullName):
        if False:
            for i in range(10):
                print('nop')
        _DecBaseObject.__init__(self, PkgFullName)
        self._SecName = TAB_PCDS.upper()

    def AddItem(self, Item, Scope):
        if False:
            i = 10
            return i + 15
        if not Scope:
            return
        if not Item:
            return
        ArchModule = []
        for (Type, Arch) in Scope:
            if (Type, Arch) in self.ValueDict:
                self.ValueDict[Type, Arch].append(Item)
            else:
                self.ValueDict[Type, Arch] = [Item]
            ArchModule.append([Type, Arch])
        Item.ArchAndModuleType = ArchModule

    def GetPcds(self, PcdType, Arch=TAB_ARCH_COMMON):
        if False:
            for i in range(10):
                print('nop')
        PcdType = PcdType.upper()
        Arch = Arch.upper()
        if (PcdType, Arch) not in self.ValueDict:
            return []
        return self.ValueDict[PcdType, Arch]

    def GetPcdsByType(self, PcdType):
        if False:
            return 10
        PcdType = PcdType.upper()
        Retlst = []
        for (TypeInDict, Arch) in self.ValueDict:
            if TypeInDict != PcdType:
                continue
            for Item in self.ValueDict[PcdType, Arch]:
                if Item not in Retlst:
                    Retlst.append(Item)
        return Retlst

class DecPcdItemObject(_DecItemBaseObject):

    def __init__(self, Guid, Name, Value, DatumType, Token, MaxDatumSize=''):
        if False:
            while True:
                i = 10
        _DecItemBaseObject.__init__(self)
        self.TokenCName = Name
        self.TokenSpaceGuidCName = Guid
        self.DatumType = DatumType
        self.DefaultValue = Value
        self.TokenValue = Token
        self.MaxDatumSize = MaxDatumSize

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.TokenSpaceGuidCName + self.TokenCName)

    def __eq__(self, Other):
        if False:
            print('Hello World!')
        return id(self) == id(Other)

    def GetArchListOfType(self, PcdType):
        if False:
            print('Hello World!')
        ItemSet = set()
        PcdType = PcdType.upper()
        for (Type, Arch) in self.ArchAndModuleType:
            if Type != PcdType:
                continue
            ItemSet.add(Arch)
        return list(ItemSet)

class DecGuidObjectBase(_DecBaseObject):

    def __init__(self, PkgFullName):
        if False:
            i = 10
            return i + 15
        _DecBaseObject.__init__(self, PkgFullName)

    def GetGuidStyleItems(self, Arch=TAB_ARCH_COMMON):
        if False:
            i = 10
            return i + 15
        return self._GetItemByArch(Arch)

    def GetGuidStyleAllItems(self):
        if False:
            return 10
        return self._GetAllItems()

class DecGuidItemObject(_DecItemBaseObject):

    def __init__(self, CName, GuidCValue, GuidString):
        if False:
            return 10
        _DecItemBaseObject.__init__(self)
        self.GuidCName = CName
        self.GuidCValue = GuidCValue
        self.GuidString = GuidString

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.GuidCName)

    def __eq__(self, Other):
        if False:
            return 10
        return id(self) == id(Other)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.GuidCName + ' = ' + self.GuidCValue

class DecGuidObject(DecGuidObjectBase):

    def __init__(self, PkgFullName):
        if False:
            i = 10
            return i + 15
        DecGuidObjectBase.__init__(self, PkgFullName)
        self._SecName = TAB_GUIDS.upper()

    def GetGuids(self, Arch=TAB_ARCH_COMMON):
        if False:
            return 10
        return self._GetItemByArch(Arch)

    def GetAllGuids(self):
        if False:
            print('Hello World!')
        return self._GetAllItems()

class DecPpiObject(DecGuidObjectBase):

    def __init__(self, PkgFullName):
        if False:
            for i in range(10):
                print('nop')
        DecGuidObjectBase.__init__(self, PkgFullName)
        self._SecName = TAB_PPIS.upper()

    def GetPpis(self, Arch=TAB_ARCH_COMMON):
        if False:
            print('Hello World!')
        return self._GetItemByArch(Arch)

    def GetAllPpis(self):
        if False:
            print('Hello World!')
        return self._GetAllItems()

class DecProtocolObject(DecGuidObjectBase):

    def __init__(self, PkgFullName):
        if False:
            return 10
        DecGuidObjectBase.__init__(self, PkgFullName)
        self._SecName = TAB_PROTOCOLS.upper()

    def GetProtocols(self, Arch=TAB_ARCH_COMMON):
        if False:
            while True:
                i = 10
        return self._GetItemByArch(Arch)

    def GetAllProtocols(self):
        if False:
            for i in range(10):
                print('nop')
        return self._GetAllItems()

class DecUserExtensionObject(_DecBaseObject):

    def __init__(self, PkgFullName):
        if False:
            while True:
                i = 10
        _DecBaseObject.__init__(self, PkgFullName)
        self._SecName = TAB_USER_EXTENSIONS.upper()
        self.ItemList = []

    def AddItem(self, Item, Scope):
        if False:
            i = 10
            return i + 15
        if not Scope:
            pass
        if not Item:
            return
        self.ItemList.append(Item)

    def GetAllUserExtensions(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ItemList

class DecUserExtensionItemObject(_DecItemBaseObject):

    def __init__(self):
        if False:
            print('Hello World!')
        _DecItemBaseObject.__init__(self)
        self.UserString = ''
        self.UserId = ''
        self.IdString = ''