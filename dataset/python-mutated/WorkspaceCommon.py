from __future__ import absolute_import
from collections import OrderedDict, defaultdict
from Common.DataType import SUP_MODULE_USER_DEFINED
from Common.DataType import SUP_MODULE_HOST_APPLICATION
from .BuildClassObject import LibraryClassObject
import Common.GlobalData as GlobalData
from Workspace.BuildClassObject import StructurePcd
from Common.BuildToolError import RESOURCE_NOT_AVAILABLE
from Common.BuildToolError import OPTION_MISSING
from Common.BuildToolError import BUILD_ERROR
import Common.EdkLogger as EdkLogger

class OrderedListDict(OrderedDict):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(OrderedListDict, self).__init__(*args, **kwargs)
        self.default_factory = list

    def __missing__(self, key):
        if False:
            print('Hello World!')
        self[key] = Value = self.default_factory()
        return Value

def GetPackageList(Platform, BuildDatabase, Arch, Target, Toolchain):
    if False:
        return 10
    PkgSet = set()
    if Platform.Packages:
        PkgSet.update(Platform.Packages)
    for ModuleFile in Platform.Modules:
        Data = BuildDatabase[ModuleFile, Arch, Target, Toolchain]
        PkgSet.update(Data.Packages)
        for Lib in GetLiabraryInstances(Data, Platform, BuildDatabase, Arch, Target, Toolchain):
            PkgSet.update(Lib.Packages)
    return list(PkgSet)

def GetDeclaredPcd(Platform, BuildDatabase, Arch, Target, Toolchain, additionalPkgs):
    if False:
        for i in range(10):
            print('nop')
    PkgList = GetPackageList(Platform, BuildDatabase, Arch, Target, Toolchain)
    PkgList = set(PkgList)
    PkgList |= additionalPkgs
    DecPcds = {}
    GuidDict = {}
    for Pkg in PkgList:
        Guids = Pkg.Guids
        GuidDict.update(Guids)
        for Pcd in Pkg.Pcds:
            PcdCName = Pcd[0]
            PcdTokenName = Pcd[1]
            if GlobalData.MixedPcd:
                for PcdItem in GlobalData.MixedPcd:
                    if (PcdCName, PcdTokenName) in GlobalData.MixedPcd[PcdItem]:
                        PcdCName = PcdItem[0]
                        break
            if (PcdCName, PcdTokenName) not in DecPcds:
                DecPcds[PcdCName, PcdTokenName] = Pkg.Pcds[Pcd]
    return (DecPcds, GuidDict)

def GetLiabraryInstances(Module, Platform, BuildDatabase, Arch, Target, Toolchain):
    if False:
        for i in range(10):
            print('nop')
    return GetModuleLibInstances(Module, Platform, BuildDatabase, Arch, Target, Toolchain, Platform.MetaFile, EdkLogger)

def GetModuleLibInstances(Module, Platform, BuildDatabase, Arch, Target, Toolchain, FileName='', EdkLogger=None):
    if False:
        while True:
            i = 10
    if Module.LibInstances:
        return Module.LibInstances
    ModuleType = Module.ModuleType
    if Module.ModuleType != SUP_MODULE_USER_DEFINED:
        for LibraryClass in Platform.LibraryClasses.GetKeys():
            if LibraryClass.startswith('NULL') and Platform.LibraryClasses[LibraryClass, Module.ModuleType]:
                Module.LibraryClasses[LibraryClass] = Platform.LibraryClasses[LibraryClass, Module.ModuleType]
    for LibraryClass in Platform.Modules[str(Module)].LibraryClasses:
        if LibraryClass.startswith('NULL'):
            Module.LibraryClasses[LibraryClass] = Platform.Modules[str(Module)].LibraryClasses[LibraryClass]
    LibraryConsumerList = [Module]
    Constructor = []
    ConsumedByList = OrderedListDict()
    LibraryInstance = OrderedDict()
    if not Module.LibraryClass:
        EdkLogger.verbose('')
        EdkLogger.verbose('Library instances of module [%s] [%s]:' % (str(Module), Arch))
    while len(LibraryConsumerList) > 0:
        M = LibraryConsumerList.pop()
        for LibraryClassName in M.LibraryClasses:
            if LibraryClassName not in LibraryInstance:
                LibraryPath = Platform.Modules[str(Module)].LibraryClasses.get(LibraryClassName, Platform.LibraryClasses[LibraryClassName, ModuleType])
                if LibraryPath is None:
                    LibraryPath = M.LibraryClasses.get(LibraryClassName)
                    if LibraryPath is None:
                        if not Module.LibraryClass:
                            EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'Instance of library class [%s] is not found' % LibraryClassName, File=FileName, ExtraData='in [%s] [%s]\n\tconsumed by module [%s]' % (str(M), Arch, str(Module)))
                        else:
                            return []
                LibraryModule = BuildDatabase[LibraryPath, Arch, Target, Toolchain]
                if LibraryClassName.startswith('NULL'):
                    LibraryModule.LibraryClass.append(LibraryClassObject(LibraryClassName, [ModuleType]))
                elif LibraryModule.LibraryClass is None or len(LibraryModule.LibraryClass) == 0 or (ModuleType != SUP_MODULE_USER_DEFINED and ModuleType != SUP_MODULE_HOST_APPLICATION and (ModuleType not in LibraryModule.LibraryClass[0].SupModList)):
                    if not Module.LibraryClass:
                        EdkLogger.error('build', OPTION_MISSING, 'Module type [%s] is not supported by library instance [%s]' % (ModuleType, LibraryPath), File=FileName, ExtraData='consumed by library instance [%s] which is consumed by module [%s]' % (str(M), str(Module)))
                    else:
                        return []
                LibraryInstance[LibraryClassName] = LibraryModule
                LibraryConsumerList.append(LibraryModule)
                if not Module.LibraryClass:
                    EdkLogger.verbose('\t' + str(LibraryClassName) + ' : ' + str(LibraryModule))
            else:
                LibraryModule = LibraryInstance[LibraryClassName]
            if LibraryModule is None:
                continue
            if LibraryModule.ConstructorList != [] and LibraryModule not in Constructor:
                Constructor.append(LibraryModule)
            if M != Module:
                if M in ConsumedByList[LibraryModule]:
                    continue
                ConsumedByList[LibraryModule].append(M)
    SortedLibraryList = []
    LibraryList = []
    Q = []
    for LibraryClassName in LibraryInstance:
        M = LibraryInstance[LibraryClassName]
        LibraryList.append(M)
        if not ConsumedByList[M]:
            Q.append(M)
    while True:
        EdgeRemoved = True
        while Q == [] and EdgeRemoved:
            EdgeRemoved = False
            for Item in LibraryList:
                if Item not in Constructor:
                    continue
                for Node in ConsumedByList[Item]:
                    if Node in Constructor:
                        continue
                    ConsumedByList[Item].remove(Node)
                    EdgeRemoved = True
                    if not ConsumedByList[Item]:
                        Q.insert(0, Item)
                        break
                if Q != []:
                    break
        if Q == []:
            break
        Node = Q.pop()
        SortedLibraryList.append(Node)
        for Item in LibraryList:
            if Node not in ConsumedByList[Item]:
                continue
            ConsumedByList[Item].remove(Node)
            if ConsumedByList[Item]:
                continue
            Q.insert(0, Item)
    for Item in LibraryList:
        if ConsumedByList[Item] and Item in Constructor and (len(Constructor) > 1):
            if not Module.LibraryClass:
                ErrorMessage = '\tconsumed by ' + '\n\tconsumed by '.join((str(L) for L in ConsumedByList[Item]))
                EdkLogger.error('build', BUILD_ERROR, 'Library [%s] with constructors has a cycle' % str(Item), ExtraData=ErrorMessage, File=FileName)
            else:
                return []
        if Item not in SortedLibraryList:
            SortedLibraryList.append(Item)
    SortedLibraryList.reverse()
    Module.LibInstances = SortedLibraryList
    SortedLibraryList = [lib.SetReferenceModule(Module) for lib in SortedLibraryList]
    return SortedLibraryList