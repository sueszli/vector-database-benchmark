"""
Dependency
"""
from os.path import dirname
import os
import Logger.Log as Logger
from Logger import StringTable as ST
from Library.Parsing import GetWorkspacePackage
from Library.Parsing import GetWorkspaceModule
from Library.Parsing import GetPkgInfoFromDec
from Library.Misc import GetRelativePath
from Library import GlobalData
from Logger.ToolError import FatalError
from Logger.ToolError import EDK1_INF_ERROR
from Logger.ToolError import UNKNOWN_ERROR
(DEPEX_CHECK_SUCCESS, DEPEX_CHECK_MODULE_NOT_FOUND, DEPEX_CHECK_PACKAGE_NOT_FOUND, DEPEX_CHECK_DP_NOT_FOUND) = (0, 1, 2, 3)

class DependencyRules(object):

    def __init__(self, Datab, ToBeInstalledPkgList=None):
        if False:
            i = 10
            return i + 15
        self.IpiDb = Datab
        self.WsPkgList = GetWorkspacePackage()
        self.WsModuleList = GetWorkspaceModule()
        self.PkgsToBeDepend = [(PkgInfo[1], PkgInfo[2]) for PkgInfo in self.WsPkgList]
        self.PkgsToBeDepend.extend(self.GenToBeInstalledPkgList(ToBeInstalledPkgList))

    def GenToBeInstalledPkgList(self, ToBeInstalledPkgList):
        if False:
            while True:
                i = 10
        if not ToBeInstalledPkgList:
            return []
        RtnList = []
        for Dist in ToBeInstalledPkgList:
            for Package in Dist.PackageSurfaceArea:
                RtnList.append((Package[0], Package[1]))
        return RtnList

    def CheckModuleExists(self, Guid, Version, Name, Path):
        if False:
            return 10
        Logger.Verbose(ST.MSG_CHECK_MODULE_EXIST)
        ModuleList = self.IpiDb.GetModInPackage(Guid, Version, Name, Path)
        ModuleList.extend(self.IpiDb.GetStandaloneModule(Guid, Version, Name, Path))
        Logger.Verbose(ST.MSG_CHECK_MODULE_EXIST_FINISH)
        if len(ModuleList) > 0:
            return True
        else:
            return False

    def CheckModuleDepexSatisfied(self, ModuleObj, DpObj=None):
        if False:
            while True:
                i = 10
        Logger.Verbose(ST.MSG_CHECK_MODULE_DEPEX_START)
        Result = True
        Dep = None
        if ModuleObj.GetPackageDependencyList():
            Dep = ModuleObj.GetPackageDependencyList()[0]
        for Dep in ModuleObj.GetPackageDependencyList():
            Exist = self.CheckPackageExists(Dep.GetGuid(), Dep.GetVersion())
            if not Exist:
                if DpObj is None:
                    Result = False
                    break
                for GuidVerPair in DpObj.PackageSurfaceArea.keys():
                    if Dep.GetGuid() == GuidVerPair[0]:
                        if Dep.GetVersion() is None or len(Dep.GetVersion()) == 0:
                            Result = True
                            break
                        if Dep.GetVersion() == GuidVerPair[1]:
                            Result = True
                            break
                else:
                    Result = False
                    break
        if not Result:
            Logger.Error('CheckModuleDepex', UNKNOWN_ERROR, ST.ERR_DEPENDENCY_NOT_MATCH % (ModuleObj.GetName(), Dep.GetPackageFilePath(), Dep.GetGuid(), Dep.GetVersion()))
        return Result

    def CheckPackageExists(self, Guid, Version):
        if False:
            while True:
                i = 10
        Logger.Verbose(ST.MSG_CHECK_PACKAGE_START)
        Found = False
        for (PkgGuid, PkgVer) in self.PkgsToBeDepend:
            if PkgGuid == Guid:
                if Version and PkgVer != Version:
                    Found = False
                    break
                else:
                    Found = True
                    break
        else:
            Found = False
        Logger.Verbose(ST.MSG_CHECK_PACKAGE_FINISH)
        return Found

    def CheckPackageDepexSatisfied(self, PkgObj, DpObj=None):
        if False:
            while True:
                i = 10
        ModuleDict = PkgObj.GetModuleDict()
        for ModKey in ModuleDict.keys():
            ModObj = ModuleDict[ModKey]
            if self.CheckModuleDepexSatisfied(ModObj, DpObj):
                continue
            else:
                return False
        return True

    def CheckDpExists(self, Guid, Version):
        if False:
            print('Hello World!')
        Logger.Verbose(ST.MSG_CHECK_DP_START)
        DpList = self.IpiDb.GetDp(Guid, Version)
        if len(DpList) > 0:
            Found = True
        else:
            Found = False
        Logger.Verbose(ST.MSG_CHECK_DP_FINISH)
        return Found

    def CheckInstallDpDepexSatisfied(self, DpObj):
        if False:
            i = 10
            return i + 15
        return self.CheckDpDepexSatisfied(DpObj)

    def CheckTestInstallPdDepexSatisfied(self, DpObjList):
        if False:
            for i in range(10):
                print('nop')
        for DpObj in DpObjList:
            if self.CheckDpDepexSatisfied(DpObj):
                for PkgKey in DpObj.PackageSurfaceArea.keys():
                    PkgObj = DpObj.PackageSurfaceArea[PkgKey]
                    self.PkgsToBeDepend.append((PkgObj.Guid, PkgObj.Version))
            else:
                return (False, DpObj)
        return (True, DpObj)

    def ReplaceCheckNewDpDepex(self, DpObj, OrigDpGuid, OrigDpVersion):
        if False:
            print('Hello World!')
        self.PkgsToBeDepend = [(PkgInfo[1], PkgInfo[2]) for PkgInfo in self.WsPkgList]
        OrigDpPackageList = self.IpiDb.GetPackageListFromDp(OrigDpGuid, OrigDpVersion)
        for OrigPkgInfo in OrigDpPackageList:
            (Guid, Version) = (OrigPkgInfo[0], OrigPkgInfo[1])
            if (Guid, Version) in self.PkgsToBeDepend:
                self.PkgsToBeDepend.remove((Guid, Version))
        return self.CheckDpDepexSatisfied(DpObj)

    def CheckDpDepexSatisfied(self, DpObj):
        if False:
            return 10
        for PkgKey in DpObj.PackageSurfaceArea.keys():
            PkgObj = DpObj.PackageSurfaceArea[PkgKey]
            if self.CheckPackageDepexSatisfied(PkgObj, DpObj):
                continue
            else:
                return False
        for ModKey in DpObj.ModuleSurfaceArea.keys():
            ModObj = DpObj.ModuleSurfaceArea[ModKey]
            if self.CheckModuleDepexSatisfied(ModObj, DpObj):
                continue
            else:
                return False
        return True

    def CheckDpDepexForRemove(self, DpGuid, DpVersion):
        if False:
            while True:
                i = 10
        Removable = True
        DependModuleList = []
        WsModuleList = self.WsModuleList
        DpModuleList = self.IpiDb.GetDpModuleList(DpGuid, DpVersion)
        for Module in DpModuleList:
            if Module in WsModuleList:
                WsModuleList.remove(Module)
            else:
                Logger.Warn('UPT\n', ST.ERR_MODULE_NOT_INSTALLED % Module)
        DpPackageList = self.IpiDb.GetPackageListFromDp(DpGuid, DpVersion)
        DpPackagePathList = []
        WorkSP = GlobalData.gWORKSPACE
        for (PkgName, PkgGuid, PkgVersion, DecFile) in self.WsPkgList:
            if PkgName:
                pass
            DecPath = dirname(DecFile)
            if DecPath.find(WorkSP) > -1:
                InstallPath = GetRelativePath(DecPath, WorkSP)
                DecFileRelaPath = GetRelativePath(DecFile, WorkSP)
            else:
                InstallPath = DecPath
                DecFileRelaPath = DecFile
            if (PkgGuid, PkgVersion, InstallPath) in DpPackageList:
                DpPackagePathList.append(DecFileRelaPath)
                DpPackageList.remove((PkgGuid, PkgVersion, InstallPath))
        for (PkgGuid, PkgVersion, InstallPath) in DpPackageList:
            Logger.Warn('UPT', ST.WARN_INSTALLED_PACKAGE_NOT_FOUND % (PkgGuid, PkgVersion, InstallPath))
        for Module in WsModuleList:
            if not VerifyRemoveModuleDep(Module, DpPackagePathList):
                Removable = False
                DependModuleList.append(Module)
        return (Removable, DependModuleList)

    def CheckDpDepexForReplace(self, OrigDpGuid, OrigDpVersion, NewDpPkgList):
        if False:
            return 10
        Replaceable = True
        DependModuleList = []
        WsModuleList = self.WsModuleList
        DpModuleList = self.IpiDb.GetDpModuleList(OrigDpGuid, OrigDpVersion)
        for Module in DpModuleList:
            if Module in WsModuleList:
                WsModuleList.remove(Module)
            else:
                Logger.Warn('UPT\n', ST.ERR_MODULE_NOT_INSTALLED % Module)
        OtherPkgList = NewDpPkgList
        DpPackageList = self.IpiDb.GetPackageListFromDp(OrigDpGuid, OrigDpVersion)
        DpPackagePathList = []
        WorkSP = GlobalData.gWORKSPACE
        for (PkgName, PkgGuid, PkgVersion, DecFile) in self.WsPkgList:
            if PkgName:
                pass
            DecPath = dirname(DecFile)
            if DecPath.find(WorkSP) > -1:
                InstallPath = GetRelativePath(DecPath, WorkSP)
                DecFileRelaPath = GetRelativePath(DecFile, WorkSP)
            else:
                InstallPath = DecPath
                DecFileRelaPath = DecFile
            if (PkgGuid, PkgVersion, InstallPath) in DpPackageList:
                DpPackagePathList.append(DecFileRelaPath)
                DpPackageList.remove((PkgGuid, PkgVersion, InstallPath))
            else:
                OtherPkgList.append((PkgGuid, PkgVersion))
        for (PkgGuid, PkgVersion, InstallPath) in DpPackageList:
            Logger.Warn('UPT', ST.WARN_INSTALLED_PACKAGE_NOT_FOUND % (PkgGuid, PkgVersion, InstallPath))
        for Module in WsModuleList:
            if not VerifyReplaceModuleDep(Module, DpPackagePathList, OtherPkgList):
                Replaceable = False
                DependModuleList.append(Module)
        return (Replaceable, DependModuleList)

def VerifyRemoveModuleDep(Path, DpPackagePathList):
    if False:
        for i in range(10):
            print('nop')
    try:
        for Item in GetPackagePath(Path):
            if Item in DpPackagePathList:
                DecPath = os.path.normpath(os.path.join(GlobalData.gWORKSPACE, Item))
                Logger.Info(ST.MSG_MODULE_DEPEND_ON % (Path, DecPath))
                return False
        else:
            return True
    except FatalError as ErrCode:
        if ErrCode.message == EDK1_INF_ERROR:
            Logger.Warn('UPT', ST.WRN_EDK1_INF_FOUND % Path)
            return True
        else:
            return True

def GetPackagePath(InfPath):
    if False:
        for i in range(10):
            print('nop')
    PackagePath = []
    if os.path.exists(InfPath):
        FindSection = False
        for Line in open(InfPath).readlines():
            Line = Line.strip()
            if not Line:
                continue
            if Line.startswith('#'):
                continue
            if Line.startswith('[Packages') and Line.endswith(']'):
                FindSection = True
                continue
            if Line.startswith('[') and Line.endswith(']') and FindSection:
                break
            if FindSection:
                PackagePath.append(os.path.normpath(Line))
    return PackagePath

def VerifyReplaceModuleDep(Path, DpPackagePathList, OtherPkgList):
    if False:
        return 10
    try:
        for Item in GetPackagePath(Path):
            if Item in DpPackagePathList:
                DecPath = os.path.normpath(os.path.join(GlobalData.gWORKSPACE, Item))
                (Name, Guid, Version) = GetPkgInfoFromDec(DecPath)
                if (Guid, Version) not in OtherPkgList:
                    Logger.Info(ST.MSG_MODULE_DEPEND_ON % (Path, DecPath))
                    return False
        else:
            return True
    except FatalError as ErrCode:
        if ErrCode.message == EDK1_INF_ERROR:
            Logger.Warn('UPT', ST.WRN_EDK1_INF_FOUND % Path)
            return True
        else:
            return True