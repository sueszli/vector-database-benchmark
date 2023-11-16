"""
DistributionPackageClass
"""
import os.path
from Library.Misc import Sdict
from Library.Misc import GetNonMetaDataFiles
from PomAdapter.InfPomAlignment import InfPomAlignment
from PomAdapter.DecPomAlignment import DecPomAlignment
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger.ToolError import OPTION_VALUE_INVALID
from Logger.ToolError import FatalError
from Logger.ToolError import EDK1_INF_ERROR
from Object.POM.CommonObject import IdentificationObject
from Object.POM.CommonObject import CommonHeaderObject
from Object.POM.CommonObject import MiscFileObject
from Common.MultipleWorkspace import MultipleWorkspace as mws

class DistributionPackageHeaderObject(IdentificationObject, CommonHeaderObject):

    def __init__(self):
        if False:
            return 10
        IdentificationObject.__init__(self)
        CommonHeaderObject.__init__(self)
        self.ReadOnly = ''
        self.RePackage = ''
        self.Vendor = ''
        self.Date = ''
        self.Signature = 'Md5Sum'
        self.XmlSpecification = ''

    def GetReadOnly(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ReadOnly

    def SetReadOnly(self, ReadOnly):
        if False:
            i = 10
            return i + 15
        self.ReadOnly = ReadOnly

    def GetRePackage(self):
        if False:
            print('Hello World!')
        return self.RePackage

    def SetRePackage(self, RePackage):
        if False:
            print('Hello World!')
        self.RePackage = RePackage

    def GetVendor(self):
        if False:
            i = 10
            return i + 15
        return self.Vendor

    def SetDate(self, Date):
        if False:
            while True:
                i = 10
        self.Date = Date

    def GetDate(self):
        if False:
            while True:
                i = 10
        return self.Date

    def SetSignature(self, Signature):
        if False:
            for i in range(10):
                print('nop')
        self.Signature = Signature

    def GetSignature(self):
        if False:
            while True:
                i = 10
        return self.Signature

    def SetXmlSpecification(self, XmlSpecification):
        if False:
            i = 10
            return i + 15
        self.XmlSpecification = XmlSpecification

    def GetXmlSpecification(self):
        if False:
            i = 10
            return i + 15
        return self.XmlSpecification

class DistributionPackageClass(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.Header = DistributionPackageHeaderObject()
        self.PackageSurfaceArea = Sdict()
        self.ModuleSurfaceArea = Sdict()
        self.Tools = MiscFileObject()
        self.MiscellaneousFiles = MiscFileObject()
        self.UserExtensions = []
        self.FileList = []

    def GetDistributionPackage(self, WorkspaceDir, PackageList, ModuleList):
        if False:
            print('Hello World!')
        Root = WorkspaceDir
        if PackageList:
            for PackageFile in PackageList:
                PackageFileFullPath = mws.join(Root, PackageFile)
                WorkspaceDir = mws.getWs(Root, PackageFile)
                DecObj = DecPomAlignment(PackageFileFullPath, WorkspaceDir, CheckMulDec=True)
                PackageObj = DecObj
                ModuleInfFileList = PackageObj.GetModuleFileList()
                for File in ModuleInfFileList:
                    WsRelPath = os.path.join(PackageObj.GetPackagePath(), File)
                    WsRelPath = os.path.normpath(WsRelPath)
                    if ModuleList and WsRelPath in ModuleList:
                        Logger.Error('UPT', OPTION_VALUE_INVALID, ST.ERR_NOT_STANDALONE_MODULE_ERROR % (WsRelPath, PackageFile))
                    Filename = os.path.normpath(os.path.join(PackageObj.GetRelaPath(), File))
                    os.path.splitext(Filename)
                    try:
                        ModuleObj = InfPomAlignment(Filename, WorkspaceDir, PackageObj.GetPackagePath())
                        ModuleDict = PackageObj.GetModuleDict()
                        ModuleDict[ModuleObj.GetGuid(), ModuleObj.GetVersion(), ModuleObj.GetName(), ModuleObj.GetCombinePath()] = ModuleObj
                        PackageObj.SetModuleDict(ModuleDict)
                    except FatalError as ErrCode:
                        if ErrCode.message == EDK1_INF_ERROR:
                            Logger.Warn('UPT', ST.WRN_EDK1_INF_FOUND % Filename)
                        else:
                            raise
                self.PackageSurfaceArea[PackageObj.GetGuid(), PackageObj.GetVersion(), PackageObj.GetCombinePath()] = PackageObj
        if ModuleList:
            for ModuleFile in ModuleList:
                ModuleFileFullPath = mws.join(Root, ModuleFile)
                WorkspaceDir = mws.getWs(Root, ModuleFile)
                try:
                    ModuleObj = InfPomAlignment(ModuleFileFullPath, WorkspaceDir)
                    ModuleKey = (ModuleObj.GetGuid(), ModuleObj.GetVersion(), ModuleObj.GetName(), ModuleObj.GetCombinePath())
                    self.ModuleSurfaceArea[ModuleKey] = ModuleObj
                except FatalError as ErrCode:
                    if ErrCode.message == EDK1_INF_ERROR:
                        Logger.Error('UPT', EDK1_INF_ERROR, ST.WRN_EDK1_INF_FOUND % ModuleFileFullPath, ExtraData=ST.ERR_NOT_SUPPORTED_SA_MODULE)
                    else:
                        raise
        WorkspaceDir = Root

    def GetDistributionFileList(self):
        if False:
            return 10
        MetaDataFileList = []
        SkipModulesUniList = []
        for (Guid, Version, Path) in self.PackageSurfaceArea:
            Package = self.PackageSurfaceArea[Guid, Version, Path]
            PackagePath = Package.GetPackagePath()
            FullPath = Package.GetFullPath()
            MetaDataFileList.append(Path)
            IncludePathList = Package.GetIncludePathList()
            for IncludePath in IncludePathList:
                SearchPath = os.path.normpath(os.path.join(os.path.dirname(FullPath), IncludePath))
                AddPath = os.path.normpath(os.path.join(PackagePath, IncludePath))
                self.FileList += GetNonMetaDataFiles(SearchPath, ['CVS', '.svn'], False, AddPath)
            for MiscFileObj in Package.GetMiscFileList():
                for FileObj in MiscFileObj.GetFileList():
                    MiscFileFullPath = os.path.normpath(os.path.join(PackagePath, FileObj.GetURI()))
                    if MiscFileFullPath not in self.FileList:
                        self.FileList.append(MiscFileFullPath)
            Module = None
            ModuleDict = Package.GetModuleDict()
            for (Guid, Version, Name, Path) in ModuleDict:
                Module = ModuleDict[Guid, Version, Name, Path]
                ModulePath = Module.GetModulePath()
                FullPath = Module.GetFullPath()
                PkgRelPath = os.path.normpath(os.path.join(PackagePath, ModulePath))
                MetaDataFileList.append(Path)
                SkipList = ['CVS', '.svn']
                NonMetaDataFileList = []
                if Module.UniFileClassObject:
                    for UniFile in Module.UniFileClassObject.IncFileList:
                        OriPath = os.path.normpath(os.path.dirname(FullPath))
                        UniFilePath = os.path.normpath(os.path.join(PkgRelPath, UniFile.Path[len(OriPath) + 1:]))
                        if UniFilePath not in SkipModulesUniList:
                            SkipModulesUniList.append(UniFilePath)
                    for IncludeFile in Module.UniFileClassObject.IncludePathList:
                        if IncludeFile not in SkipModulesUniList:
                            SkipModulesUniList.append(IncludeFile)
                NonMetaDataFileList = GetNonMetaDataFiles(os.path.dirname(FullPath), SkipList, False, PkgRelPath)
                for NonMetaDataFile in NonMetaDataFileList:
                    if NonMetaDataFile not in self.FileList:
                        self.FileList.append(NonMetaDataFile)
        for (Guid, Version, Name, Path) in self.ModuleSurfaceArea:
            Module = self.ModuleSurfaceArea[Guid, Version, Name, Path]
            ModulePath = Module.GetModulePath()
            FullPath = Module.GetFullPath()
            MetaDataFileList.append(Path)
            SkipList = ['CVS', '.svn']
            NonMetaDataFileList = []
            if Module.UniFileClassObject:
                for UniFile in Module.UniFileClassObject.IncFileList:
                    OriPath = os.path.normpath(os.path.dirname(FullPath))
                    UniFilePath = os.path.normpath(os.path.join(ModulePath, UniFile.Path[len(OriPath) + 1:]))
                    if UniFilePath not in SkipModulesUniList:
                        SkipModulesUniList.append(UniFilePath)
            NonMetaDataFileList = GetNonMetaDataFiles(os.path.dirname(FullPath), SkipList, False, ModulePath)
            for NonMetaDataFile in NonMetaDataFileList:
                if NonMetaDataFile not in self.FileList:
                    self.FileList.append(NonMetaDataFile)
        for SkipModuleUni in SkipModulesUniList:
            if SkipModuleUni in self.FileList:
                self.FileList.remove(SkipModuleUni)
        return (self.FileList, MetaDataFileList)