"""
Install a distribution package
"""
from Core.FileHook import __FileHookOpen__
import os.path
from os import chmod
from os import SEEK_SET
from os import SEEK_END
import stat
from hashlib import md5
import copy
from sys import stdin
from sys import platform
from shutil import rmtree
from shutil import copyfile
from traceback import format_exc
from platform import python_version
from Logger import StringTable as ST
from Logger.ToolError import UNKNOWN_ERROR
from Logger.ToolError import FILE_UNKNOWN_ERROR
from Logger.ToolError import OPTION_MISSING
from Logger.ToolError import UPT_ALREADY_INSTALLED_ERROR
from Logger.ToolError import FatalError
from Logger.ToolError import ABORT_ERROR
from Logger.ToolError import CODE_ERROR
from Logger.ToolError import FORMAT_INVALID
from Logger.ToolError import FILE_TYPE_MISMATCH
import Logger.Log as Logger
from Library.Misc import Sdict
from Library.Misc import ConvertPath
from Library.ParserValidate import IsValidInstallPath
from Xml.XmlParser import DistributionPackageXml
from GenMetaFile.GenDecFile import PackageToDec
from GenMetaFile.GenInfFile import ModuleToInf
from Core.PackageFile import PackageFile
from Core.PackageFile import FILE_NOT_FOUND
from Core.PackageFile import FILE_CHECKSUM_FAILURE
from Core.PackageFile import CreateDirectory
from Core.DependencyRules import DependencyRules
from Library import GlobalData

def InstallNewPackage(WorkspaceDir, Path, CustomPath=False):
    if False:
        print('Hello World!')
    if os.path.isabs(Path):
        Logger.Info(ST.MSG_RELATIVE_PATH_ONLY % Path)
    elif CustomPath:
        Logger.Info(ST.MSG_NEW_PKG_PATH)
    else:
        Path = ConvertPath(Path)
        Path = os.path.normpath(Path)
        FullPath = os.path.normpath(os.path.join(WorkspaceDir, Path))
        if os.path.exists(FullPath):
            Logger.Info(ST.ERR_DIR_ALREADY_EXIST % FullPath)
        else:
            return Path
    Input = stdin.readline()
    Input = Input.replace('\r', '').replace('\n', '')
    if Input == '':
        Logger.Error('InstallPkg', UNKNOWN_ERROR, ST.ERR_USER_INTERRUPT)
    Input = Input.replace('\r', '').replace('\n', '')
    return InstallNewPackage(WorkspaceDir, Input, False)

def InstallNewModule(WorkspaceDir, Path, PathList=None):
    if False:
        i = 10
        return i + 15
    if PathList is None:
        PathList = []
    Path = ConvertPath(Path)
    Path = os.path.normpath(Path)
    FullPath = os.path.normpath(os.path.join(WorkspaceDir, Path))
    if os.path.exists(FullPath) and FullPath not in PathList:
        Logger.Info(ST.ERR_DIR_ALREADY_EXIST % Path)
    elif Path == FullPath:
        Logger.Info(ST.MSG_RELATIVE_PATH_ONLY % FullPath)
    else:
        return Path
    Input = stdin.readline()
    Input = Input.replace('\r', '').replace('\n', '')
    if Input == '':
        Logger.Error('InstallPkg', UNKNOWN_ERROR, ST.ERR_USER_INTERRUPT)
    Input = Input.replace('\r', '').replace('\n', '')
    return InstallNewModule(WorkspaceDir, Input, PathList)

def InstallNewFile(WorkspaceDir, File):
    if False:
        while True:
            i = 10
    FullPath = os.path.normpath(os.path.join(WorkspaceDir, File))
    if os.path.exists(FullPath):
        Logger.Info(ST.ERR_FILE_ALREADY_EXIST % File)
        Input = stdin.readline()
        Input = Input.replace('\r', '').replace('\n', '')
        if Input == '':
            Logger.Error('InstallPkg', UNKNOWN_ERROR, ST.ERR_USER_INTERRUPT)
        Input = Input.replace('\r', '').replace('\n', '')
        return InstallNewFile(WorkspaceDir, Input)
    else:
        return File

def UnZipDp(WorkspaceDir, DpPkgFileName, Index=1):
    if False:
        return 10
    ContentZipFile = None
    Logger.Quiet(ST.MSG_UZIP_PARSE_XML)
    DistFile = PackageFile(DpPkgFileName)
    (DpDescFileName, ContentFileName) = GetDPFile(DistFile.GetZipFile())
    TempDir = os.path.normpath(os.path.join(WorkspaceDir, 'Conf/.tmp%s' % str(Index)))
    GlobalData.gUNPACK_DIR.append(TempDir)
    DistPkgFile = DistFile.UnpackFile(DpDescFileName, os.path.normpath(os.path.join(TempDir, DpDescFileName)))
    if not DistPkgFile:
        Logger.Error('InstallPkg', FILE_NOT_FOUND, ST.ERR_FILE_BROKEN % DpDescFileName)
    DistPkgObj = DistributionPackageXml()
    DistPkg = DistPkgObj.FromXml(DistPkgFile)
    if DistPkg.Header.RePackage == '':
        DistPkg.Header.RePackage = False
    if DistPkg.Header.ReadOnly == '':
        DistPkg.Header.ReadOnly = False
    ContentFile = DistFile.UnpackFile(ContentFileName, os.path.normpath(os.path.join(TempDir, ContentFileName)))
    if not ContentFile:
        Logger.Error('InstallPkg', FILE_NOT_FOUND, ST.ERR_FILE_BROKEN % ContentFileName)
    FileSize = os.path.getsize(ContentFile)
    if FileSize != 0:
        ContentZipFile = PackageFile(ContentFile)
    if DistPkg.Header.Signature != '':
        Md5Signature = md5(__FileHookOpen__(ContentFile, 'rb').read())
        if DistPkg.Header.Signature != Md5Signature.hexdigest():
            ContentZipFile.Close()
            Logger.Error('InstallPkg', FILE_CHECKSUM_FAILURE, ExtraData=ContentFile)
    return (DistPkg, ContentZipFile, DpPkgFileName, DistFile)

def GetPackageList(DistPkg, Dep, WorkspaceDir, Options, ContentZipFile, ModuleList, PackageList):
    if False:
        return 10
    NewDict = Sdict()
    for (Guid, Version, Path) in DistPkg.PackageSurfaceArea:
        PackagePath = Path
        Package = DistPkg.PackageSurfaceArea[Guid, Version, Path]
        Logger.Info(ST.MSG_INSTALL_PACKAGE % Package.GetName())
        if Options.UseGuidedPkgPath:
            GuidedPkgPath = '%s_%s_%s' % (Package.GetName(), Guid, Version)
            NewPackagePath = InstallNewPackage(WorkspaceDir, GuidedPkgPath, Options.CustomPath)
        else:
            NewPackagePath = InstallNewPackage(WorkspaceDir, PackagePath, Options.CustomPath)
        InstallPackageContent(PackagePath, NewPackagePath, Package, ContentZipFile, Dep, WorkspaceDir, ModuleList, DistPkg.Header.ReadOnly)
        PackageList.append(Package)
        NewDict[Guid, Version, Package.GetPackagePath()] = Package
    for Package in PackageList:
        FilePath = PackageToDec(Package, DistPkg.Header)
        Md5Signature = md5(__FileHookOpen__(str(FilePath), 'rb').read())
        Md5Sum = Md5Signature.hexdigest()
        if (FilePath, Md5Sum) not in Package.FileList:
            Package.FileList.append((FilePath, Md5Sum))
    return NewDict

def GetModuleList(DistPkg, Dep, WorkspaceDir, ContentZipFile, ModuleList):
    if False:
        for i in range(10):
            print('nop')
    ModulePathList = []
    Module = None
    NewDict = Sdict()
    for (Guid, Version, Name, Path) in DistPkg.ModuleSurfaceArea:
        ModulePath = Path
        Module = DistPkg.ModuleSurfaceArea[Guid, Version, Name, Path]
        Logger.Info(ST.MSG_INSTALL_MODULE % Module.GetName())
        if Dep.CheckModuleExists(Guid, Version, Name, Path):
            Logger.Quiet(ST.WRN_MODULE_EXISTED % Path)
        ModuleFullPath = os.path.normpath(os.path.join(WorkspaceDir, ModulePath))
        if ModuleFullPath not in ModulePathList:
            NewModulePath = InstallNewModule(WorkspaceDir, ModulePath, ModulePathList)
            NewModuleFullPath = os.path.normpath(os.path.join(WorkspaceDir, NewModulePath))
            ModulePathList.append(NewModuleFullPath)
        else:
            NewModulePath = ModulePath
        InstallModuleContent(ModulePath, NewModulePath, '', Module, ContentZipFile, WorkspaceDir, ModuleList, None, DistPkg.Header.ReadOnly)
        Module.SetModulePath(Module.GetModulePath().replace(Path, NewModulePath, 1))
        NewDict[Guid, Version, Name, Module.GetModulePath()] = Module
    for (Module, Package) in ModuleList:
        CheckCNameInModuleRedefined(Module, DistPkg)
        FilePath = ModuleToInf(Module, Package, DistPkg.Header)
        Md5Signature = md5(__FileHookOpen__(str(FilePath), 'rb').read())
        Md5Sum = Md5Signature.hexdigest()
        if Package:
            if (FilePath, Md5Sum) not in Package.FileList:
                Package.FileList.append((FilePath, Md5Sum))
        elif (FilePath, Md5Sum) not in Module.FileList:
            Module.FileList.append((FilePath, Md5Sum))
        for (FilePath, Md5Sum) in Module.FileList:
            if str(FilePath).endswith('.uni') and Package and ((FilePath, Md5Sum) not in Package.FileList):
                Package.FileList.append((FilePath, Md5Sum))
    return NewDict

def GetDepProtocolPpiGuidPcdNames(DePackageObjList):
    if False:
        i = 10
        return i + 15
    DependentProtocolCNames = []
    DependentPpiCNames = []
    DependentGuidCNames = []
    DependentPcdNames = []
    for PackageObj in DePackageObjList:
        ProtocolCNames = []
        for Protocol in PackageObj.GetProtocolList():
            if Protocol.GetCName() not in ProtocolCNames:
                ProtocolCNames.append(Protocol.GetCName())
        DependentProtocolCNames.append(ProtocolCNames)
        PpiCNames = []
        for Ppi in PackageObj.GetPpiList():
            if Ppi.GetCName() not in PpiCNames:
                PpiCNames.append(Ppi.GetCName())
        DependentPpiCNames.append(PpiCNames)
        GuidCNames = []
        for Guid in PackageObj.GetGuidList():
            if Guid.GetCName() not in GuidCNames:
                GuidCNames.append(Guid.GetCName())
        DependentGuidCNames.append(GuidCNames)
        PcdNames = []
        for Pcd in PackageObj.GetPcdList():
            PcdName = '.'.join([Pcd.GetTokenSpaceGuidCName(), Pcd.GetCName()])
            if PcdName not in PcdNames:
                PcdNames.append(PcdName)
        DependentPcdNames.append(PcdNames)
    return (DependentProtocolCNames, DependentPpiCNames, DependentGuidCNames, DependentPcdNames)

def CheckProtoclCNameRedefined(Module, DependentProtocolCNames):
    if False:
        print('Hello World!')
    for ProtocolInModule in Module.GetProtocolList():
        IsCNameDefined = False
        for PackageProtocolCNames in DependentProtocolCNames:
            if ProtocolInModule.GetCName() in PackageProtocolCNames:
                if IsCNameDefined:
                    Logger.Error('\nUPT', FORMAT_INVALID, File=Module.GetFullPath(), ExtraData=ST.ERR_INF_PARSER_ITEM_DUPLICATE_IN_DEC % ProtocolInModule.GetCName())
                else:
                    IsCNameDefined = True

def CheckPpiCNameRedefined(Module, DependentPpiCNames):
    if False:
        print('Hello World!')
    for PpiInModule in Module.GetPpiList():
        IsCNameDefined = False
        for PackagePpiCNames in DependentPpiCNames:
            if PpiInModule.GetCName() in PackagePpiCNames:
                if IsCNameDefined:
                    Logger.Error('\nUPT', FORMAT_INVALID, File=Module.GetFullPath(), ExtraData=ST.ERR_INF_PARSER_ITEM_DUPLICATE_IN_DEC % PpiInModule.GetCName())
                else:
                    IsCNameDefined = True

def CheckGuidCNameRedefined(Module, DependentGuidCNames):
    if False:
        while True:
            i = 10
    for GuidInModule in Module.GetGuidList():
        IsCNameDefined = False
        for PackageGuidCNames in DependentGuidCNames:
            if GuidInModule.GetCName() in PackageGuidCNames:
                if IsCNameDefined:
                    Logger.Error('\nUPT', FORMAT_INVALID, File=Module.GetFullPath(), ExtraData=ST.ERR_INF_PARSER_ITEM_DUPLICATE_IN_DEC % GuidInModule.GetCName())
                else:
                    IsCNameDefined = True

def CheckPcdNameRedefined(Module, DependentPcdNames):
    if False:
        i = 10
        return i + 15
    PcdObjs = []
    if not Module.GetBinaryFileList():
        PcdObjs += Module.GetPcdList()
    else:
        Binary = Module.GetBinaryFileList()[0]
        for AsBuild in Binary.GetAsBuiltList():
            PcdObjs += AsBuild.GetPatchPcdList() + AsBuild.GetPcdExList()
    for PcdObj in PcdObjs:
        PcdName = '.'.join([PcdObj.GetTokenSpaceGuidCName(), PcdObj.GetCName()])
        IsPcdNameDefined = False
        for PcdNames in DependentPcdNames:
            if PcdName in PcdNames:
                if IsPcdNameDefined:
                    Logger.Error('\nUPT', FORMAT_INVALID, File=Module.GetFullPath(), ExtraData=ST.ERR_INF_PARSER_ITEM_DUPLICATE_IN_DEC % PcdName)
                else:
                    IsPcdNameDefined = True

def CheckCNameInModuleRedefined(Module, DistPkg):
    if False:
        print('Hello World!')
    DePackageObjList = []
    for Obj in Module.GetPackageDependencyList():
        Guid = Obj.GetGuid()
        Version = Obj.GetVersion()
        for Key in DistPkg.PackageSurfaceArea:
            if Key[0] == Guid and Key[1] == Version:
                if DistPkg.PackageSurfaceArea[Key] not in DePackageObjList:
                    DePackageObjList.append(DistPkg.PackageSurfaceArea[Key])
    (DependentProtocolCNames, DependentPpiCNames, DependentGuidCNames, DependentPcdNames) = GetDepProtocolPpiGuidPcdNames(DePackageObjList)
    CheckProtoclCNameRedefined(Module, DependentProtocolCNames)
    CheckPpiCNameRedefined(Module, DependentPpiCNames)
    CheckGuidCNameRedefined(Module, DependentGuidCNames)
    CheckPcdNameRedefined(Module, DependentPcdNames)

def GenToolMisc(DistPkg, WorkspaceDir, ContentZipFile):
    if False:
        i = 10
        return i + 15
    ToolObject = DistPkg.Tools
    MiscObject = DistPkg.MiscellaneousFiles
    DistPkg.FileList = []
    FileList = []
    ToolFileNum = 0
    FileNum = 0
    RootDir = WorkspaceDir
    if ToolObject:
        FileList += ToolObject.GetFileList()
        ToolFileNum = len(ToolObject.GetFileList())
        if 'EDK_TOOLS_PATH' in os.environ:
            RootDir = os.environ['EDK_TOOLS_PATH']
    if MiscObject:
        FileList += MiscObject.GetFileList()
    for FileObject in FileList:
        FileNum += 1
        if FileNum > ToolFileNum:
            RootDir = WorkspaceDir
        File = ConvertPath(FileObject.GetURI())
        ToFile = os.path.normpath(os.path.join(RootDir, File))
        if os.path.exists(ToFile):
            Logger.Info(ST.WRN_FILE_EXISTED % ToFile)
            Logger.Info(ST.MSG_NEW_FILE_NAME)
            Input = stdin.readline()
            Input = Input.replace('\r', '').replace('\n', '')
            OrigPath = os.path.split(ToFile)[0]
            ToFile = os.path.normpath(os.path.join(OrigPath, Input))
        FromFile = os.path.join(FileObject.GetURI())
        Md5Sum = InstallFile(ContentZipFile, FromFile, ToFile, DistPkg.Header.ReadOnly, FileObject.GetExecutable())
        DistPkg.FileList.append((ToFile, Md5Sum))

def Main(Options=None):
    if False:
        i = 10
        return i + 15
    try:
        DataBase = GlobalData.gDB
        WorkspaceDir = GlobalData.gWORKSPACE
        if not Options.PackageFile:
            Logger.Error('InstallPkg', OPTION_MISSING, ExtraData=ST.ERR_SPECIFY_PACKAGE)
        DistInfoList = []
        DistPkgList = []
        Index = 1
        for ToBeInstalledDist in Options.PackageFile:
            DistInfoList.append(UnZipDp(WorkspaceDir, ToBeInstalledDist, Index))
            DistPkgList.append(DistInfoList[-1][0])
            Index += 1
            GlobalData.gTO_BE_INSTALLED_DIST_LIST.append(DistInfoList[-1][0])
        Dep = DependencyRules(DataBase, DistPkgList)
        for ToBeInstalledDist in DistInfoList:
            CheckInstallDpx(Dep, ToBeInstalledDist[0], ToBeInstalledDist[2])
            InstallDp(ToBeInstalledDist[0], ToBeInstalledDist[2], ToBeInstalledDist[1], Options, Dep, WorkspaceDir, DataBase)
        ReturnCode = 0
    except FatalError as XExcept:
        ReturnCode = XExcept.args[0]
        if Logger.GetLevel() <= Logger.DEBUG_9:
            Logger.Quiet(ST.MSG_PYTHON_ON % (python_version(), platform) + format_exc())
    except KeyboardInterrupt:
        ReturnCode = ABORT_ERROR
        if Logger.GetLevel() <= Logger.DEBUG_9:
            Logger.Quiet(ST.MSG_PYTHON_ON % (python_version(), platform) + format_exc())
    except:
        ReturnCode = CODE_ERROR
        Logger.Error('\nInstallPkg', CODE_ERROR, ST.ERR_UNKNOWN_FATAL_INSTALL_ERR % Options.PackageFile, ExtraData=ST.MSG_SEARCH_FOR_HELP % ST.MSG_EDKII_MAIL_ADDR, RaiseError=False)
        Logger.Quiet(ST.MSG_PYTHON_ON % (python_version(), platform) + format_exc())
    finally:
        Logger.Quiet(ST.MSG_REMOVE_TEMP_FILE_STARTED)
        for ToBeInstalledDist in DistInfoList:
            if ToBeInstalledDist[3]:
                ToBeInstalledDist[3].Close()
            if ToBeInstalledDist[1]:
                ToBeInstalledDist[1].Close()
        for TempDir in GlobalData.gUNPACK_DIR:
            rmtree(TempDir)
        GlobalData.gUNPACK_DIR = []
        Logger.Quiet(ST.MSG_REMOVE_TEMP_FILE_DONE)
    if ReturnCode == 0:
        Logger.Quiet(ST.MSG_FINISH)
    return ReturnCode

def BackupDist(DpPkgFileName, Guid, Version, WorkspaceDir):
    if False:
        while True:
            i = 10
    DistFileName = os.path.split(DpPkgFileName)[1]
    DestDir = os.path.normpath(os.path.join(WorkspaceDir, GlobalData.gUPT_DIR))
    CreateDirectory(DestDir)
    DestFile = os.path.normpath(os.path.join(DestDir, DistFileName))
    if os.path.exists(DestFile):
        (FileName, Ext) = os.path.splitext(DistFileName)
        NewFileName = FileName + '_' + Guid + '_' + Version + Ext
        DestFile = os.path.normpath(os.path.join(DestDir, NewFileName))
        if os.path.exists(DestFile):
            Logger.Info(ST.MSG_NEW_FILE_NAME_FOR_DIST)
            Input = stdin.readline()
            Input = Input.replace('\r', '').replace('\n', '')
            DestFile = os.path.normpath(os.path.join(DestDir, Input))
    copyfile(DpPkgFileName, DestFile)
    NewDpPkgFileName = DestFile[DestFile.find(DestDir) + len(DestDir) + 1:]
    return NewDpPkgFileName

def CheckInstallDpx(Dep, DistPkg, DistPkgFileName):
    if False:
        for i in range(10):
            print('nop')
    if Dep.CheckDpExists(DistPkg.Header.GetGuid(), DistPkg.Header.GetVersion()):
        Logger.Error('InstallPkg', UPT_ALREADY_INSTALLED_ERROR, ST.WRN_DIST_PKG_INSTALLED % os.path.basename(DistPkgFileName))
    if not Dep.CheckInstallDpDepexSatisfied(DistPkg):
        Logger.Error('InstallPkg', UNKNOWN_ERROR, ST.ERR_PACKAGE_NOT_MATCH_DEPENDENCY, ExtraData=DistPkg.Header.Name)

def InstallModuleContent(FromPath, NewPath, ModulePath, Module, ContentZipFile, WorkspaceDir, ModuleList, Package=None, ReadOnly=False):
    if False:
        return 10
    if NewPath.startswith('\\') or NewPath.startswith('/'):
        NewPath = NewPath[1:]
    if not IsValidInstallPath(NewPath):
        Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % NewPath)
    NewModuleFullPath = os.path.normpath(os.path.join(WorkspaceDir, NewPath, ConvertPath(ModulePath)))
    Module.SetFullPath(os.path.normpath(os.path.join(NewModuleFullPath, ConvertPath(Module.GetName()) + '.inf')))
    Module.FileList = []
    for MiscFile in Module.GetMiscFileList():
        if not MiscFile:
            continue
        for Item in MiscFile.GetFileList():
            File = Item.GetURI()
            if File.startswith('\\') or File.startswith('/'):
                File = File[1:]
            if not IsValidInstallPath(File):
                Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % File)
            FromFile = os.path.join(FromPath, ModulePath, File)
            Executable = Item.GetExecutable()
            ToFile = os.path.normpath(os.path.join(NewModuleFullPath, ConvertPath(File)))
            Md5Sum = InstallFile(ContentZipFile, FromFile, ToFile, ReadOnly, Executable)
            if Package and (ToFile, Md5Sum) not in Package.FileList:
                Package.FileList.append((ToFile, Md5Sum))
            elif Package:
                continue
            elif (ToFile, Md5Sum) not in Module.FileList:
                Module.FileList.append((ToFile, Md5Sum))
    for Item in Module.GetSourceFileList():
        File = Item.GetSourceFile()
        if File.startswith('\\') or File.startswith('/'):
            File = File[1:]
        if not IsValidInstallPath(File):
            Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % File)
        FromFile = os.path.join(FromPath, ModulePath, File)
        ToFile = os.path.normpath(os.path.join(NewModuleFullPath, ConvertPath(File)))
        Md5Sum = InstallFile(ContentZipFile, FromFile, ToFile, ReadOnly)
        if Package and (ToFile, Md5Sum) not in Package.FileList:
            Package.FileList.append((ToFile, Md5Sum))
        elif Package:
            continue
        elif (ToFile, Md5Sum) not in Module.FileList:
            Module.FileList.append((ToFile, Md5Sum))
    for Item in Module.GetBinaryFileList():
        FileNameList = Item.GetFileNameList()
        for FileName in FileNameList:
            File = FileName.GetFilename()
            if File.startswith('\\') or File.startswith('/'):
                File = File[1:]
            if not IsValidInstallPath(File):
                Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % File)
            FromFile = os.path.join(FromPath, ModulePath, File)
            ToFile = os.path.normpath(os.path.join(NewModuleFullPath, ConvertPath(File)))
            Md5Sum = InstallFile(ContentZipFile, FromFile, ToFile, ReadOnly)
            if Package and (ToFile, Md5Sum) not in Package.FileList:
                Package.FileList.append((ToFile, Md5Sum))
            elif Package:
                continue
            elif (ToFile, Md5Sum) not in Module.FileList:
                Module.FileList.append((ToFile, Md5Sum))
    InstallModuleContentZipFile(ContentZipFile, FromPath, ModulePath, WorkspaceDir, NewPath, Module, Package, ReadOnly, ModuleList)

def InstallModuleContentZipFile(ContentZipFile, FromPath, ModulePath, WorkspaceDir, NewPath, Module, Package, ReadOnly, ModuleList):
    if False:
        while True:
            i = 10
    if ContentZipFile:
        for FileName in ContentZipFile.GetZipFile().namelist():
            FileName = os.path.normpath(FileName)
            CheckPath = os.path.normpath(os.path.join(FromPath, ModulePath))
            if FileUnderPath(FileName, CheckPath):
                if FileName.startswith('\\') or FileName.startswith('/'):
                    FileName = FileName[1:]
                if not IsValidInstallPath(FileName):
                    Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % FileName)
                FromFile = FileName
                ToFile = os.path.normpath(os.path.join(WorkspaceDir, ConvertPath(FileName.replace(FromPath, NewPath, 1))))
                CheckList = copy.copy(Module.FileList)
                if Package:
                    CheckList += Package.FileList
                for Item in CheckList:
                    if Item[0] == ToFile:
                        break
                else:
                    Md5Sum = InstallFile(ContentZipFile, FromFile, ToFile, ReadOnly)
                    if Package and (ToFile, Md5Sum) not in Package.FileList:
                        Package.FileList.append((ToFile, Md5Sum))
                    elif Package:
                        continue
                    elif (ToFile, Md5Sum) not in Module.FileList:
                        Module.FileList.append((ToFile, Md5Sum))
    ModuleList.append((Module, Package))

def FileUnderPath(FileName, CheckPath):
    if False:
        print('Hello World!')
    FileName = FileName.replace('\\', '/')
    FileName = os.path.normpath(FileName)
    CheckPath = CheckPath.replace('\\', '/')
    CheckPath = os.path.normpath(CheckPath)
    if FileName.startswith(CheckPath):
        RemainingPath = os.path.normpath(FileName.replace(CheckPath, '', 1))
        while RemainingPath.startswith('\\') or RemainingPath.startswith('/'):
            RemainingPath = RemainingPath[1:]
        if FileName == os.path.normpath(os.path.join(CheckPath, RemainingPath)):
            return True
    return False

def InstallFile(ContentZipFile, FromFile, ToFile, ReadOnly, Executable=False):
    if False:
        print('Hello World!')
    if os.path.exists(os.path.normpath(ToFile)):
        pass
    else:
        if not ContentZipFile or not ContentZipFile.UnpackFile(FromFile, ToFile):
            Logger.Error('UPT', FILE_NOT_FOUND, ST.ERR_INSTALL_FILE_FROM_EMPTY_CONTENT % FromFile)
        if ReadOnly:
            if not Executable:
                chmod(ToFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            else:
                chmod(ToFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        elif Executable:
            chmod(ToFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        else:
            chmod(ToFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    Md5Signature = md5(__FileHookOpen__(str(ToFile), 'rb').read())
    Md5Sum = Md5Signature.hexdigest()
    return Md5Sum

def InstallPackageContent(FromPath, ToPath, Package, ContentZipFile, Dep, WorkspaceDir, ModuleList, ReadOnly=False):
    if False:
        print('Hello World!')
    if Dep:
        pass
    Package.FileList = []
    if ToPath.startswith('\\') or ToPath.startswith('/'):
        ToPath = ToPath[1:]
    if not IsValidInstallPath(ToPath):
        Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % ToPath)
    if FromPath.startswith('\\') or FromPath.startswith('/'):
        FromPath = FromPath[1:]
    if not IsValidInstallPath(FromPath):
        Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % FromPath)
    PackageFullPath = os.path.normpath(os.path.join(WorkspaceDir, ToPath))
    for MiscFile in Package.GetMiscFileList():
        for Item in MiscFile.GetFileList():
            FileName = Item.GetURI()
            if FileName.startswith('\\') or FileName.startswith('/'):
                FileName = FileName[1:]
            if not IsValidInstallPath(FileName):
                Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % FileName)
            FromFile = os.path.join(FromPath, FileName)
            Executable = Item.GetExecutable()
            ToFile = os.path.join(PackageFullPath, ConvertPath(FileName))
            Md5Sum = InstallFile(ContentZipFile, FromFile, ToFile, ReadOnly, Executable)
            if (ToFile, Md5Sum) not in Package.FileList:
                Package.FileList.append((ToFile, Md5Sum))
    PackageIncludeArchList = []
    for Item in Package.GetPackageIncludeFileList():
        FileName = Item.GetFilePath()
        if FileName.startswith('\\') or FileName.startswith('/'):
            FileName = FileName[1:]
        if not IsValidInstallPath(FileName):
            Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % FileName)
        FromFile = os.path.join(FromPath, FileName)
        ToFile = os.path.normpath(os.path.join(PackageFullPath, ConvertPath(FileName)))
        RetFile = ContentZipFile.UnpackFile(FromFile, ToFile)
        if RetFile == '':
            PackageIncludeArchList.append([Item.GetFilePath(), Item.GetSupArchList()])
            CreateDirectory(ToFile)
            continue
        if ReadOnly:
            chmod(ToFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        else:
            chmod(ToFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        Md5Signature = md5(__FileHookOpen__(str(ToFile), 'rb').read())
        Md5Sum = Md5Signature.hexdigest()
        if (ToFile, Md5Sum) not in Package.FileList:
            Package.FileList.append((ToFile, Md5Sum))
    Package.SetIncludeArchList(PackageIncludeArchList)
    for Item in Package.GetStandardIncludeFileList():
        FileName = Item.GetFilePath()
        if FileName.startswith('\\') or FileName.startswith('/'):
            FileName = FileName[1:]
        if not IsValidInstallPath(FileName):
            Logger.Error('UPT', FORMAT_INVALID, ST.ERR_FILE_NAME_INVALIDE % FileName)
        FromFile = os.path.join(FromPath, FileName)
        ToFile = os.path.normpath(os.path.join(PackageFullPath, ConvertPath(FileName)))
        Md5Sum = InstallFile(ContentZipFile, FromFile, ToFile, ReadOnly)
        if (ToFile, Md5Sum) not in Package.FileList:
            Package.FileList.append((ToFile, Md5Sum))
    Package.SetPackagePath(Package.GetPackagePath().replace(FromPath, ToPath, 1))
    Package.SetFullPath(os.path.normpath(os.path.join(PackageFullPath, ConvertPath(Package.GetName()) + '.dec')))
    Module = None
    ModuleDict = Package.GetModuleDict()
    for (ModuleGuid, ModuleVersion, ModuleName, ModulePath) in ModuleDict:
        Module = ModuleDict[ModuleGuid, ModuleVersion, ModuleName, ModulePath]
        InstallModuleContent(FromPath, ToPath, ModulePath, Module, ContentZipFile, WorkspaceDir, ModuleList, Package, ReadOnly)

def GetDPFile(ZipFile):
    if False:
        return 10
    ContentFile = ''
    DescFile = ''
    for FileName in ZipFile.namelist():
        if FileName.endswith('.content'):
            if not ContentFile:
                ContentFile = FileName
                continue
        elif FileName.endswith('.pkg'):
            if not DescFile:
                DescFile = FileName
                continue
        else:
            continue
        Logger.Error('PackagingTool', FILE_TYPE_MISMATCH, ExtraData=ST.ERR_DIST_FILE_TOOMANY)
    if not DescFile or not ContentFile:
        Logger.Error('PackagingTool', FILE_UNKNOWN_ERROR, ExtraData=ST.ERR_DIST_FILE_TOOFEW)
    return (DescFile, ContentFile)

def InstallDp(DistPkg, DpPkgFileName, ContentZipFile, Options, Dep, WorkspaceDir, DataBase):
    if False:
        i = 10
        return i + 15
    PackageList = []
    ModuleList = []
    DistPkg.PackageSurfaceArea = GetPackageList(DistPkg, Dep, WorkspaceDir, Options, ContentZipFile, ModuleList, PackageList)
    DistPkg.ModuleSurfaceArea = GetModuleList(DistPkg, Dep, WorkspaceDir, ContentZipFile, ModuleList)
    GenToolMisc(DistPkg, WorkspaceDir, ContentZipFile)
    DistFileName = os.path.split(DpPkgFileName)[1]
    NewDpPkgFileName = BackupDist(DpPkgFileName, DistPkg.Header.GetGuid(), DistPkg.Header.GetVersion(), WorkspaceDir)
    Logger.Quiet(ST.MSG_UPDATE_PACKAGE_DATABASE)
    DataBase.AddDPObject(DistPkg, NewDpPkgFileName, DistFileName, DistPkg.Header.RePackage)