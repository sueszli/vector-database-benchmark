"""
MkPkg
"""
from os import remove
from os import getcwd
from os import chdir
import os.path
from sys import stdin
from sys import platform
from traceback import format_exc
from platform import python_version
from hashlib import md5
from time import strftime
from time import localtime
from uuid import uuid4
from Logger import StringTable as ST
from Logger.ToolError import OPTION_UNKNOWN_ERROR
from Logger.ToolError import OPTION_VALUE_INVALID
from Logger.ToolError import ABORT_ERROR
from Logger.ToolError import UPT_REPKG_ERROR
from Logger.ToolError import CODE_ERROR
from Logger.ToolError import FatalError
from Logger.ToolError import FILE_NOT_FOUND
import Logger.Log as Logger
from Xml.XmlParser import DistributionPackageXml
from Xml.IniToXml import IniToXml
from Library import GlobalData
from Library.ParserValidate import IsValidPath
from Core.DistributionPackageClass import DistributionPackageClass
from Core.PackageFile import PackageFile
from Common.MultipleWorkspace import MultipleWorkspace as mws

def CheckForExistingDp(Path):
    if False:
        print('Hello World!')
    if os.path.exists(Path):
        Logger.Info(ST.MSG_DISTRIBUTION_PACKAGE_FILE_EXISTS % Path)
        Input = stdin.readline()
        Input = Input.replace('\r', '').replace('\n', '')
        if Input.upper() != 'Y':
            Logger.Error('\nMkPkg', ABORT_ERROR, ST.ERR_USER_ABORT, RaiseError=True)

def Main(Options=None):
    if False:
        while True:
            i = 10
    if Options is None:
        Logger.Error('\nMkPkg', OPTION_UNKNOWN_ERROR, ST.ERR_OPTION_NOT_FOUND)
    try:
        DataBase = GlobalData.gDB
        ContentFileClosed = True
        WorkspaceDir = GlobalData.gWORKSPACE
        if not Options.PackFileToCreate:
            Logger.Error('\nMkPkg', OPTION_UNKNOWN_ERROR, ST.ERR_OPTION_NOT_FOUND)
        CheckForExistingDp(Options.PackFileToCreate)
        CheckFileList('.DEC', Options.PackageFileList, ST.ERR_INVALID_PACKAGE_NAME, ST.ERR_INVALID_PACKAGE_PATH)
        CheckFileList('.INF', Options.ModuleFileList, ST.ERR_INVALID_MODULE_NAME, ST.ERR_INVALID_MODULE_PATH)
        RePkgDict = DataBase.GetRePkgDict()
        ContentFile = PackageFile(GlobalData.gCONTENT_FILE, 'w')
        ContentFileClosed = False
        if Options.PackageInformationDataFile:
            XmlFile = IniToXml(Options.PackageInformationDataFile)
            DistPkg = DistributionPackageXml().FromXml(XmlFile)
            remove(XmlFile)
            Cwd = getcwd()
            chdir(WorkspaceDir)
            ToolObject = DistPkg.Tools
            MiscObject = DistPkg.MiscellaneousFiles
            FileList = []
            if ToolObject:
                FileList += ToolObject.GetFileList()
            if MiscObject:
                FileList += MiscObject.GetFileList()
            for FileObject in FileList:
                FromFile = os.path.normpath(FileObject.GetURI()).encode('utf_8')
                FileFullPath = mws.join(WorkspaceDir, FromFile)
                if FileFullPath in RePkgDict:
                    (DpGuid, DpVersion, DpName, Repackage) = RePkgDict[FileFullPath]
                    if not Repackage:
                        Logger.Error('\nMkPkg', UPT_REPKG_ERROR, ST.ERR_UPT_REPKG_ERROR, ExtraData=ST.MSG_REPKG_CONFLICT % (FileFullPath, DpGuid, DpVersion, DpName))
                    else:
                        DistPkg.Header.RePackage = True
                ContentFile.PackFile(FromFile)
            chdir(Cwd)
        else:
            DistPkg = DistributionPackageClass()
            DistPkg.Header.Name = 'Distribution Package'
            DistPkg.Header.Guid = str(uuid4())
            DistPkg.Header.Version = '1.0'
        DistPkg.GetDistributionPackage(WorkspaceDir, Options.PackageFileList, Options.ModuleFileList)
        (FileList, MetaDataFileList) = DistPkg.GetDistributionFileList()
        for File in FileList + MetaDataFileList:
            FileFullPath = os.path.normpath(os.path.join(WorkspaceDir, File))
            if FileFullPath in RePkgDict:
                (DpGuid, DpVersion, DpName, Repackage) = RePkgDict[FileFullPath]
                if not Repackage:
                    Logger.Error('\nMkPkg', UPT_REPKG_ERROR, ST.ERR_UPT_REPKG_ERROR, ExtraData=ST.MSG_REPKG_CONFLICT % (FileFullPath, DpName, DpGuid, DpVersion))
                else:
                    DistPkg.Header.RePackage = True
        Cwd = getcwd()
        chdir(WorkspaceDir)
        ContentFile.PackFiles(FileList)
        chdir(Cwd)
        Logger.Verbose(ST.MSG_COMPRESS_DISTRIBUTION_PKG)
        ContentFile.Close()
        ContentFileClosed = True
        DistPkg.Header.Signature = md5(open(str(ContentFile), 'rb').read()).hexdigest()
        DistPkg.Header.Date = str(strftime('%Y-%m-%dT%H:%M:%S', localtime()))
        DistPkgFile = PackageFile(Options.PackFileToCreate, 'w')
        DistPkgFile.PackFile(str(ContentFile))
        DistPkgXml = DistributionPackageXml()
        DistPkgFile.PackData(DistPkgXml.ToXml(DistPkg), GlobalData.gDESC_FILE)
        DistPkgFile.Close()
        Logger.Quiet(ST.MSG_FINISH)
        ReturnCode = 0
    except FatalError as XExcept:
        ReturnCode = XExcept.args[0]
        if Logger.GetLevel() <= Logger.DEBUG_9:
            Logger.Quiet(ST.MSG_PYTHON_ON % (python_version(), platform) + format_exc())
    except KeyboardInterrupt:
        ReturnCode = ABORT_ERROR
        if Logger.GetLevel() <= Logger.DEBUG_9:
            Logger.Quiet(ST.MSG_PYTHON_ON % (python_version(), platform) + format_exc())
    except OSError:
        pass
    except:
        Logger.Error('\nMkPkg', CODE_ERROR, ST.ERR_UNKNOWN_FATAL_CREATING_ERR % Options.PackFileToCreate, ExtraData=ST.MSG_SEARCH_FOR_HELP % ST.MSG_EDKII_MAIL_ADDR, RaiseError=False)
        Logger.Quiet(ST.MSG_PYTHON_ON % (python_version(), platform) + format_exc())
        ReturnCode = CODE_ERROR
    finally:
        if os.path.exists(GlobalData.gCONTENT_FILE):
            if not ContentFileClosed:
                ContentFile.Close()
            os.remove(GlobalData.gCONTENT_FILE)
    return ReturnCode

def CheckFileList(QualifiedExt, FileList, ErrorStringExt, ErrorStringFullPath):
    if False:
        while True:
            i = 10
    if not FileList:
        return
    WorkspaceDir = GlobalData.gWORKSPACE
    WorkspaceDir = os.path.normpath(WorkspaceDir)
    for Item in FileList:
        Ext = os.path.splitext(Item)[1]
        if Ext.upper() != QualifiedExt.upper():
            Logger.Error('\nMkPkg', OPTION_VALUE_INVALID, ErrorStringExt % Item)
        Item = os.path.normpath(Item)
        Path = mws.join(WorkspaceDir, Item)
        if not os.path.exists(Path):
            Logger.Error('\nMkPkg', FILE_NOT_FOUND, ST.ERR_NOT_FOUND % Item)
        elif Item == Path:
            Logger.Error('\nMkPkg', OPTION_VALUE_INVALID, ErrorStringFullPath % Item)
        elif not IsValidPath(Item, WorkspaceDir):
            Logger.Error('\nMkPkg', OPTION_VALUE_INVALID, ErrorStringExt % Item)
        if not os.path.split(Item)[0]:
            Logger.Error('\nMkPkg', OPTION_VALUE_INVALID, ST.ERR_INVALID_METAFILE_PATH % Item)