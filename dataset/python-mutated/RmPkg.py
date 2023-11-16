"""
RmPkg
"""
import os.path
from stat import S_IWUSR
from traceback import format_exc
from platform import python_version
from hashlib import md5
from sys import stdin
from sys import platform
from Core.DependencyRules import DependencyRules
from Library import GlobalData
from Logger import StringTable as ST
import Logger.Log as Logger
from Logger.ToolError import OPTION_MISSING
from Logger.ToolError import UNKNOWN_ERROR
from Logger.ToolError import ABORT_ERROR
from Logger.ToolError import CODE_ERROR
from Logger.ToolError import FatalError

def CheckDpDepex(Dep, Guid, Version, WorkspaceDir):
    if False:
        return 10
    (Removable, DependModuleList) = Dep.CheckDpDepexForRemove(Guid, Version)
    if not Removable:
        Logger.Info(ST.MSG_CONFIRM_REMOVE)
        Logger.Info(ST.MSG_USER_DELETE_OP)
        Input = stdin.readline()
        Input = Input.replace('\r', '').replace('\n', '')
        if Input.upper() != 'Y':
            Logger.Error('RmPkg', UNKNOWN_ERROR, ST.ERR_USER_INTERRUPT)
            return 1
        else:
            Logger.Info(ST.MSG_INVALID_MODULE_INTRODUCED)
            LogFilePath = os.path.normpath(os.path.join(WorkspaceDir, GlobalData.gINVALID_MODULE_FILE))
            Logger.Info(ST.MSG_CHECK_LOG_FILE % LogFilePath)
            try:
                LogFile = open(LogFilePath, 'w')
                try:
                    for ModulePath in DependModuleList:
                        LogFile.write('%s\n' % ModulePath)
                        Logger.Info(ModulePath)
                except IOError:
                    Logger.Warn('\nRmPkg', ST.ERR_FILE_WRITE_FAILURE, File=LogFilePath)
            except IOError:
                Logger.Warn('\nRmPkg', ST.ERR_FILE_OPEN_FAILURE, File=LogFilePath)
            finally:
                LogFile.close()

def RemovePath(Path):
    if False:
        for i in range(10):
            print('nop')
    Logger.Info(ST.MSG_REMOVE_FILE % Path)
    if not os.access(Path, os.W_OK):
        os.chmod(Path, S_IWUSR)
    os.remove(Path)
    try:
        os.removedirs(os.path.split(Path)[0])
    except OSError:
        pass

def GetCurrentFileList(DataBase, Guid, Version, WorkspaceDir):
    if False:
        print('Hello World!')
    NewFileList = []
    for Dir in DataBase.GetDpInstallDirList(Guid, Version):
        RootDir = os.path.normpath(os.path.join(WorkspaceDir, Dir))
        for (Root, Dirs, Files) in os.walk(RootDir):
            Logger.Debug(0, Dirs)
            for File in Files:
                FilePath = os.path.join(Root, File)
                if FilePath not in NewFileList:
                    NewFileList.append(FilePath)
    return NewFileList

def Main(Options=None):
    if False:
        for i in range(10):
            print('nop')
    try:
        DataBase = GlobalData.gDB
        if not Options.DistributionFile:
            Logger.Error('RmPkg', OPTION_MISSING, ExtraData=ST.ERR_SPECIFY_PACKAGE)
        WorkspaceDir = GlobalData.gWORKSPACE
        Dep = DependencyRules(DataBase)
        (StoredDistFile, Guid, Version) = GetInstalledDpInfo(Options.DistributionFile, Dep, DataBase, WorkspaceDir)
        CheckDpDepex(Dep, Guid, Version, WorkspaceDir)
        RemoveDist(Guid, Version, StoredDistFile, DataBase, WorkspaceDir, Options.Yes)
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
    except:
        Logger.Error('\nRmPkg', CODE_ERROR, ST.ERR_UNKNOWN_FATAL_REMOVING_ERR, ExtraData=ST.MSG_SEARCH_FOR_HELP % ST.MSG_EDKII_MAIL_ADDR, RaiseError=False)
        Logger.Quiet(ST.MSG_PYTHON_ON % (python_version(), platform) + format_exc())
        ReturnCode = CODE_ERROR
    return ReturnCode

def GetInstalledDpInfo(DistributionFile, Dep, DataBase, WorkspaceDir):
    if False:
        for i in range(10):
            print('nop')
    (Guid, Version, NewDpFileName) = DataBase.GetDpByName(os.path.split(DistributionFile)[1])
    if not Guid:
        Logger.Error('RmPkg', UNKNOWN_ERROR, ST.ERR_PACKAGE_NOT_INSTALLED % DistributionFile)
    if not Dep.CheckDpExists(Guid, Version):
        Logger.Error('RmPkg', UNKNOWN_ERROR, ST.ERR_DISTRIBUTION_NOT_INSTALLED)
    StoredDistFile = os.path.normpath(os.path.join(WorkspaceDir, GlobalData.gUPT_DIR, NewDpFileName))
    if not os.path.isfile(StoredDistFile):
        Logger.Warn('RmPkg', ST.WRN_DIST_NOT_FOUND % StoredDistFile)
        StoredDistFile = None
    return (StoredDistFile, Guid, Version)

def RemoveDist(Guid, Version, StoredDistFile, DataBase, WorkspaceDir, ForceRemove):
    if False:
        i = 10
        return i + 15
    NewFileList = GetCurrentFileList(DataBase, Guid, Version, WorkspaceDir)
    MissingFileList = []
    for (Path, Md5Sum) in DataBase.GetDpFileList(Guid, Version):
        if os.path.isfile(Path):
            if Path in NewFileList:
                NewFileList.remove(Path)
            if not ForceRemove:
                Md5Signature = md5(open(str(Path), 'rb').read())
                if Md5Sum != Md5Signature.hexdigest():
                    Logger.Info(ST.MSG_CONFIRM_REMOVE2 % Path)
                    Input = stdin.readline()
                    Input = Input.replace('\r', '').replace('\n', '')
                    if Input.upper() != 'Y':
                        continue
            RemovePath(Path)
        else:
            MissingFileList.append(Path)
    for Path in NewFileList:
        if os.path.isfile(Path):
            if not ForceRemove and (not os.path.split(Path)[1].startswith('.')):
                Logger.Info(ST.MSG_CONFIRM_REMOVE3 % Path)
                Input = stdin.readline()
                Input = Input.replace('\r', '').replace('\n', '')
                if Input.upper() != 'Y':
                    continue
            RemovePath(Path)
    if StoredDistFile is not None:
        os.remove(StoredDistFile)
    Logger.Quiet(ST.MSG_UPDATE_PACKAGE_DATABASE)
    DataBase.RemoveDpObj(Guid, Version)