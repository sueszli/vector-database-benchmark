"""
Test Install multiple distribution package
"""
from Library import GlobalData
import Logger.Log as Logger
from Logger import StringTable as ST
import Logger.ToolError as TE
from Core.DependencyRules import DependencyRules
from InstallPkg import UnZipDp
import shutil
from traceback import format_exc
from platform import python_version
from sys import platform

def Main(Options=None):
    if False:
        for i in range(10):
            print('nop')
    (ContentZipFile, DistFile) = (None, None)
    ReturnCode = 0
    try:
        DataBase = GlobalData.gDB
        WorkspaceDir = GlobalData.gWORKSPACE
        if not Options.DistFiles:
            Logger.Error('TestInstallPkg', TE.OPTION_MISSING, ExtraData=ST.ERR_SPECIFY_PACKAGE)
        DistPkgList = []
        for DistFile in Options.DistFiles:
            (DistPkg, ContentZipFile, __, DistFile) = UnZipDp(WorkspaceDir, DistFile)
            DistPkgList.append(DistPkg)
        Dep = DependencyRules(DataBase)
        Result = True
        DpObj = None
        try:
            (Result, DpObj) = Dep.CheckTestInstallPdDepexSatisfied(DistPkgList)
        except:
            Result = False
        if Result:
            Logger.Quiet(ST.MSG_TEST_INSTALL_PASS)
        else:
            Logger.Quiet(ST.MSG_TEST_INSTALL_FAIL)
    except TE.FatalError as XExcept:
        ReturnCode = XExcept.args[0]
        if Logger.GetLevel() <= Logger.DEBUG_9:
            Logger.Quiet(ST.MSG_PYTHON_ON % (python_version(), platform) + format_exc())
    except Exception as x:
        ReturnCode = TE.CODE_ERROR
        Logger.Error('\nTestInstallPkg', TE.CODE_ERROR, ST.ERR_UNKNOWN_FATAL_INSTALL_ERR % Options.DistFiles, ExtraData=ST.MSG_SEARCH_FOR_HELP % ST.MSG_EDKII_MAIL_ADDR, RaiseError=False)
        Logger.Quiet(ST.MSG_PYTHON_ON % (python_version(), platform) + format_exc())
    finally:
        Logger.Quiet(ST.MSG_REMOVE_TEMP_FILE_STARTED)
        if DistFile:
            DistFile.Close()
        if ContentZipFile:
            ContentZipFile.Close()
        for TempDir in GlobalData.gUNPACK_DIR:
            shutil.rmtree(TempDir)
        GlobalData.gUNPACK_DIR = []
        Logger.Quiet(ST.MSG_REMOVE_TEMP_FILE_DONE)
    if ReturnCode == 0:
        Logger.Quiet(ST.MSG_FINISH)
    return ReturnCode