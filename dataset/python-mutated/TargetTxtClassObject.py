from __future__ import print_function
from __future__ import absolute_import
import Common.GlobalData as GlobalData
import Common.LongFilePathOs as os
from . import EdkLogger
from . import DataType
from .BuildToolError import *
from Common.LongFilePathSupport import OpenLongFilePath as open
from Common.MultipleWorkspace import MultipleWorkspace as mws
gDefaultTargetTxtFile = 'target.txt'

class TargetTxtClassObject(object):

    def __init__(self, Filename=None):
        if False:
            for i in range(10):
                print('nop')
        self.TargetTxtDictionary = {DataType.TAB_TAT_DEFINES_ACTIVE_PLATFORM: '', DataType.TAB_TAT_DEFINES_ACTIVE_MODULE: '', DataType.TAB_TAT_DEFINES_TOOL_CHAIN_CONF: '', DataType.TAB_TAT_DEFINES_MAX_CONCURRENT_THREAD_NUMBER: '', DataType.TAB_TAT_DEFINES_TARGET: [], DataType.TAB_TAT_DEFINES_TOOL_CHAIN_TAG: [], DataType.TAB_TAT_DEFINES_TARGET_ARCH: [], DataType.TAB_TAT_DEFINES_BUILD_RULE_CONF: ''}
        self.ConfDirectoryPath = ''
        if Filename is not None:
            self.LoadTargetTxtFile(Filename)

    def LoadTargetTxtFile(self, Filename):
        if False:
            for i in range(10):
                print('nop')
        if os.path.exists(Filename) and os.path.isfile(Filename):
            return self.ConvertTextFileToDict(Filename, '#', '=')
        else:
            EdkLogger.error('Target.txt Parser', FILE_NOT_FOUND, ExtraData=Filename)
            return 1

    def ConvertTextFileToDict(self, FileName, CommentCharacter, KeySplitCharacter):
        if False:
            print('Hello World!')
        F = None
        try:
            F = open(FileName, 'r')
            self.ConfDirectoryPath = os.path.dirname(FileName)
        except:
            EdkLogger.error('build', FILE_OPEN_FAILURE, ExtraData=FileName)
            if F is not None:
                F.close()
        for Line in F:
            Line = Line.strip()
            if Line.startswith(CommentCharacter) or Line == '':
                continue
            LineList = Line.split(KeySplitCharacter, 1)
            Key = LineList[0].strip()
            if len(LineList) == 2:
                Value = LineList[1].strip()
            else:
                Value = ''
            if Key in [DataType.TAB_TAT_DEFINES_ACTIVE_PLATFORM, DataType.TAB_TAT_DEFINES_TOOL_CHAIN_CONF, DataType.TAB_TAT_DEFINES_ACTIVE_MODULE, DataType.TAB_TAT_DEFINES_BUILD_RULE_CONF]:
                self.TargetTxtDictionary[Key] = Value.replace('\\', '/')
                if Key == DataType.TAB_TAT_DEFINES_TOOL_CHAIN_CONF and self.TargetTxtDictionary[Key]:
                    if self.TargetTxtDictionary[Key].startswith('Conf/'):
                        Tools_Def = os.path.join(self.ConfDirectoryPath, self.TargetTxtDictionary[Key].strip())
                        if not os.path.exists(Tools_Def) or not os.path.isfile(Tools_Def):
                            Tools_Def = os.path.join(self.ConfDirectoryPath, self.TargetTxtDictionary[Key].replace('Conf/', '', 1).strip())
                    else:
                        Tools_Def = os.path.join(self.ConfDirectoryPath, self.TargetTxtDictionary[Key].strip())
                    self.TargetTxtDictionary[Key] = Tools_Def
                if Key == DataType.TAB_TAT_DEFINES_BUILD_RULE_CONF and self.TargetTxtDictionary[Key]:
                    if self.TargetTxtDictionary[Key].startswith('Conf/'):
                        Build_Rule = os.path.join(self.ConfDirectoryPath, self.TargetTxtDictionary[Key].strip())
                        if not os.path.exists(Build_Rule) or not os.path.isfile(Build_Rule):
                            Build_Rule = os.path.join(self.ConfDirectoryPath, self.TargetTxtDictionary[Key].replace('Conf/', '', 1).strip())
                    else:
                        Build_Rule = os.path.join(self.ConfDirectoryPath, self.TargetTxtDictionary[Key].strip())
                    self.TargetTxtDictionary[Key] = Build_Rule
            elif Key in [DataType.TAB_TAT_DEFINES_TARGET, DataType.TAB_TAT_DEFINES_TARGET_ARCH, DataType.TAB_TAT_DEFINES_TOOL_CHAIN_TAG]:
                self.TargetTxtDictionary[Key] = Value.split()
            elif Key == DataType.TAB_TAT_DEFINES_MAX_CONCURRENT_THREAD_NUMBER:
                try:
                    V = int(Value, 0)
                except:
                    EdkLogger.error('build', FORMAT_INVALID, 'Invalid number of [%s]: %s.' % (Key, Value), File=FileName)
                self.TargetTxtDictionary[Key] = Value
        F.close()
        return 0

class TargetTxtDict:

    def __new__(cls, *args, **kw):
        if False:
            return 10
        if not hasattr(cls, '_instance'):
            orig = super(TargetTxtDict, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self):
        if False:
            print('Hello World!')
        if not hasattr(self, 'Target'):
            self.TxtTarget = None

    @property
    def Target(self):
        if False:
            i = 10
            return i + 15
        if not self.TxtTarget:
            self._GetTarget()
        return self.TxtTarget

    def _GetTarget(self):
        if False:
            while True:
                i = 10
        Target = TargetTxtClassObject()
        ConfDirectory = GlobalData.gCmdConfDir
        if ConfDirectory:
            ConfDirectoryPath = os.path.normpath(ConfDirectory)
            if not os.path.isabs(ConfDirectoryPath):
                ConfDirectoryPath = mws.join(os.environ['WORKSPACE'], ConfDirectoryPath)
        elif 'CONF_PATH' in os.environ:
            ConfDirectoryPath = os.path.normcase(os.path.normpath(os.environ['CONF_PATH']))
        else:
            ConfDirectoryPath = mws.join(os.environ['WORKSPACE'], 'Conf')
        GlobalData.gConfDirectory = ConfDirectoryPath
        targettxt = os.path.normpath(os.path.join(ConfDirectoryPath, gDefaultTargetTxtFile))
        if os.path.exists(targettxt):
            Target.LoadTargetTxtFile(targettxt)
        self.TxtTarget = Target
if __name__ == '__main__':
    pass
    Target = TargetTxtDict(os.getenv('WORKSPACE'))
    print(Target.TargetTxtDictionary[DataType.TAB_TAT_DEFINES_MAX_CONCURRENT_THREAD_NUMBER])
    print(Target.TargetTxtDictionary[DataType.TAB_TAT_DEFINES_TARGET])
    print(Target.TargetTxtDictionary)