"""
Misc
"""
import os.path
from os import access
from os import F_OK
from os import makedirs
from os import getcwd
from os import chdir
from os import listdir
from os import remove
from os import rmdir
from os import linesep
from os import walk
from os import environ
import re
from collections import OrderedDict as Sdict
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger import ToolError
from Library import GlobalData
from Library.DataType import SUP_MODULE_LIST
from Library.DataType import END_OF_LINE
from Library.DataType import TAB_SPLIT
from Library.DataType import TAB_LANGUAGE_EN_US
from Library.DataType import TAB_LANGUAGE_EN
from Library.DataType import TAB_LANGUAGE_EN_X
from Library.DataType import TAB_UNI_FILE_SUFFIXS
from Library.StringUtils import GetSplitValueList
from Library.ParserValidate import IsValidHexVersion
from Library.ParserValidate import IsValidPath
from Object.POM.CommonObject import TextObject
from Core.FileHook import __FileHookOpen__
from Common.MultipleWorkspace import MultipleWorkspace as mws

def GuidStringToGuidStructureString(Guid):
    if False:
        while True:
            i = 10
    GuidList = Guid.split('-')
    Result = '{'
    for Index in range(0, 3, 1):
        Result = Result + '0x' + GuidList[Index] + ', '
    Result = Result + '{0x' + GuidList[3][0:2] + ', 0x' + GuidList[3][2:4]
    for Index in range(0, 12, 2):
        Result = Result + ', 0x' + GuidList[4][Index:Index + 2]
    Result += '}}'
    return Result

def CheckGuidRegFormat(GuidValue):
    if False:
        i = 10
        return i + 15
    RegFormatGuidPattern = re.compile('^\\s*([0-9a-fA-F]){8}-([0-9a-fA-F]){4}-([0-9a-fA-F]){4}-([0-9a-fA-F]){4}-([0-9a-fA-F]){12}\\s*$')
    if RegFormatGuidPattern.match(GuidValue):
        return True
    else:
        return False

def GuidStructureStringToGuidString(GuidValue):
    if False:
        while True:
            i = 10
    GuidValueString = GuidValue.lower().replace('{', '').replace('}', '').replace(' ', '').replace(';', '')
    GuidValueList = GuidValueString.split(',')
    if len(GuidValueList) != 11:
        return ''
    try:
        return '%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x' % (int(GuidValueList[0], 16), int(GuidValueList[1], 16), int(GuidValueList[2], 16), int(GuidValueList[3], 16), int(GuidValueList[4], 16), int(GuidValueList[5], 16), int(GuidValueList[6], 16), int(GuidValueList[7], 16), int(GuidValueList[8], 16), int(GuidValueList[9], 16), int(GuidValueList[10], 16))
    except BaseException:
        return ''

def CreateDirectory(Directory):
    if False:
        print('Hello World!')
    if Directory is None or Directory.strip() == '':
        return True
    try:
        if not access(Directory, F_OK):
            makedirs(Directory)
    except BaseException:
        return False
    return True

def RemoveDirectory(Directory, Recursively=False):
    if False:
        while True:
            i = 10
    if Directory is None or Directory.strip() == '' or (not os.path.exists(Directory)):
        return
    if Recursively:
        CurrentDirectory = getcwd()
        chdir(Directory)
        for File in listdir('.'):
            if os.path.isdir(File):
                RemoveDirectory(File, Recursively)
            else:
                remove(File)
        chdir(CurrentDirectory)
    rmdir(Directory)

def SaveFileOnChange(File, Content, IsBinaryFile=True):
    if False:
        i = 10
        return i + 15
    if os.path.exists(File):
        if IsBinaryFile:
            try:
                if Content == __FileHookOpen__(File, 'rb').read():
                    return False
            except BaseException:
                Logger.Error(None, ToolError.FILE_OPEN_FAILURE, ExtraData=File)
        else:
            try:
                if Content == __FileHookOpen__(File, 'r').read():
                    return False
            except BaseException:
                Logger.Error(None, ToolError.FILE_OPEN_FAILURE, ExtraData=File)
    CreateDirectory(os.path.dirname(File))
    if IsBinaryFile:
        try:
            FileFd = __FileHookOpen__(File, 'wb')
            FileFd.write(Content)
            FileFd.close()
        except BaseException:
            Logger.Error(None, ToolError.FILE_CREATE_FAILURE, ExtraData=File)
    else:
        try:
            FileFd = __FileHookOpen__(File, 'w')
            FileFd.write(Content)
            FileFd.close()
        except BaseException:
            Logger.Error(None, ToolError.FILE_CREATE_FAILURE, ExtraData=File)
    return True

def GetFiles(Root, SkipList=None, FullPath=True):
    if False:
        while True:
            i = 10
    OriPath = os.path.normpath(Root)
    FileList = []
    for (Root, Dirs, Files) in walk(Root):
        if SkipList:
            for Item in SkipList:
                if Item in Dirs:
                    Dirs.remove(Item)
                if Item in Files:
                    Files.remove(Item)
        for Dir in Dirs:
            if Dir.startswith('.'):
                Dirs.remove(Dir)
        for File in Files:
            if File.startswith('.'):
                continue
            File = os.path.normpath(os.path.join(Root, File))
            if not FullPath:
                File = File[len(OriPath) + 1:]
            FileList.append(File)
    return FileList

def GetNonMetaDataFiles(Root, SkipList, FullPath, PrefixPath):
    if False:
        while True:
            i = 10
    FileList = GetFiles(Root, SkipList, FullPath)
    NewFileList = []
    for File in FileList:
        ExtName = os.path.splitext(File)[1]
        if ExtName.lower() not in ['.dec', '.inf', '.dsc', '.fdf']:
            NewFileList.append(os.path.normpath(os.path.join(PrefixPath, File)))
    return NewFileList

def ValidFile(File, Ext=None):
    if False:
        return 10
    File = File.replace('\\', '/')
    if Ext is not None:
        FileExt = os.path.splitext(File)[1]
        if FileExt.lower() != Ext.lower():
            return False
    if not os.path.exists(File):
        return False
    return True

def RealPath(File, Dir='', OverrideDir=''):
    if False:
        for i in range(10):
            print('nop')
    NewFile = os.path.normpath(os.path.join(Dir, File))
    NewFile = GlobalData.gALL_FILES[NewFile]
    if not NewFile and OverrideDir:
        NewFile = os.path.normpath(os.path.join(OverrideDir, File))
        NewFile = GlobalData.gALL_FILES[NewFile]
    return NewFile

def RealPath2(File, Dir='', OverrideDir=''):
    if False:
        return 10
    if OverrideDir:
        NewFile = GlobalData.gALL_FILES[os.path.normpath(os.path.join(OverrideDir, File))]
        if NewFile:
            if OverrideDir[-1] == os.path.sep:
                return (NewFile[len(OverrideDir):], NewFile[0:len(OverrideDir)])
            else:
                return (NewFile[len(OverrideDir) + 1:], NewFile[0:len(OverrideDir)])
    NewFile = GlobalData.gALL_FILES[os.path.normpath(os.path.join(Dir, File))]
    if NewFile:
        if Dir:
            if Dir[-1] == os.path.sep:
                return (NewFile[len(Dir):], NewFile[0:len(Dir)])
            else:
                return (NewFile[len(Dir) + 1:], NewFile[0:len(Dir)])
        else:
            return (NewFile, '')
    return (None, None)

def CommonPath(PathList):
    if False:
        while True:
            i = 10
    Path1 = min(PathList).split(os.path.sep)
    Path2 = max(PathList).split(os.path.sep)
    for Index in range(min(len(Path1), len(Path2))):
        if Path1[Index] != Path2[Index]:
            return os.path.sep.join(Path1[:Index])
    return os.path.sep.join(Path1)

class PathClass(object):

    def __init__(self, File='', Root='', AlterRoot='', Type='', IsBinary=False, Arch='COMMON', ToolChainFamily='', Target='', TagName='', ToolCode=''):
        if False:
            i = 10
            return i + 15
        self.Arch = Arch
        self.File = str(File)
        if os.path.isabs(self.File):
            self.Root = ''
            self.AlterRoot = ''
        else:
            self.Root = str(Root)
            self.AlterRoot = str(AlterRoot)
        if self.Root:
            self.Path = os.path.normpath(os.path.join(self.Root, self.File))
            self.Root = os.path.normpath(CommonPath([self.Root, self.Path]))
            if self.Root[-1] == ':':
                self.Root += os.path.sep
            if self.Root[-1] == os.path.sep:
                self.File = self.Path[len(self.Root):]
            else:
                self.File = self.Path[len(self.Root) + 1:]
        else:
            self.Path = os.path.normpath(self.File)
        (self.SubDir, self.Name) = os.path.split(self.File)
        (self.BaseName, self.Ext) = os.path.splitext(self.Name)
        if self.Root:
            if self.SubDir:
                self.Dir = os.path.join(self.Root, self.SubDir)
            else:
                self.Dir = self.Root
        else:
            self.Dir = self.SubDir
        if IsBinary:
            self.Type = Type
        else:
            self.Type = self.Ext.lower()
        self.IsBinary = IsBinary
        self.Target = Target
        self.TagName = TagName
        self.ToolCode = ToolCode
        self.ToolChainFamily = ToolChainFamily
        self._Key = None

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Path

    def __eq__(self, Other):
        if False:
            i = 10
            return i + 15
        if isinstance(Other, type(self)):
            return self.Path == Other.Path
        else:
            return self.Path == str(Other)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.Path)

    def _GetFileKey(self):
        if False:
            for i in range(10):
                print('nop')
        if self._Key is None:
            self._Key = self.Path.upper()
        return self._Key

    def Validate(self, Type='', CaseSensitive=True):
        if False:
            for i in range(10):
                print('nop')
        if GlobalData.gCASE_INSENSITIVE:
            CaseSensitive = False
        if Type and Type.lower() != self.Type:
            return (ToolError.FILE_TYPE_MISMATCH, '%s (expect %s but got %s)' % (self.File, Type, self.Type))
        (RealFile, RealRoot) = RealPath2(self.File, self.Root, self.AlterRoot)
        if not RealRoot and (not RealFile):
            RealFile = self.File
            if self.AlterRoot:
                RealFile = os.path.join(self.AlterRoot, self.File)
            elif self.Root:
                RealFile = os.path.join(self.Root, self.File)
            return (ToolError.FILE_NOT_FOUND, os.path.join(self.AlterRoot, RealFile))
        ErrorCode = 0
        ErrorInfo = ''
        if RealRoot != self.Root or RealFile != self.File:
            if CaseSensitive and (RealFile != self.File or (RealRoot != self.Root and RealRoot != self.AlterRoot)):
                ErrorCode = ToolError.FILE_CASE_MISMATCH
                ErrorInfo = self.File + '\n\t' + RealFile + ' [in file system]'
            (self.SubDir, self.Name) = os.path.split(RealFile)
            (self.BaseName, self.Ext) = os.path.splitext(self.Name)
            if self.SubDir:
                self.Dir = os.path.join(RealRoot, self.SubDir)
            else:
                self.Dir = RealRoot
            self.File = RealFile
            self.Root = RealRoot
            self.Path = os.path.join(RealRoot, RealFile)
        return (ErrorCode, ErrorInfo)
    Key = property(_GetFileKey)

def GetWorkspace():
    if False:
        print('Hello World!')
    if 'WORKSPACE' in environ:
        WorkspaceDir = os.path.normpath(environ['WORKSPACE'])
        if not os.path.exists(WorkspaceDir):
            Logger.Error('UPT', ToolError.UPT_ENVIRON_MISSING_ERROR, ST.ERR_WORKSPACE_NOTEXIST, ExtraData='%s' % WorkspaceDir)
    else:
        WorkspaceDir = os.getcwd()
    if WorkspaceDir[-1] == ':':
        WorkspaceDir += os.sep
    PackagesPath = os.environ.get('PACKAGES_PATH')
    mws.setWs(WorkspaceDir, PackagesPath)
    return (WorkspaceDir, mws.PACKAGES_PATH)

def GetRelativePath(Fullpath, Workspace):
    if False:
        for i in range(10):
            print('nop')
    RelativePath = ''
    if Workspace.endswith(os.sep):
        RelativePath = Fullpath[Fullpath.upper().find(Workspace.upper()) + len(Workspace):]
    else:
        RelativePath = Fullpath[Fullpath.upper().find(Workspace.upper()) + len(Workspace) + 1:]
    return RelativePath

def IsAllModuleList(ModuleList):
    if False:
        print('Hello World!')
    NewModuleList = [Module.upper() for Module in ModuleList]
    for Module in SUP_MODULE_LIST:
        if Module not in NewModuleList:
            return False
    else:
        return True

class MergeCommentDict(dict):

    def __setitem__(self, Key, CommentVal):
        if False:
            i = 10
            return i + 15
        (GenericComment, TailComment) = CommentVal
        if Key in self:
            (OrigVal1, OrigVal2) = dict.__getitem__(self, Key)
            Statement = Key[0]
            dict.__setitem__(self, Key, (OrigVal1 + GenericComment, OrigVal2 + len(Statement) * ' ' + TailComment))
        else:
            dict.__setitem__(self, Key, (GenericComment, TailComment))

    def __getitem__(self, Key):
        if False:
            return 10
        return dict.__getitem__(self, Key)

def GenDummyHelpTextObj():
    if False:
        for i in range(10):
            print('nop')
    HelpTxt = TextObject()
    HelpTxt.SetLang(TAB_LANGUAGE_EN_US)
    HelpTxt.SetString(' ')
    return HelpTxt

def ConvertVersionToDecimal(StringIn):
    if False:
        i = 10
        return i + 15
    if IsValidHexVersion(StringIn):
        Value = int(StringIn, 16)
        Major = Value >> 16
        Minor = Value & 65535
        MinorStr = str(Minor)
        if len(MinorStr) == 1:
            MinorStr = '0' + MinorStr
        return str(Major) + '.' + MinorStr
    elif StringIn.find(TAB_SPLIT) != -1:
        return StringIn
    elif StringIn:
        return StringIn + '.0'
    else:
        return StringIn

def GetHelpStringByRemoveHashKey(String):
    if False:
        i = 10
        return i + 15
    ReturnString = ''
    PattenRemoveHashKey = re.compile('^[#+\\s]+', re.DOTALL)
    String = String.strip()
    if String == '':
        return String
    LineList = GetSplitValueList(String, END_OF_LINE)
    for Line in LineList:
        ValueList = PattenRemoveHashKey.split(Line)
        if len(ValueList) == 1:
            ReturnString += ValueList[0] + END_OF_LINE
        else:
            ReturnString += ValueList[1] + END_OF_LINE
    if ReturnString.endswith('\n') and (not ReturnString.endswith('\n\n')) and (ReturnString != '\n'):
        ReturnString = ReturnString[:-1]
    return ReturnString

def ConvPathFromAbsToRel(Path, Root):
    if False:
        for i in range(10):
            print('nop')
    Path = os.path.normpath(Path)
    Root = os.path.normpath(Root)
    FullPath = os.path.normpath(os.path.join(Root, Path))
    if os.path.isabs(Path):
        return FullPath[FullPath.find(Root) + len(Root) + 1:]
    else:
        return Path

def ConvertPath(Path):
    if False:
        print('Hello World!')
    RetPath = ''
    for Char in Path.strip():
        if Char.isalnum() or Char in '.-_/':
            RetPath = RetPath + Char
        elif Char == '\\':
            RetPath = RetPath + '/'
        else:
            RetPath = RetPath + '_'
    return RetPath

def ConvertSpec(SpecStr):
    if False:
        for i in range(10):
            print('nop')
    RetStr = ''
    for Char in SpecStr:
        if Char.isalnum() or Char == '_':
            RetStr = RetStr + Char
        else:
            RetStr = RetStr + '_'
    return RetStr

def IsEqualList(ListA, ListB):
    if False:
        while True:
            i = 10
    if ListA == ListB:
        return True
    for ItemA in ListA:
        if not ItemA in ListB:
            return False
    for ItemB in ListB:
        if not ItemB in ListA:
            return False
    return True

def ConvertArchList(ArchList):
    if False:
        i = 10
        return i + 15
    NewArchList = []
    if not ArchList:
        return NewArchList
    if isinstance(ArchList, list):
        for Arch in ArchList:
            Arch = Arch.upper()
            NewArchList.append(Arch)
    elif isinstance(ArchList, str):
        ArchList = ArchList.upper()
        NewArchList.append(ArchList)
    return NewArchList

def ProcessLineExtender(LineList):
    if False:
        return 10
    NewList = []
    Count = 0
    while Count < len(LineList):
        if LineList[Count].strip().endswith('\\') and Count + 1 < len(LineList):
            NewList.append(LineList[Count].strip()[:-2] + LineList[Count + 1])
            Count = Count + 1
        else:
            NewList.append(LineList[Count])
        Count = Count + 1
    return NewList

def ProcessEdkComment(LineList):
    if False:
        i = 10
        return i + 15
    FindEdkBlockComment = False
    Count = 0
    StartPos = -1
    EndPos = -1
    FirstPos = -1
    while Count < len(LineList):
        Line = LineList[Count].strip()
        if Line.startswith('/*'):
            StartPos = Count
            while Count < len(LineList):
                Line = LineList[Count].strip()
                if Line.endswith('*/'):
                    if Count == StartPos and Line.strip() == '/*/':
                        Count = Count + 1
                        continue
                    EndPos = Count
                    FindEdkBlockComment = True
                    break
                Count = Count + 1
            if FindEdkBlockComment:
                if FirstPos == -1:
                    FirstPos = StartPos
                for Index in range(StartPos, EndPos + 1):
                    LineList[Index] = ''
                FindEdkBlockComment = False
        elif Line.find('//') != -1 and (not Line.startswith('#')):
            LineList[Count] = Line.replace('//', '#')
            if FirstPos == -1:
                FirstPos = Count
        Count = Count + 1
    return (LineList, FirstPos)

def GetLibInstanceInfo(String, WorkSpace, LineNo):
    if False:
        while True:
            i = 10
    FileGuidString = ''
    VerString = ''
    OriginalString = String
    String = String.strip()
    if not String:
        return (None, None)
    String = GetHelpStringByRemoveHashKey(String)
    String = String.strip()
    FullFileName = os.path.normpath(os.path.realpath(os.path.join(WorkSpace, String)))
    if not ValidFile(FullFileName):
        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_FILELIST_EXIST % String, File=GlobalData.gINF_MODULE_NAME, Line=LineNo, ExtraData=OriginalString)
    if IsValidPath(String, WorkSpace):
        IsValidFileFlag = True
    else:
        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID % String, File=GlobalData.gINF_MODULE_NAME, Line=LineNo, ExtraData=OriginalString)
        return False
    if IsValidFileFlag:
        FileLinesList = []
        try:
            FInputfile = open(FullFileName, 'r')
            try:
                FileLinesList = FInputfile.readlines()
            except BaseException:
                Logger.Error('InfParser', ToolError.FILE_READ_FAILURE, ST.ERR_FILE_OPEN_FAILURE, File=FullFileName)
            finally:
                FInputfile.close()
        except BaseException:
            Logger.Error('InfParser', ToolError.FILE_READ_FAILURE, ST.ERR_FILE_OPEN_FAILURE, File=FullFileName)
        ReFileGuidPattern = re.compile('^\\s*FILE_GUID\\s*=.*$')
        ReVerStringPattern = re.compile('^\\s*VERSION_STRING\\s*=.*$')
        FileLinesList = ProcessLineExtender(FileLinesList)
        for Line in FileLinesList:
            if ReFileGuidPattern.match(Line):
                FileGuidString = Line
            if ReVerStringPattern.match(Line):
                VerString = Line
        if FileGuidString:
            FileGuidString = GetSplitValueList(FileGuidString, '=', 1)[1]
        if VerString:
            VerString = GetSplitValueList(VerString, '=', 1)[1]
        return (FileGuidString, VerString)

def GetLocalValue(ValueList, UseFirstValue=False):
    if False:
        i = 10
        return i + 15
    Value1 = ''
    Value2 = ''
    Value3 = ''
    Value4 = ''
    Value5 = ''
    for (Key, Value) in ValueList:
        if Key == TAB_LANGUAGE_EN_X:
            if UseFirstValue:
                if not Value1:
                    Value1 = Value
            else:
                Value1 = Value
        if Key == TAB_LANGUAGE_EN_US:
            if UseFirstValue:
                if not Value2:
                    Value2 = Value
            else:
                Value2 = Value
        if Key == TAB_LANGUAGE_EN:
            if UseFirstValue:
                if not Value3:
                    Value3 = Value
            else:
                Value3 = Value
        if Key.startswith(TAB_LANGUAGE_EN):
            if UseFirstValue:
                if not Value4:
                    Value4 = Value
            else:
                Value4 = Value
        if Key == '':
            if UseFirstValue:
                if not Value5:
                    Value5 = Value
            else:
                Value5 = Value
    if Value1:
        return Value1
    if Value2:
        return Value2
    if Value3:
        return Value3
    if Value4:
        return Value4
    if Value5:
        return Value5
    return ''

def GetCharIndexOutStr(CommentCharacter, Line):
    if False:
        print('Hello World!')
    Line = Line.strip()
    InString = False
    for Index in range(0, len(Line)):
        if Line[Index] == '"':
            InString = not InString
        elif Line[Index] == CommentCharacter and InString:
            pass
        elif Line[Index] == CommentCharacter and Index + 1 < len(Line) and (Line[Index + 1] == CommentCharacter) and (not InString):
            return Index
    return -1

def ValidateUNIFilePath(Path):
    if False:
        while True:
            i = 10
    Suffix = Path[Path.rfind(TAB_SPLIT):]
    if Suffix not in TAB_UNI_FILE_SUFFIXS:
        Logger.Error('Unicode File Parser', ToolError.FORMAT_INVALID, Message=ST.ERR_UNI_FILE_SUFFIX_WRONG, ExtraData=Path)
    if TAB_SPLIT + TAB_SPLIT in Path:
        Logger.Error('Unicode File Parser', ToolError.FORMAT_INVALID, Message=ST.ERR_UNI_FILE_NAME_INVALID, ExtraData=Path)
    Pattern = '[a-zA-Z0-9_][a-zA-Z0-9_\\-\\.]*'
    FileName = Path.replace(Suffix, '')
    InvalidCh = re.sub(Pattern, '', FileName)
    if InvalidCh:
        Logger.Error('Unicode File Parser', ToolError.FORMAT_INVALID, Message=ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID, ExtraData=Path)