"""
StringUtils
"""
import re
import os.path
import Logger.Log as Logger
import Library.DataType as DataType
from Logger.ToolError import FORMAT_INVALID
from Logger.ToolError import PARSER_ERROR
from Logger import StringTable as ST
gMACRO_PATTERN = re.compile('\\$\\(([_A-Z][_A-Z0-9]*)\\)', re.UNICODE)

def GetSplitValueList(String, SplitTag=DataType.TAB_VALUE_SPLIT, MaxSplit=-1):
    if False:
        return 10
    return list(map(lambda l: l.strip(), String.split(SplitTag, MaxSplit)))

def MergeArches(Dict, Key, Arch):
    if False:
        return 10
    if Key in Dict.keys():
        Dict[Key].append(Arch)
    else:
        Dict[Key] = Arch.split()

def GenDefines(String, Arch, Defines):
    if False:
        i = 10
        return i + 15
    if String.find(DataType.TAB_DEFINE + ' ') > -1:
        List = String.replace(DataType.TAB_DEFINE + ' ', '').split(DataType.TAB_EQUAL_SPLIT)
        if len(List) == 2:
            Defines[CleanString(List[0]), Arch] = CleanString(List[1])
            return 0
        else:
            return -1
    return 1

def GetLibraryClassesWithModuleType(Lines, Key, KeyValues, CommentCharacter):
    if False:
        return 10
    NewKey = SplitModuleType(Key)
    Lines = Lines.split(DataType.TAB_SECTION_END, 1)[1]
    LineList = Lines.splitlines()
    for Line in LineList:
        Line = CleanString(Line, CommentCharacter)
        if Line != '' and Line[0] != CommentCharacter:
            KeyValues.append([CleanString(Line, CommentCharacter), NewKey[1]])
    return True

def GetDynamics(Lines, Key, KeyValues, CommentCharacter):
    if False:
        for i in range(10):
            print('nop')
    SkuIdNameList = SplitModuleType(Key)
    Lines = Lines.split(DataType.TAB_SECTION_END, 1)[1]
    LineList = Lines.splitlines()
    for Line in LineList:
        Line = CleanString(Line, CommentCharacter)
        if Line != '' and Line[0] != CommentCharacter:
            KeyValues.append([CleanString(Line, CommentCharacter), SkuIdNameList[1]])
    return True

def SplitModuleType(Key):
    if False:
        i = 10
        return i + 15
    KeyList = Key.split(DataType.TAB_SPLIT)
    KeyList.append('')
    KeyList.append('')
    ReturnValue = []
    KeyValue = KeyList[0]
    if KeyList[1] != '':
        KeyValue = KeyValue + DataType.TAB_SPLIT + KeyList[1]
    ReturnValue.append(KeyValue)
    ReturnValue.append(GetSplitValueList(KeyList[2]))
    return ReturnValue

def ReplaceMacro(String, MacroDefinitions=None, SelfReplacement=False, Line=None, FileName=None, Flag=False):
    if False:
        print('Hello World!')
    LastString = String
    if MacroDefinitions is None:
        MacroDefinitions = {}
    while MacroDefinitions:
        QuotedStringList = []
        HaveQuotedMacroFlag = False
        if not Flag:
            MacroUsed = gMACRO_PATTERN.findall(String)
        else:
            ReQuotedString = re.compile('"')
            QuotedStringList = ReQuotedString.split(String)
            if len(QuotedStringList) >= 3:
                HaveQuotedMacroFlag = True
            Count = 0
            MacroString = ''
            for QuotedStringItem in QuotedStringList:
                Count += 1
                if Count % 2 != 0:
                    MacroString += QuotedStringItem
                if Count == len(QuotedStringList) and Count % 2 == 0:
                    MacroString += QuotedStringItem
            MacroUsed = gMACRO_PATTERN.findall(MacroString)
        if len(MacroUsed) == 0:
            break
        for Macro in MacroUsed:
            if Macro not in MacroDefinitions:
                if SelfReplacement:
                    String = String.replace('$(%s)' % Macro, '')
                    Logger.Debug(5, 'Delete undefined MACROs in file %s line %d: %s!' % (FileName, Line[1], Line[0]))
                continue
            if not HaveQuotedMacroFlag:
                String = String.replace('$(%s)' % Macro, MacroDefinitions[Macro])
            else:
                Count = 0
                for QuotedStringItem in QuotedStringList:
                    Count += 1
                    if Count % 2 != 0:
                        QuotedStringList[Count - 1] = QuotedStringList[Count - 1].replace('$(%s)' % Macro, MacroDefinitions[Macro])
                    elif Count == len(QuotedStringList) and Count % 2 == 0:
                        QuotedStringList[Count - 1] = QuotedStringList[Count - 1].replace('$(%s)' % Macro, MacroDefinitions[Macro])
        RetString = ''
        if HaveQuotedMacroFlag:
            Count = 0
            for QuotedStringItem in QuotedStringList:
                Count += 1
                if Count != len(QuotedStringList):
                    RetString += QuotedStringList[Count - 1] + '"'
                else:
                    RetString += QuotedStringList[Count - 1]
            String = RetString
        if String == LastString:
            break
        LastString = String
    return String

def NormPath(Path, Defines=None):
    if False:
        print('Hello World!')
    IsRelativePath = False
    if Defines is None:
        Defines = {}
    if Path:
        if Path[0] == '.':
            IsRelativePath = True
        if Defines:
            Path = ReplaceMacro(Path, Defines)
        Path = os.path.normpath(Path)
    if IsRelativePath and Path[0] != '.':
        Path = os.path.join('.', Path)
    return Path

def CleanString(Line, CommentCharacter=DataType.TAB_COMMENT_SPLIT, AllowCppStyleComment=False):
    if False:
        i = 10
        return i + 15
    Line = Line.strip()
    if AllowCppStyleComment:
        Line = Line.replace(DataType.TAB_COMMENT_EDK1_SPLIT, CommentCharacter)
    InString = False
    for Index in range(0, len(Line)):
        if Line[Index] == '"':
            InString = not InString
        elif Line[Index] == CommentCharacter and (not InString):
            Line = Line[0:Index]
            break
    Line = Line.strip()
    return Line

def CleanString2(Line, CommentCharacter=DataType.TAB_COMMENT_SPLIT, AllowCppStyleComment=False):
    if False:
        return 10
    Line = Line.strip()
    if AllowCppStyleComment:
        Line = Line.replace(DataType.TAB_COMMENT_EDK1_SPLIT, CommentCharacter)
    LineParts = Line.split(CommentCharacter, 1)
    Line = LineParts[0].strip()
    if len(LineParts) > 1:
        Comment = LineParts[1].strip()
        Start = 0
        End = len(Comment)
        while Start < End and Comment.startswith(CommentCharacter, Start, End):
            Start += 1
        while End >= 0 and Comment.endswith(CommentCharacter, Start, End):
            End -= 1
        Comment = Comment[Start:End]
        Comment = Comment.strip()
    else:
        Comment = ''
    return (Line, Comment)

def GetMultipleValuesOfKeyFromLines(Lines, Key, KeyValues, CommentCharacter):
    if False:
        for i in range(10):
            print('nop')
    if Key:
        pass
    if KeyValues:
        pass
    Lines = Lines.split(DataType.TAB_SECTION_END, 1)[1]
    LineList = Lines.split('\n')
    for Line in LineList:
        Line = CleanString(Line, CommentCharacter)
        if Line != '' and Line[0] != CommentCharacter:
            KeyValues += [Line]
    return True

def GetDefineValue(String, Key, CommentCharacter):
    if False:
        print('Hello World!')
    if CommentCharacter:
        pass
    String = CleanString(String)
    return String[String.find(Key + ' ') + len(Key + ' '):]

def GetSingleValueOfKeyFromLines(Lines, Dictionary, CommentCharacter, KeySplitCharacter, ValueSplitFlag, ValueSplitCharacter):
    if False:
        i = 10
        return i + 15
    Lines = Lines.split('\n')
    Keys = []
    Value = ''
    DefineValues = ['']
    SpecValues = ['']
    for Line in Lines:
        if Line.find(DataType.TAB_INF_DEFINES_DEFINE + ' ') > -1:
            if '' in DefineValues:
                DefineValues.remove('')
            DefineValues.append(GetDefineValue(Line, DataType.TAB_INF_DEFINES_DEFINE, CommentCharacter))
            continue
        if Line.find(DataType.TAB_INF_DEFINES_SPEC + ' ') > -1:
            if '' in SpecValues:
                SpecValues.remove('')
            SpecValues.append(GetDefineValue(Line, DataType.TAB_INF_DEFINES_SPEC, CommentCharacter))
            continue
        LineList = Line.split(KeySplitCharacter, 1)
        if len(LineList) >= 2:
            Key = LineList[0].split()
            if len(Key) == 1 and Key[0][0] != CommentCharacter:
                LineList[1] = CleanString(LineList[1], CommentCharacter)
                if ValueSplitFlag:
                    Value = list(map(lambda x: x.strip(), LineList[1].split(ValueSplitCharacter)))
                else:
                    Value = CleanString(LineList[1], CommentCharacter).splitlines()
                if Key[0] in Dictionary:
                    if Key[0] not in Keys:
                        Dictionary[Key[0]] = Value
                        Keys.append(Key[0])
                    else:
                        Dictionary[Key[0]].extend(Value)
                else:
                    Dictionary[DataType.TAB_INF_DEFINES_MACRO][Key[0]] = Value[0]
    if DefineValues == []:
        DefineValues = ['']
    if SpecValues == []:
        SpecValues = ['']
    Dictionary[DataType.TAB_INF_DEFINES_DEFINE] = DefineValues
    Dictionary[DataType.TAB_INF_DEFINES_SPEC] = SpecValues
    return True

def PreCheck(FileName, FileContent, SupSectionTag):
    if False:
        for i in range(10):
            print('nop')
    if SupSectionTag:
        pass
    LineNo = 0
    IsFailed = False
    NewFileContent = ''
    for Line in FileContent.splitlines():
        LineNo = LineNo + 1
        Line = CleanString(Line)
        if Line.find(DataType.TAB_COMMA_SPLIT) == 0:
            Line = ''
        if Line.find('$') > -1:
            if Line.find('$(') < 0 or Line.find(')') < 0:
                Logger.Error('Parser', FORMAT_INVALID, Line=LineNo, File=FileName, RaiseError=Logger.IS_RAISE_ERROR)
        if Line.find('[') > -1 or Line.find(']') > -1:
            if not (Line.find('[') > -1 and Line.find(']') > -1):
                Logger.Error('Parser', FORMAT_INVALID, Line=LineNo, File=FileName, RaiseError=Logger.IS_RAISE_ERROR)
        NewFileContent = NewFileContent + Line + '\r\n'
    if IsFailed:
        Logger.Error('Parser', FORMAT_INVALID, Line=LineNo, File=FileName, RaiseError=Logger.IS_RAISE_ERROR)
    return NewFileContent

def CheckFileType(CheckFilename, ExtName, ContainerFilename, SectionName, Line, LineNo=-1):
    if False:
        for i in range(10):
            print('nop')
    if CheckFilename != '' and CheckFilename is not None:
        (Root, Ext) = os.path.splitext(CheckFilename)
        if Ext.upper() != ExtName.upper() and Root:
            ContainerFile = open(ContainerFilename, 'r').read()
            if LineNo == -1:
                LineNo = GetLineNo(ContainerFile, Line)
            ErrorMsg = ST.ERR_SECTIONNAME_INVALID % (SectionName, CheckFilename, ExtName)
            Logger.Error('Parser', PARSER_ERROR, ErrorMsg, Line=LineNo, File=ContainerFilename, RaiseError=Logger.IS_RAISE_ERROR)
    return True

def CheckFileExist(WorkspaceDir, CheckFilename, ContainerFilename, SectionName, Line, LineNo=-1):
    if False:
        while True:
            i = 10
    CheckFile = ''
    if CheckFilename != '' and CheckFilename is not None:
        CheckFile = WorkspaceFile(WorkspaceDir, CheckFilename)
        if not os.path.isfile(CheckFile):
            ContainerFile = open(ContainerFilename, 'r').read()
            if LineNo == -1:
                LineNo = GetLineNo(ContainerFile, Line)
            ErrorMsg = ST.ERR_CHECKFILE_NOTFOUND % (CheckFile, SectionName)
            Logger.Error('Parser', PARSER_ERROR, ErrorMsg, File=ContainerFilename, Line=LineNo, RaiseError=Logger.IS_RAISE_ERROR)
    return CheckFile

def GetLineNo(FileContent, Line, IsIgnoreComment=True):
    if False:
        i = 10
        return i + 15
    LineList = FileContent.splitlines()
    for Index in range(len(LineList)):
        if LineList[Index].find(Line) > -1:
            if IsIgnoreComment:
                if LineList[Index].strip()[0] == DataType.TAB_COMMENT_SPLIT:
                    continue
            return Index + 1
    return -1

def RaiseParserError(Line, Section, File, Format='', LineNo=-1):
    if False:
        i = 10
        return i + 15
    if LineNo == -1:
        LineNo = GetLineNo(open(os.path.normpath(File), 'r').read(), Line)
    ErrorMsg = ST.ERR_INVALID_NOTFOUND % (Line, Section)
    if Format != '':
        Format = 'Correct format is ' + Format
    Logger.Error('Parser', PARSER_ERROR, ErrorMsg, File=File, Line=LineNo, ExtraData=Format, RaiseError=Logger.IS_RAISE_ERROR)

def WorkspaceFile(WorkspaceDir, Filename):
    if False:
        return 10
    return os.path.join(NormPath(WorkspaceDir), NormPath(Filename))

def SplitString(String):
    if False:
        for i in range(10):
            print('nop')
    if String.startswith('"'):
        String = String[1:]
    if String.endswith('"'):
        String = String[:-1]
    return String

def ConvertToSqlString(StringList):
    if False:
        i = 10
        return i + 15
    return list(map(lambda s: s.replace("'", "''"), StringList))

def ConvertToSqlString2(String):
    if False:
        print('Hello World!')
    return String.replace("'", "''")

def GetStringOfList(List, Split=' '):
    if False:
        while True:
            i = 10
    if not isinstance(List, type([])):
        return List
    Str = ''
    for Item in List:
        Str = Str + Item + Split
    return Str.strip()

def GetHelpTextList(HelpTextClassList):
    if False:
        while True:
            i = 10
    List = []
    if HelpTextClassList:
        for HelpText in HelpTextClassList:
            if HelpText.String.endswith('\n'):
                HelpText.String = HelpText.String[0:len(HelpText.String) - len('\n')]
                List.extend(HelpText.String.split('\n'))
    return List

def StringArrayLength(String):
    if False:
        for i in range(10):
            print('nop')
    if String.startswith('L"'):
        return (len(String) - 3 + 1) * 2
    elif String.startswith('"'):
        return len(String) - 2 + 1
    else:
        return len(String.split()) + 1

def RemoveDupOption(OptionString, Which='/I', Against=None):
    if False:
        i = 10
        return i + 15
    OptionList = OptionString.split()
    ValueList = []
    if Against:
        ValueList += Against
    for Index in range(len(OptionList)):
        Opt = OptionList[Index]
        if not Opt.startswith(Which):
            continue
        if len(Opt) > len(Which):
            Val = Opt[len(Which):]
        else:
            Val = ''
        if Val in ValueList:
            OptionList[Index] = ''
        else:
            ValueList.append(Val)
    return ' '.join(OptionList)

def IsHexDigit(Str):
    if False:
        print('Hello World!')
    try:
        int(Str, 10)
        return True
    except ValueError:
        if len(Str) > 2 and Str.upper().startswith('0X'):
            try:
                int(Str, 16)
                return True
            except ValueError:
                return False
    return False

def IsHexDigitUINT32(Str):
    if False:
        return 10
    try:
        Value = int(Str, 10)
        if Value <= 4294967295 and Value >= 0:
            return True
    except ValueError:
        if len(Str) > 2 and Str.upper().startswith('0X'):
            try:
                Value = int(Str, 16)
                if Value <= 4294967295 and Value >= 0:
                    return True
            except ValueError:
                return False
    return False

def ConvertSpecialChar(Lines):
    if False:
        print('Hello World!')
    RetLines = []
    for line in Lines:
        ReMatchSpecialChar = re.compile('[\\x00-\\x08]|\\x09|\\x0b|\\x0c|[\\x0e-\\x1f]|[\\x7f-\\xff]')
        RetLines.append(ReMatchSpecialChar.sub(' ', line))
    return RetLines

def __GetTokenList(Str):
    if False:
        print('Hello World!')
    InQuote = False
    Token = ''
    TokenOP = ''
    PreChar = ''
    List = []
    for Char in Str:
        if InQuote:
            Token += Char
            if Char == '"' and PreChar != '\\':
                InQuote = not InQuote
                List.append(Token)
                Token = ''
            continue
        if Char == '"':
            if Token and Token != 'L':
                List.append(Token)
                Token = ''
            if TokenOP:
                List.append(TokenOP)
                TokenOP = ''
            InQuote = not InQuote
            Token += Char
            continue
        if not (Char.isalnum() or Char in '_'):
            TokenOP += Char
            if Token:
                List.append(Token)
                Token = ''
        else:
            Token += Char
            if TokenOP:
                List.append(TokenOP)
                TokenOP = ''
        if PreChar == '\\' and Char == '\\':
            PreChar = ''
        else:
            PreChar = Char
    if Token:
        List.append(Token)
    if TokenOP:
        List.append(TokenOP)
    return List

def ConvertNEToNOTEQ(Expr):
    if False:
        return 10
    List = __GetTokenList(Expr)
    for Index in range(len(List)):
        if List[Index] == 'NE':
            List[Index] = 'NOT EQ'
    return ''.join(List)

def ConvertNOTEQToNE(Expr):
    if False:
        while True:
            i = 10
    List = __GetTokenList(Expr)
    HasNOT = False
    RetList = []
    for Token in List:
        if HasNOT and Token == 'EQ':
            while not RetList[-1].strip():
                RetList.pop()
            RetList[-1] = 'NE'
            HasNOT = False
            continue
        if Token == 'NOT':
            HasNOT = True
        elif Token.strip():
            HasNOT = False
        RetList.append(Token)
    return ''.join(RetList)

def SplitPcdEntry(String):
    if False:
        return 10
    if not String:
        return (['', '', ''], False)
    PcdTokenCName = ''
    PcdValue = ''
    PcdFeatureFlagExp = ''
    ValueList = GetSplitValueList(String, '|', 1)
    if len(ValueList) == 1:
        return ([ValueList[0]], True)
    NewValueList = []
    if len(ValueList) == 2:
        PcdTokenCName = ValueList[0]
        InQuote = False
        InParenthesis = False
        StrItem = ''
        for StrCh in ValueList[1]:
            if StrCh == '"':
                InQuote = not InQuote
            elif StrCh == '(' or StrCh == ')':
                InParenthesis = not InParenthesis
            if StrCh == '|':
                if not InQuote or not InParenthesis:
                    NewValueList.append(StrItem.strip())
                    StrItem = ' '
                    continue
            StrItem += StrCh
        NewValueList.append(StrItem.strip())
        if len(NewValueList) == 1:
            PcdValue = NewValueList[0]
            return ([PcdTokenCName, PcdValue], True)
        elif len(NewValueList) == 2:
            PcdValue = NewValueList[0]
            PcdFeatureFlagExp = NewValueList[1]
            return ([PcdTokenCName, PcdValue, PcdFeatureFlagExp], True)
        else:
            return (['', '', ''], False)
    return (['', '', ''], False)

def IsMatchArch(Arch1, Arch2):
    if False:
        return 10
    if 'COMMON' in Arch1 or 'COMMON' in Arch2:
        return True
    try:
        if isinstance(Arch1, list) and isinstance(Arch2, list):
            for Item1 in Arch1:
                for Item2 in Arch2:
                    if Item1 == Item2:
                        return True
        elif isinstance(Arch1, list):
            return Arch2 in Arch1
        elif isinstance(Arch2, list):
            return Arch1 in Arch2
        elif Arch1 == Arch2:
            return True
    except:
        return False

def GetUniFileName(FilePath, FileName):
    if False:
        while True:
            i = 10
    Files = []
    try:
        Files = os.listdir(FilePath)
    except:
        pass
    LargestIndex = -1
    IndexNotFound = True
    for File in Files:
        if File.upper().startswith(FileName.upper()) and File.upper().endswith('.UNI'):
            Index = File.upper().replace(FileName.upper(), '').replace('.UNI', '')
            if Index:
                try:
                    Index = int(Index)
                except Exception:
                    Index = -1
            else:
                IndexNotFound = False
                Index = 0
            if Index > LargestIndex:
                LargestIndex = Index + 1
    if LargestIndex > -1 and (not IndexNotFound):
        return os.path.normpath(os.path.join(FilePath, FileName + str(LargestIndex) + '.uni'))
    else:
        return os.path.normpath(os.path.join(FilePath, FileName + '.uni'))