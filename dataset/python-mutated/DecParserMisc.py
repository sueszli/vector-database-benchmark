"""
DecParserMisc
"""
import os
import Logger.Log as Logger
from Logger.ToolError import FILE_PARSE_FAILURE
from Logger import StringTable as ST
from Library.DataType import TAB_COMMENT_SPLIT
from Library.DataType import TAB_COMMENT_EDK1_SPLIT
from Library.ExpressionValidate import IsValidBareCString
from Library.ParserValidate import IsValidCFormatGuid
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Library.ExpressionValidate import IsValidLogicalExpr
from Library.ExpressionValidate import IsValidStringTest
from Library.Misc import CheckGuidRegFormat
TOOL_NAME = 'DecParser'
VERSION_PATTERN = '[0-9]+(\\.[0-9]+)?'
CVAR_PATTERN = '[_a-zA-Z][a-zA-Z0-9_]*'
PCD_TOKEN_PATTERN = '(0[xX]0*[a-fA-F0-9]{1,8})|([0-9]+)'
MACRO_PATTERN = '[A-Z][_A-Z0-9]*'

class FileContent:

    def __init__(self, Filename, FileContent2):
        if False:
            return 10
        self.Filename = Filename
        (self.PackagePath, self.PackageFile) = os.path.split(Filename)
        self.LineIndex = 0
        self.CurrentLine = ''
        self.NextLine = ''
        self.HeadComment = []
        self.TailComment = []
        self.CurrentScope = None
        self.Content = FileContent2
        self.Macros = {}
        self.FileLines = len(FileContent2)

    def GetNextLine(self):
        if False:
            for i in range(10):
                print('nop')
        if self.LineIndex >= self.FileLines:
            return ''
        Line = self.Content[self.LineIndex]
        self.LineIndex += 1
        return Line

    def UndoNextLine(self):
        if False:
            for i in range(10):
                print('nop')
        if self.LineIndex > 0:
            self.LineIndex -= 1

    def ResetNext(self):
        if False:
            print('Hello World!')
        self.HeadComment = []
        self.TailComment = []
        self.NextLine = ''

    def SetNext(self, Line, HeadComment, TailComment):
        if False:
            i = 10
            return i + 15
        self.NextLine = Line
        self.HeadComment = HeadComment
        self.TailComment = TailComment

    def IsEndOfFile(self):
        if False:
            while True:
                i = 10
        return self.LineIndex >= self.FileLines

def StripRoot(Root, Path):
    if False:
        print('Hello World!')
    OrigPath = Path
    Root = os.path.normpath(Root)
    Path = os.path.normpath(Path)
    if not os.path.isabs(Root):
        return OrigPath
    if Path.startswith(Root):
        Path = Path[len(Root):]
        if Path and Path[0] == os.sep:
            Path = Path[1:]
        return Path
    return OrigPath

def CleanString(Line, CommentCharacter=TAB_COMMENT_SPLIT, AllowCppStyleComment=False):
    if False:
        i = 10
        return i + 15
    Line = Line.strip()
    if AllowCppStyleComment:
        Line = Line.replace(TAB_COMMENT_EDK1_SPLIT, CommentCharacter)
    Comment = ''
    InQuote = False
    for Index in range(0, len(Line)):
        if Line[Index] == '"':
            InQuote = not InQuote
            continue
        if Line[Index] == CommentCharacter and (not InQuote):
            Comment = Line[Index:].strip()
            Line = Line[0:Index].strip()
            break
    return (Line, Comment)

def IsValidNumValUint8(Token):
    if False:
        for i in range(10):
            print('nop')
    Valid = True
    Cause = ''
    TokenValue = None
    Token = Token.strip()
    if Token.lower().startswith('0x'):
        Base = 16
    else:
        Base = 10
    try:
        TokenValue = int(Token, Base)
    except BaseException:
        (Valid, Cause) = IsValidLogicalExpr(Token, True)
        if Cause:
            pass
    if not Valid:
        return False
    if TokenValue and (TokenValue < 0 or TokenValue > 255):
        return False
    else:
        return True

def IsValidNList(Value):
    if False:
        while True:
            i = 10
    Par = ParserHelper(Value)
    if Par.End():
        return False
    while not Par.End():
        Token = Par.GetToken(',')
        if not IsValidNumValUint8(Token):
            return False
        if Par.Expect(','):
            if Par.End():
                return False
            continue
        else:
            break
    return Par.End()

def IsValidCArray(Array):
    if False:
        print('Hello World!')
    Par = ParserHelper(Array)
    if not Par.Expect('{'):
        return False
    if Par.End():
        return False
    while not Par.End():
        Token = Par.GetToken(',}')
        if not IsValidNumValUint8(Token):
            return False
        if Par.Expect(','):
            if Par.End():
                return False
            continue
        elif Par.Expect('}'):
            break
        else:
            return False
    return Par.End()

def IsValidPcdDatum(Type, Value):
    if False:
        for i in range(10):
            print('nop')
    if not Value:
        return (False, ST.ERR_DECPARSE_PCD_VALUE_EMPTY)
    Valid = True
    Cause = ''
    if Type not in ['UINT8', 'UINT16', 'UINT32', 'UINT64', 'VOID*', 'BOOLEAN']:
        return (False, ST.ERR_DECPARSE_PCD_TYPE)
    if Type == 'VOID*':
        if not ((Value.startswith('L"') or (Value.startswith('"') and Value.endswith('"'))) or IsValidCArray(Value) or IsValidCFormatGuid(Value) or IsValidNList(Value) or CheckGuidRegFormat(Value)):
            return (False, ST.ERR_DECPARSE_PCD_VOID % (Value, Type))
        RealString = Value[Value.find('"') + 1:-1]
        if RealString:
            if not IsValidBareCString(RealString):
                return (False, ST.ERR_DECPARSE_PCD_VOID % (Value, Type))
    elif Type == 'BOOLEAN':
        if Value in ['TRUE', 'FALSE', 'true', 'false', 'True', 'False', '0x1', '0x01', '1', '0x0', '0x00', '0']:
            return (True, '')
        (Valid, Cause) = IsValidStringTest(Value, True)
        if not Valid:
            (Valid, Cause) = IsValidFeatureFlagExp(Value, True)
        if not Valid:
            return (False, Cause)
    else:
        if Value and (Value[0] == '-' or Value[0] == '+'):
            return (False, ST.ERR_DECPARSE_PCD_INT_NEGTIVE % (Value, Type))
        try:
            StrVal = Value
            if Value and (not Value.startswith('0x')) and (not Value.startswith('0X')):
                Value = Value.lstrip('0')
                if not Value:
                    return (True, '')
            Value = int(Value, 0)
            MAX_VAL_TYPE = {'BOOLEAN': 1, 'UINT8': 255, 'UINT16': 65535, 'UINT32': 4294967295, 'UINT64': 18446744073709551615}
            if Value > MAX_VAL_TYPE[Type]:
                return (False, ST.ERR_DECPARSE_PCD_INT_EXCEED % (StrVal, Type))
        except BaseException:
            (Valid, Cause) = IsValidLogicalExpr(Value, True)
        if not Valid:
            return (False, Cause)
    return (True, '')

class ParserHelper:

    def __init__(self, String, File=''):
        if False:
            for i in range(10):
                print('nop')
        self._String = String
        self._StrLen = len(String)
        self._Index = 0
        self._File = File

    def End(self):
        if False:
            print('Hello World!')
        self.__SkipWhitespace()
        return self._Index >= self._StrLen

    def __SkipWhitespace(self):
        if False:
            while True:
                i = 10
        for Char in self._String[self._Index:]:
            if Char not in ' \t':
                break
            self._Index += 1

    def Expect(self, ExpectChar):
        if False:
            print('Hello World!')
        self.__SkipWhitespace()
        for Char in self._String[self._Index:]:
            if Char != ExpectChar:
                return False
            else:
                self._Index += 1
                return True
        return False

    def GetToken(self, StopChar='.,|\t ', SkipPair='"'):
        if False:
            print('Hello World!')
        self.__SkipWhitespace()
        PreIndex = self._Index
        InQuote = False
        LastChar = ''
        for Char in self._String[self._Index:]:
            if Char == SkipPair and LastChar != '\\':
                InQuote = not InQuote
            if Char in StopChar and (not InQuote):
                break
            self._Index += 1
            if Char == '\\' and LastChar == '\\':
                LastChar = ''
            else:
                LastChar = Char
        return self._String[PreIndex:self._Index]

    def AssertChar(self, AssertChar, ErrorString, ErrorLineNum):
        if False:
            while True:
                i = 10
        if not self.Expect(AssertChar):
            Logger.Error(TOOL_NAME, FILE_PARSE_FAILURE, File=self._File, Line=ErrorLineNum, ExtraData=ErrorString)

    def AssertEnd(self, ErrorString, ErrorLineNum):
        if False:
            i = 10
            return i + 15
        self.__SkipWhitespace()
        if self._Index != self._StrLen:
            Logger.Error(TOOL_NAME, FILE_PARSE_FAILURE, File=self._File, Line=ErrorLineNum, ExtraData=ErrorString)