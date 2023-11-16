"""
DecParser
"""
import Logger.Log as Logger
from Logger.ToolError import FILE_PARSE_FAILURE
from Logger.ToolError import FILE_OPEN_FAILURE
from Logger import StringTable as ST
from Logger.ToolError import FORMAT_INVALID
import Library.DataType as DT
from Library.ParserValidate import IsValidToken
from Library.ParserValidate import IsValidPath
from Library.ParserValidate import IsValidCFormatGuid
from Library.ParserValidate import IsValidIdString
from Library.ParserValidate import IsValidUserId
from Library.ParserValidate import IsValidArch
from Library.ParserValidate import IsValidWord
from Library.ParserValidate import IsValidDecVersionVal
from Parser.DecParserMisc import TOOL_NAME
from Parser.DecParserMisc import CleanString
from Parser.DecParserMisc import IsValidPcdDatum
from Parser.DecParserMisc import ParserHelper
from Parser.DecParserMisc import StripRoot
from Parser.DecParserMisc import VERSION_PATTERN
from Parser.DecParserMisc import CVAR_PATTERN
from Parser.DecParserMisc import PCD_TOKEN_PATTERN
from Parser.DecParserMisc import MACRO_PATTERN
from Parser.DecParserMisc import FileContent
from Object.Parser.DecObject import _DecComments
from Object.Parser.DecObject import DecDefineObject
from Object.Parser.DecObject import DecDefineItemObject
from Object.Parser.DecObject import DecIncludeObject
from Object.Parser.DecObject import DecIncludeItemObject
from Object.Parser.DecObject import DecLibraryclassObject
from Object.Parser.DecObject import DecLibraryclassItemObject
from Object.Parser.DecObject import DecGuidObject
from Object.Parser.DecObject import DecPpiObject
from Object.Parser.DecObject import DecProtocolObject
from Object.Parser.DecObject import DecGuidItemObject
from Object.Parser.DecObject import DecUserExtensionObject
from Object.Parser.DecObject import DecUserExtensionItemObject
from Object.Parser.DecObject import DecPcdObject
from Object.Parser.DecObject import DecPcdItemObject
from Library.Misc import GuidStructureStringToGuidString
from Library.Misc import CheckGuidRegFormat
from Library.StringUtils import ReplaceMacro
from Library.StringUtils import GetSplitValueList
from Library.StringUtils import gMACRO_PATTERN
from Library.StringUtils import ConvertSpecialChar
from Library.CommentParsing import ParsePcdErrorCode

class _DecBase:

    def __init__(self, RawData):
        if False:
            return 10
        self._RawData = RawData
        self._ItemDict = {}
        self._LocalMacro = {}
        self.ItemObject = None

    def GetDataObject(self):
        if False:
            print('Hello World!')
        return self.ItemObject

    def GetLocalMacro(self):
        if False:
            for i in range(10):
                print('nop')
        return self._LocalMacro

    def BlockStart(self):
        if False:
            for i in range(10):
                print('nop')
        self._LocalMacro = {}

    def _CheckReDefine(self, Key, Scope=None):
        if False:
            for i in range(10):
                print('nop')
        if not Scope:
            Scope = self._RawData.CurrentScope
            return
        SecArch = []
        SecArch[0:1] = Scope[:]
        if Key not in self._ItemDict:
            self._ItemDict[Key] = [[SecArch, self._RawData.LineIndex]]
            return
        for Value in self._ItemDict[Key]:
            for SubValue in Scope:
                if SubValue[-1] == 'COMMON':
                    for Other in Value[0]:
                        if Other[:-1] == SubValue[:-1]:
                            self._LoggerError(ST.ERR_DECPARSE_REDEFINE % (Key, Value[1]))
                            return
                    continue
                CommonScope = []
                CommonScope[0:1] = SubValue
                CommonScope[-1] = 'COMMON'
                if SubValue in Value[0] or CommonScope in Value[0]:
                    self._LoggerError(ST.ERR_DECPARSE_REDEFINE % (Key, Value[1]))
                    return
        self._ItemDict[Key].append([SecArch, self._RawData.LineIndex])

    def CheckRequiredFields(self):
        if False:
            while True:
                i = 10
        if self._RawData:
            pass
        return True

    def _IsStatementRequired(self):
        if False:
            return 10
        if self._RawData:
            pass
        return False

    def _LoggerError(self, ErrorString):
        if False:
            for i in range(10):
                print('nop')
        Logger.Error(TOOL_NAME, FILE_PARSE_FAILURE, File=self._RawData.Filename, Line=self._RawData.LineIndex, ExtraData=ErrorString + ST.ERR_DECPARSE_LINE % self._RawData.CurrentLine)

    def _ReplaceMacro(self, String):
        if False:
            for i in range(10):
                print('nop')
        if gMACRO_PATTERN.findall(String):
            String = ReplaceMacro(String, self._LocalMacro, False, FileName=self._RawData.Filename, Line=['', self._RawData.LineIndex])
            String = ReplaceMacro(String, self._RawData.Macros, False, FileName=self._RawData.Filename, Line=['', self._RawData.LineIndex])
            MacroUsed = gMACRO_PATTERN.findall(String)
            if MacroUsed:
                Logger.Error(TOOL_NAME, FILE_PARSE_FAILURE, File=self._RawData.Filename, Line=self._RawData.LineIndex, ExtraData=ST.ERR_DECPARSE_MACRO_RESOLVE % (str(MacroUsed), String))
        return String

    def _MacroParser(self, String):
        if False:
            print('Hello World!')
        TokenList = GetSplitValueList(String, ' ', 1)
        if len(TokenList) < 2 or TokenList[1] == '':
            self._LoggerError(ST.ERR_DECPARSE_MACRO_PAIR)
        TokenList = GetSplitValueList(TokenList[1], DT.TAB_EQUAL_SPLIT, 1)
        if TokenList[0] == '':
            self._LoggerError(ST.ERR_DECPARSE_MACRO_NAME)
        elif not IsValidToken(MACRO_PATTERN, TokenList[0]):
            self._LoggerError(ST.ERR_DECPARSE_MACRO_NAME_UPPER % TokenList[0])
        if len(TokenList) == 1:
            self._LocalMacro[TokenList[0]] = ''
        else:
            self._LocalMacro[TokenList[0]] = self._ReplaceMacro(TokenList[1])

    def _ParseItem(self):
        if False:
            while True:
                i = 10
        if self._RawData:
            pass
        return None

    def _TailCommentStrategy(self, Comment):
        if False:
            return 10
        if Comment:
            pass
        if self._RawData:
            pass
        return False

    def _StopCurrentParsing(self, Line):
        if False:
            print('Hello World!')
        if self._RawData:
            pass
        return Line[0] == DT.TAB_SECTION_START and Line[-1] == DT.TAB_SECTION_END

    def _TryBackSlash(self, ProcessedLine, ProcessedComments):
        if False:
            i = 10
            return i + 15
        CatLine = ''
        Comment = ''
        Line = ProcessedLine
        CommentList = ProcessedComments
        while not self._RawData.IsEndOfFile():
            if Line == '':
                self._LoggerError(ST.ERR_DECPARSE_BACKSLASH_EMPTY)
                break
            if Comment:
                CommentList.append((Comment, self._RawData.LineIndex))
            if Line[-1] != DT.TAB_SLASH:
                CatLine += Line
                break
            elif len(Line) < 2 or Line[-2] != ' ':
                self._LoggerError(ST.ERR_DECPARSE_BACKSLASH)
            else:
                CatLine += Line[:-1]
                (Line, Comment) = CleanString(self._RawData.GetNextLine())
        if self._RawData.IsEndOfFile():
            if not CatLine:
                if ProcessedLine[-1] == DT.TAB_SLASH:
                    self._LoggerError(ST.ERR_DECPARSE_BACKSLASH_EMPTY)
                CatLine = ProcessedLine
            else:
                if not Line or Line[-1] == DT.TAB_SLASH:
                    self._LoggerError(ST.ERR_DECPARSE_BACKSLASH_EMPTY)
                CatLine += Line
        __IsReplaceMacro = True
        Header = self._RawData.CurrentScope[0] if self._RawData.CurrentScope else None
        if Header and len(Header) > 2:
            if Header[0].upper() == 'USEREXTENSIONS' and (not (Header[1] == 'TianoCore' and Header[2] == '"ExtraFiles"')):
                __IsReplaceMacro = False
        if __IsReplaceMacro:
            self._RawData.CurrentLine = self._ReplaceMacro(CatLine)
        else:
            self._RawData.CurrentLine = CatLine
        return (CatLine, CommentList)

    def Parse(self):
        if False:
            while True:
                i = 10
        HeadComments = []
        TailComments = []
        CurComments = HeadComments
        CurObj = None
        ItemNum = 0
        FromBuf = False
        Index = self._RawData.LineIndex
        LineStr = self._RawData.CurrentLine
        while not self._RawData.IsEndOfFile() or self._RawData.NextLine:
            if self._RawData.NextLine:
                Line = self._RawData.NextLine
                HeadComments.extend(self._RawData.HeadComment)
                TailComments.extend(self._RawData.TailComment)
                self._RawData.ResetNext()
                Comment = ''
                FromBuf = True
            else:
                (Line, Comment) = CleanString(self._RawData.GetNextLine())
                FromBuf = False
            if Line:
                if not FromBuf and CurObj and TailComments:
                    CurObj.SetTailComment(CurObj.GetTailComment() + TailComments)
                if not FromBuf:
                    del TailComments[:]
                CurComments = TailComments
                Comments = []
                if Comment:
                    Comments = [(Comment, self._RawData.LineIndex)]
                (Line, Comments) = self._TryBackSlash(Line, Comments)
                CurComments.extend(Comments)
                if Line.startswith('DEFINE '):
                    self._MacroParser(Line)
                    del HeadComments[:]
                    del TailComments[:]
                    CurComments = HeadComments
                    continue
                if self._StopCurrentParsing(Line):
                    self._RawData.SetNext(Line, HeadComments, TailComments)
                    break
                Obj = self._ParseItem()
                ItemNum += 1
                if Obj:
                    Obj.SetHeadComment(Obj.GetHeadComment() + HeadComments)
                    Obj.SetTailComment(Obj.GetTailComment() + TailComments)
                    del HeadComments[:]
                    del TailComments[:]
                    CurObj = Obj
                else:
                    CurObj = None
            else:
                if id(CurComments) == id(TailComments):
                    if not self._TailCommentStrategy(Comment):
                        CurComments = HeadComments
                if Comment:
                    CurComments.append((Comment, self._RawData.LineIndex))
                else:
                    del CurComments[:]
        if self._IsStatementRequired() and ItemNum == 0:
            Logger.Error(TOOL_NAME, FILE_PARSE_FAILURE, File=self._RawData.Filename, Line=Index, ExtraData=ST.ERR_DECPARSE_STATEMENT_EMPTY % LineStr)

class _DecDefine(_DecBase):

    def __init__(self, RawData):
        if False:
            i = 10
            return i + 15
        _DecBase.__init__(self, RawData)
        self.ItemObject = DecDefineObject(RawData.Filename)
        self._LocalMacro = self._RawData.Macros
        self._DefSecNum = 0
        self.DefineValidation = {DT.TAB_DEC_DEFINES_DEC_SPECIFICATION: self._SetDecSpecification, DT.TAB_DEC_DEFINES_PACKAGE_NAME: self._SetPackageName, DT.TAB_DEC_DEFINES_PACKAGE_GUID: self._SetPackageGuid, DT.TAB_DEC_DEFINES_PACKAGE_VERSION: self._SetPackageVersion, DT.TAB_DEC_DEFINES_PKG_UNI_FILE: self._SetPackageUni}

    def BlockStart(self):
        if False:
            while True:
                i = 10
        self._DefSecNum += 1
        if self._DefSecNum > 1:
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_MULTISEC)

    def CheckRequiredFields(self):
        if False:
            for i in range(10):
                print('nop')
        Ret = False
        if self.ItemObject.GetPackageSpecification() == '':
            Logger.Error(TOOL_NAME, FILE_PARSE_FAILURE, File=self._RawData.Filename, ExtraData=ST.ERR_DECPARSE_DEFINE_REQUIRED % DT.TAB_DEC_DEFINES_DEC_SPECIFICATION)
        elif self.ItemObject.GetPackageName() == '':
            Logger.Error(TOOL_NAME, FILE_PARSE_FAILURE, File=self._RawData.Filename, ExtraData=ST.ERR_DECPARSE_DEFINE_REQUIRED % DT.TAB_DEC_DEFINES_PACKAGE_NAME)
        elif self.ItemObject.GetPackageGuid() == '':
            Logger.Error(TOOL_NAME, FILE_PARSE_FAILURE, File=self._RawData.Filename, ExtraData=ST.ERR_DECPARSE_DEFINE_REQUIRED % DT.TAB_DEC_DEFINES_PACKAGE_GUID)
        elif self.ItemObject.GetPackageVersion() == '':
            Logger.Error(TOOL_NAME, FILE_PARSE_FAILURE, File=self._RawData.Filename, ExtraData=ST.ERR_DECPARSE_DEFINE_REQUIRED % DT.TAB_DEC_DEFINES_PACKAGE_VERSION)
        else:
            Ret = True
        return Ret

    def _ParseItem(self):
        if False:
            while True:
                i = 10
        Line = self._RawData.CurrentLine
        TokenList = GetSplitValueList(Line, DT.TAB_EQUAL_SPLIT, 1)
        if TokenList[0] == DT.TAB_DEC_DEFINES_PKG_UNI_FILE:
            self.DefineValidation[TokenList[0]](TokenList[1])
        elif len(TokenList) < 2:
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_FORMAT)
        elif TokenList[0] not in self.DefineValidation:
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_UNKNOWKEY % TokenList[0])
        else:
            self.DefineValidation[TokenList[0]](TokenList[1])
        DefineItem = DecDefineItemObject()
        DefineItem.Key = TokenList[0]
        DefineItem.Value = TokenList[1]
        self.ItemObject.AddItem(DefineItem, self._RawData.CurrentScope)
        return DefineItem

    def _SetDecSpecification(self, Token):
        if False:
            print('Hello World!')
        if self.ItemObject.GetPackageSpecification():
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_DEFINED % DT.TAB_DEC_DEFINES_DEC_SPECIFICATION)
        if not IsValidToken('0[xX][0-9a-fA-F]{8}', Token):
            if not IsValidDecVersionVal(Token):
                self._LoggerError(ST.ERR_DECPARSE_DEFINE_SPEC)
        self.ItemObject.SetPackageSpecification(Token)

    def _SetPackageName(self, Token):
        if False:
            print('Hello World!')
        if self.ItemObject.GetPackageName():
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_DEFINED % DT.TAB_DEC_DEFINES_PACKAGE_NAME)
        if not IsValidWord(Token):
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_PKGNAME)
        self.ItemObject.SetPackageName(Token)

    def _SetPackageGuid(self, Token):
        if False:
            for i in range(10):
                print('nop')
        if self.ItemObject.GetPackageGuid():
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_DEFINED % DT.TAB_DEC_DEFINES_PACKAGE_GUID)
        if not CheckGuidRegFormat(Token):
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_PKGGUID)
        self.ItemObject.SetPackageGuid(Token)

    def _SetPackageVersion(self, Token):
        if False:
            while True:
                i = 10
        if self.ItemObject.GetPackageVersion():
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_DEFINED % DT.TAB_DEC_DEFINES_PACKAGE_VERSION)
        if not IsValidToken(VERSION_PATTERN, Token):
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_PKGVERSION)
        else:
            if not DT.TAB_SPLIT in Token:
                Token = Token + '.0'
            self.ItemObject.SetPackageVersion(Token)

    def _SetPackageUni(self, Token):
        if False:
            while True:
                i = 10
        if self.ItemObject.GetPackageUniFile():
            self._LoggerError(ST.ERR_DECPARSE_DEFINE_DEFINED % DT.TAB_DEC_DEFINES_PKG_UNI_FILE)
        self.ItemObject.SetPackageUniFile(Token)

class _DecInclude(_DecBase):

    def __init__(self, RawData):
        if False:
            while True:
                i = 10
        _DecBase.__init__(self, RawData)
        self.ItemObject = DecIncludeObject(RawData.Filename)

    def _ParseItem(self):
        if False:
            return 10
        Line = self._RawData.CurrentLine
        if not IsValidPath(Line, self._RawData.PackagePath):
            self._LoggerError(ST.ERR_DECPARSE_INCLUDE % Line)
        Item = DecIncludeItemObject(StripRoot(self._RawData.PackagePath, Line), self._RawData.PackagePath)
        self.ItemObject.AddItem(Item, self._RawData.CurrentScope)
        return Item

class _DecLibraryclass(_DecBase):

    def __init__(self, RawData):
        if False:
            for i in range(10):
                print('nop')
        _DecBase.__init__(self, RawData)
        self.ItemObject = DecLibraryclassObject(RawData.Filename)

    def _ParseItem(self):
        if False:
            return 10
        Line = self._RawData.CurrentLine
        TokenList = GetSplitValueList(Line, DT.TAB_VALUE_SPLIT)
        if len(TokenList) != 2:
            self._LoggerError(ST.ERR_DECPARSE_LIBCLASS_SPLIT)
        if TokenList[0] == '' or TokenList[1] == '':
            self._LoggerError(ST.ERR_DECPARSE_LIBCLASS_EMPTY)
        if not IsValidToken('[A-Z][0-9A-Za-z]*', TokenList[0]):
            self._LoggerError(ST.ERR_DECPARSE_LIBCLASS_LIB)
        self._CheckReDefine(TokenList[0])
        Value = TokenList[1]
        if not Value.endswith('.h'):
            self._LoggerError(ST.ERR_DECPARSE_LIBCLASS_PATH_EXT)
        if not IsValidPath(Value, self._RawData.PackagePath):
            self._LoggerError(ST.ERR_DECPARSE_INCLUDE % Value)
        Item = DecLibraryclassItemObject(TokenList[0], StripRoot(self._RawData.PackagePath, Value), self._RawData.PackagePath)
        self.ItemObject.AddItem(Item, self._RawData.CurrentScope)
        return Item

class _DecPcd(_DecBase):

    def __init__(self, RawData):
        if False:
            for i in range(10):
                print('nop')
        _DecBase.__init__(self, RawData)
        self.ItemObject = DecPcdObject(RawData.Filename)
        self.TokenMap = {}

    def _ParseItem(self):
        if False:
            while True:
                i = 10
        Line = self._RawData.CurrentLine
        TokenList = Line.split(DT.TAB_VALUE_SPLIT)
        if len(TokenList) < 4:
            self._LoggerError(ST.ERR_DECPARSE_PCD_SPLIT)
        PcdName = GetSplitValueList(TokenList[0], DT.TAB_SPLIT)
        if len(PcdName) != 2 or PcdName[0] == '' or PcdName[1] == '':
            self._LoggerError(ST.ERR_DECPARSE_PCD_NAME)
        Guid = PcdName[0]
        if not IsValidToken(CVAR_PATTERN, Guid):
            self._LoggerError(ST.ERR_DECPARSE_PCD_CVAR_GUID)
        CName = PcdName[1]
        if not IsValidToken(CVAR_PATTERN, CName):
            self._LoggerError(ST.ERR_DECPARSE_PCD_CVAR_PCDCNAME)
        self._CheckReDefine(Guid + DT.TAB_SPLIT + CName)
        Data = DT.TAB_VALUE_SPLIT.join(TokenList[1:-2]).strip()
        DataType = TokenList[-2].strip()
        (Valid, Cause) = IsValidPcdDatum(DataType, Data)
        if not Valid:
            self._LoggerError(Cause)
        PcdType = self._RawData.CurrentScope[0][0]
        if PcdType == DT.TAB_PCDS_FEATURE_FLAG_NULL.upper() and DataType != 'BOOLEAN':
            self._LoggerError(ST.ERR_DECPARSE_PCD_FEATUREFLAG)
        Token = TokenList[-1].strip()
        if not IsValidToken(PCD_TOKEN_PATTERN, Token):
            self._LoggerError(ST.ERR_DECPARSE_PCD_TOKEN % Token)
        elif not Token.startswith('0x') and (not Token.startswith('0X')):
            if int(Token) > 4294967295:
                self._LoggerError(ST.ERR_DECPARSE_PCD_TOKEN_INT % Token)
            Token = '0x%x' % int(Token)
        IntToken = int(Token, 0)
        if (Guid, IntToken) in self.TokenMap:
            if self.TokenMap[Guid, IntToken] != CName:
                self._LoggerError(ST.ERR_DECPARSE_PCD_TOKEN_UNIQUE % Token)
        else:
            self.TokenMap[Guid, IntToken] = CName
        Item = DecPcdItemObject(Guid, CName, Data, DataType, Token)
        self.ItemObject.AddItem(Item, self._RawData.CurrentScope)
        return Item

class _DecGuid(_DecBase):

    def __init__(self, RawData):
        if False:
            for i in range(10):
                print('nop')
        _DecBase.__init__(self, RawData)
        self.GuidObj = DecGuidObject(RawData.Filename)
        self.PpiObj = DecPpiObject(RawData.Filename)
        self.ProtocolObj = DecProtocolObject(RawData.Filename)
        self.ObjectDict = {DT.TAB_GUIDS.upper(): self.GuidObj, DT.TAB_PPIS.upper(): self.PpiObj, DT.TAB_PROTOCOLS.upper(): self.ProtocolObj}

    def GetDataObject(self):
        if False:
            return 10
        if self._RawData.CurrentScope:
            return self.ObjectDict[self._RawData.CurrentScope[0][0]]
        return None

    def GetGuidObject(self):
        if False:
            return 10
        return self.GuidObj

    def GetPpiObject(self):
        if False:
            for i in range(10):
                print('nop')
        return self.PpiObj

    def GetProtocolObject(self):
        if False:
            while True:
                i = 10
        return self.ProtocolObj

    def _ParseItem(self):
        if False:
            return 10
        Line = self._RawData.CurrentLine
        TokenList = GetSplitValueList(Line, DT.TAB_EQUAL_SPLIT, 1)
        if len(TokenList) < 2:
            self._LoggerError(ST.ERR_DECPARSE_CGUID)
        if TokenList[0] == '':
            self._LoggerError(ST.ERR_DECPARSE_CGUID_NAME)
        if TokenList[1] == '':
            self._LoggerError(ST.ERR_DECPARSE_CGUID_GUID)
        if not IsValidToken(CVAR_PATTERN, TokenList[0]):
            self._LoggerError(ST.ERR_DECPARSE_PCD_CVAR_GUID)
        self._CheckReDefine(TokenList[0])
        if TokenList[1][0] != '{':
            if not CheckGuidRegFormat(TokenList[1]):
                self._LoggerError(ST.ERR_DECPARSE_DEFINE_PKGGUID)
            GuidString = TokenList[1]
        else:
            GuidString = GuidStructureStringToGuidString(TokenList[1])
            if TokenList[1][0] != '{' or TokenList[1][-1] != '}' or GuidString == '':
                self._LoggerError(ST.ERR_DECPARSE_CGUID_GUIDFORMAT)
            if not IsValidCFormatGuid(TokenList[1]):
                self._LoggerError(ST.ERR_DECPARSE_CGUID_GUIDFORMAT)
        Item = DecGuidItemObject(TokenList[0], TokenList[1], GuidString)
        ItemObject = self.ObjectDict[self._RawData.CurrentScope[0][0]]
        ItemObject.AddItem(Item, self._RawData.CurrentScope)
        return Item

class _DecUserExtension(_DecBase):

    def __init__(self, RawData):
        if False:
            for i in range(10):
                print('nop')
        _DecBase.__init__(self, RawData)
        self.ItemObject = DecUserExtensionObject(RawData.Filename)
        self._Headers = []
        self._CurItems = []

    def BlockStart(self):
        if False:
            i = 10
            return i + 15
        self._CurItems = []
        for Header in self._RawData.CurrentScope:
            if Header in self._Headers:
                self._LoggerError(ST.ERR_DECPARSE_UE_DUPLICATE)
            else:
                self._Headers.append(Header)
            for Item in self._CurItems:
                if Item.UserId == Header[1] and Item.IdString == Header[2]:
                    Item.ArchAndModuleType.append(Header[3])
                    break
            else:
                Item = DecUserExtensionItemObject()
                Item.UserId = Header[1]
                Item.IdString = Header[2]
                Item.ArchAndModuleType.append(Header[3])
                self._CurItems.append(Item)
                self.ItemObject.AddItem(Item, None)
        self._LocalMacro = {}

    def _ParseItem(self):
        if False:
            while True:
                i = 10
        Line = self._RawData.CurrentLine
        Item = None
        for Item in self._CurItems:
            if Item.UserString:
                Item.UserString = '\n'.join([Item.UserString, Line])
            else:
                Item.UserString = Line
        return Item

class Dec(_DecBase, _DecComments):

    def __init__(self, DecFile, Parse=True):
        if False:
            i = 10
            return i + 15
        try:
            Content = ConvertSpecialChar(open(DecFile, 'r').readlines())
        except BaseException:
            Logger.Error(TOOL_NAME, FILE_OPEN_FAILURE, File=DecFile, ExtraData=ST.ERR_DECPARSE_FILEOPEN % DecFile)
        self._Private = ''
        __IsFoundPrivate = False
        NewContent = []
        for Line in Content:
            Line = Line.strip()
            if Line.startswith(DT.TAB_SECTION_START) and Line.endswith(DT.TAB_PRIVATE + DT.TAB_SECTION_END):
                __IsFoundPrivate = True
            if Line.startswith(DT.TAB_SECTION_START) and Line.endswith(DT.TAB_SECTION_END) and (not Line.endswith(DT.TAB_PRIVATE + DT.TAB_SECTION_END)):
                __IsFoundPrivate = False
            if __IsFoundPrivate:
                self._Private += Line + '\r'
            if not __IsFoundPrivate:
                NewContent.append(Line + '\r')
        RawData = FileContent(DecFile, NewContent)
        _DecComments.__init__(self)
        _DecBase.__init__(self, RawData)
        self.BinaryHeadComment = []
        self.PcdErrorCommentDict = {}
        self._Define = _DecDefine(RawData)
        self._Include = _DecInclude(RawData)
        self._Guid = _DecGuid(RawData)
        self._LibClass = _DecLibraryclass(RawData)
        self._Pcd = _DecPcd(RawData)
        self._UserEx = _DecUserExtension(RawData)
        self._SectionParser = {DT.TAB_DEC_DEFINES.upper(): self._Define, DT.TAB_INCLUDES.upper(): self._Include, DT.TAB_LIBRARY_CLASSES.upper(): self._LibClass, DT.TAB_GUIDS.upper(): self._Guid, DT.TAB_PPIS.upper(): self._Guid, DT.TAB_PROTOCOLS.upper(): self._Guid, DT.TAB_PCDS_FIXED_AT_BUILD_NULL.upper(): self._Pcd, DT.TAB_PCDS_PATCHABLE_IN_MODULE_NULL.upper(): self._Pcd, DT.TAB_PCDS_FEATURE_FLAG_NULL.upper(): self._Pcd, DT.TAB_PCDS_DYNAMIC_NULL.upper(): self._Pcd, DT.TAB_PCDS_DYNAMIC_EX_NULL.upper(): self._Pcd, DT.TAB_USER_EXTENSIONS.upper(): self._UserEx}
        if Parse:
            self.ParseDecComment()
            self.Parse()
            self.CheckRequiredFields()

    def CheckRequiredFields(self):
        if False:
            for i in range(10):
                print('nop')
        for SectionParser in self._SectionParser.values():
            if not SectionParser.CheckRequiredFields():
                return False
        return True

    def ParseDecComment(self):
        if False:
            while True:
                i = 10
        IsFileHeader = False
        IsBinaryHeader = False
        FileHeaderLineIndex = -1
        BinaryHeaderLineIndex = -1
        TokenSpaceGuidCName = ''
        while not self._RawData.IsEndOfFile():
            self._RawData.CurrentLine = self._RawData.GetNextLine()
            if self._RawData.CurrentLine.startswith(DT.TAB_COMMENT_SPLIT) and DT.TAB_SECTION_START in self._RawData.CurrentLine and (DT.TAB_SECTION_END in self._RawData.CurrentLine):
                self._RawData.CurrentLine = self._RawData.CurrentLine.replace(DT.TAB_COMMENT_SPLIT, '').strip()
                if self._RawData.CurrentLine[0] == DT.TAB_SECTION_START and self._RawData.CurrentLine[-1] == DT.TAB_SECTION_END:
                    RawSection = self._RawData.CurrentLine[1:-1].strip()
                    if RawSection.upper().startswith(DT.TAB_PCD_ERROR.upper() + '.'):
                        TokenSpaceGuidCName = RawSection.split(DT.TAB_PCD_ERROR + '.')[1].strip()
                        continue
            if TokenSpaceGuidCName and self._RawData.CurrentLine.startswith(DT.TAB_COMMENT_SPLIT):
                self._RawData.CurrentLine = self._RawData.CurrentLine.replace(DT.TAB_COMMENT_SPLIT, '').strip()
                if self._RawData.CurrentLine != '':
                    if DT.TAB_VALUE_SPLIT not in self._RawData.CurrentLine:
                        self._LoggerError(ST.ERR_DECPARSE_PCDERRORMSG_MISS_VALUE_SPLIT)
                    (PcdErrorNumber, PcdErrorMsg) = GetSplitValueList(self._RawData.CurrentLine, DT.TAB_VALUE_SPLIT, 1)
                    PcdErrorNumber = ParsePcdErrorCode(PcdErrorNumber, self._RawData.Filename, self._RawData.LineIndex)
                    if not PcdErrorMsg.strip():
                        self._LoggerError(ST.ERR_DECPARSE_PCD_MISS_ERRORMSG)
                    self.PcdErrorCommentDict[TokenSpaceGuidCName, PcdErrorNumber] = PcdErrorMsg.strip()
            else:
                TokenSpaceGuidCName = ''
        self._RawData.LineIndex = 0
        self._RawData.CurrentLine = ''
        self._RawData.NextLine = ''
        while not self._RawData.IsEndOfFile():
            (Line, Comment) = CleanString(self._RawData.GetNextLine())
            if Line != '':
                self._RawData.UndoNextLine()
                break
            if Comment and Comment.startswith(DT.TAB_SPECIAL_COMMENT) and (Comment.find(DT.TAB_HEADER_COMMENT) > 0) and (not Comment[2:Comment.find(DT.TAB_HEADER_COMMENT)].strip()):
                IsFileHeader = True
                IsBinaryHeader = False
                FileHeaderLineIndex = self._RawData.LineIndex
            if not IsFileHeader and (not IsBinaryHeader) and Comment and Comment.startswith(DT.TAB_COMMENT_SPLIT) and (DT.TAB_BINARY_HEADER_COMMENT not in Comment):
                self._HeadComment.append((Comment, self._RawData.LineIndex))
            if Comment and IsFileHeader and (not (Comment.startswith(DT.TAB_SPECIAL_COMMENT) and Comment.find(DT.TAB_BINARY_HEADER_COMMENT) > 0)):
                self._HeadComment.append((Comment, self._RawData.LineIndex))
            if (not Comment or Comment == DT.TAB_SPECIAL_COMMENT) and IsFileHeader:
                IsFileHeader = False
                continue
            if Comment and Comment.startswith(DT.TAB_SPECIAL_COMMENT) and (Comment.find(DT.TAB_BINARY_HEADER_COMMENT) > 0):
                IsBinaryHeader = True
                IsFileHeader = False
                BinaryHeaderLineIndex = self._RawData.LineIndex
            if Comment and IsBinaryHeader:
                self.BinaryHeadComment.append((Comment, self._RawData.LineIndex))
            if (not Comment or Comment == DT.TAB_SPECIAL_COMMENT) and IsBinaryHeader:
                IsBinaryHeader = False
                break
            if FileHeaderLineIndex > -1 and (not IsFileHeader) and (not IsBinaryHeader):
                break
        if FileHeaderLineIndex > BinaryHeaderLineIndex and FileHeaderLineIndex > -1 and (BinaryHeaderLineIndex > -1):
            self._LoggerError(ST.ERR_BINARY_HEADER_ORDER)
        if FileHeaderLineIndex == -1:
            Logger.Error(TOOL_NAME, FORMAT_INVALID, ST.ERR_NO_SOURCE_HEADER, File=self._RawData.Filename)
        return

    def _StopCurrentParsing(self, Line):
        if False:
            print('Hello World!')
        return False

    def _ParseItem(self):
        if False:
            return 10
        self._SectionHeaderParser()
        if len(self._RawData.CurrentScope) == 0:
            self._LoggerError(ST.ERR_DECPARSE_SECTION_EMPTY)
        SectionObj = self._SectionParser[self._RawData.CurrentScope[0][0]]
        SectionObj.BlockStart()
        SectionObj.Parse()
        return SectionObj.GetDataObject()

    def _UserExtentionSectionParser(self):
        if False:
            i = 10
            return i + 15
        self._RawData.CurrentScope = []
        ArchList = set()
        Section = self._RawData.CurrentLine[1:-1]
        Par = ParserHelper(Section, self._RawData.Filename)
        while not Par.End():
            Token = Par.GetToken()
            if Token.upper() != DT.TAB_USER_EXTENSIONS.upper():
                self._LoggerError(ST.ERR_DECPARSE_SECTION_UE)
            UserExtension = Token.upper()
            Par.AssertChar(DT.TAB_SPLIT, ST.ERR_DECPARSE_SECTION_UE, self._RawData.LineIndex)
            Token = Par.GetToken()
            if not IsValidUserId(Token):
                self._LoggerError(ST.ERR_DECPARSE_SECTION_UE_USERID)
            UserId = Token
            Par.AssertChar(DT.TAB_SPLIT, ST.ERR_DECPARSE_SECTION_UE, self._RawData.LineIndex)
            Token = Par.GetToken()
            if not IsValidIdString(Token):
                self._LoggerError(ST.ERR_DECPARSE_SECTION_UE_IDSTRING)
            IdString = Token
            Arch = 'COMMON'
            if Par.Expect(DT.TAB_SPLIT):
                Token = Par.GetToken()
                Arch = Token.upper()
                if not IsValidArch(Arch):
                    self._LoggerError(ST.ERR_DECPARSE_ARCH)
            ArchList.add(Arch)
            if [UserExtension, UserId, IdString, Arch] not in self._RawData.CurrentScope:
                self._RawData.CurrentScope.append([UserExtension, UserId, IdString, Arch])
            if not Par.Expect(DT.TAB_COMMA_SPLIT):
                break
            elif Par.End():
                self._LoggerError(ST.ERR_DECPARSE_SECTION_COMMA)
        Par.AssertEnd(ST.ERR_DECPARSE_SECTION_UE, self._RawData.LineIndex)
        if 'COMMON' in ArchList and len(ArchList) > 1:
            self._LoggerError(ST.ERR_DECPARSE_SECTION_COMMON)

    def _SectionHeaderParser(self):
        if False:
            i = 10
            return i + 15
        if self._RawData.CurrentLine[0] != DT.TAB_SECTION_START or self._RawData.CurrentLine[-1] != DT.TAB_SECTION_END:
            self._LoggerError(ST.ERR_DECPARSE_SECTION_IDENTIFY)
        RawSection = self._RawData.CurrentLine[1:-1].strip().upper()
        if RawSection.startswith(DT.TAB_DEC_DEFINES.upper()):
            if RawSection != DT.TAB_DEC_DEFINES.upper():
                self._LoggerError(ST.ERR_DECPARSE_DEFINE_SECNAME)
        if RawSection.startswith(DT.TAB_USER_EXTENSIONS.upper()):
            return self._UserExtentionSectionParser()
        self._RawData.CurrentScope = []
        SectionNames = []
        ArchList = set()
        for Item in GetSplitValueList(RawSection, DT.TAB_COMMA_SPLIT):
            if Item == '':
                self._LoggerError(ST.ERR_DECPARSE_SECTION_SUBEMPTY % self._RawData.CurrentLine)
            ItemList = GetSplitValueList(Item, DT.TAB_SPLIT)
            SectionName = ItemList[0]
            if SectionName not in self._SectionParser:
                self._LoggerError(ST.ERR_DECPARSE_SECTION_UNKNOW % SectionName)
            if SectionName not in SectionNames:
                SectionNames.append(SectionName)
            if len(ItemList) > 2:
                self._LoggerError(ST.ERR_DECPARSE_SECTION_SUBTOOMANY % Item)
            if DT.TAB_PCDS_FEATURE_FLAG_NULL.upper() in SectionNames and len(SectionNames) > 1:
                self._LoggerError(ST.ERR_DECPARSE_SECTION_FEATUREFLAG % DT.TAB_PCDS_FEATURE_FLAG_NULL)
            if len(ItemList) > 1:
                Str1 = ItemList[1]
                if not IsValidArch(Str1):
                    self._LoggerError(ST.ERR_DECPARSE_ARCH)
            else:
                Str1 = 'COMMON'
            ArchList.add(Str1)
            if [SectionName, Str1] not in self._RawData.CurrentScope:
                self._RawData.CurrentScope.append([SectionName, Str1])
        if 'COMMON' in ArchList and len(ArchList) > 1:
            self._LoggerError(ST.ERR_DECPARSE_SECTION_COMMON)
        if len(SectionNames) == 0:
            self._LoggerError(ST.ERR_DECPARSE_SECTION_SUBEMPTY % self._RawData.CurrentLine)
        if len(SectionNames) != 1:
            for Sec in SectionNames:
                if not Sec.startswith(DT.TAB_PCDS.upper()):
                    self._LoggerError(ST.ERR_DECPARSE_SECTION_NAME % str(SectionNames))

    def GetDefineSectionMacro(self):
        if False:
            while True:
                i = 10
        return self._Define.GetLocalMacro()

    def GetDefineSectionObject(self):
        if False:
            return 10
        return self._Define.GetDataObject()

    def GetIncludeSectionObject(self):
        if False:
            i = 10
            return i + 15
        return self._Include.GetDataObject()

    def GetGuidSectionObject(self):
        if False:
            while True:
                i = 10
        return self._Guid.GetGuidObject()

    def GetProtocolSectionObject(self):
        if False:
            i = 10
            return i + 15
        return self._Guid.GetProtocolObject()

    def GetPpiSectionObject(self):
        if False:
            while True:
                i = 10
        return self._Guid.GetPpiObject()

    def GetLibraryClassSectionObject(self):
        if False:
            print('Hello World!')
        return self._LibClass.GetDataObject()

    def GetPcdSectionObject(self):
        if False:
            return 10
        return self._Pcd.GetDataObject()

    def GetUserExtensionSectionObject(self):
        if False:
            print('Hello World!')
        return self._UserEx.GetDataObject()

    def GetPackageSpecification(self):
        if False:
            i = 10
            return i + 15
        return self._Define.GetDataObject().GetPackageSpecification()

    def GetPackageName(self):
        if False:
            i = 10
            return i + 15
        return self._Define.GetDataObject().GetPackageName()

    def GetPackageGuid(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Define.GetDataObject().GetPackageGuid()

    def GetPackageVersion(self):
        if False:
            return 10
        return self._Define.GetDataObject().GetPackageVersion()

    def GetPackageUniFile(self):
        if False:
            print('Hello World!')
        return self._Define.GetDataObject().GetPackageUniFile()

    def GetPrivateSections(self):
        if False:
            while True:
                i = 10
        return self._Private