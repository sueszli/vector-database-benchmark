"""
InfGuidPpiProtocolSectionParser
"""
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger.ToolError import FORMAT_INVALID
from Parser.InfParserMisc import InfExpandMacro
from Library import DataType as DT
from Library import GlobalData
from Library.Parsing import MacroParser
from Library.Misc import GetSplitValueList
from Library.ParserValidate import IsValidIdString
from Library.ParserValidate import IsValidUserId
from Library.ParserValidate import IsValidArch
from Parser.InfParserMisc import InfParserSectionRoot

class InfGuidPpiProtocolSectionParser(InfParserSectionRoot):

    def InfGuidParser(self, SectionString, InfSectionObject, FileName):
        if False:
            return 10
        SectionMacros = {}
        ValueList = []
        GuidList = []
        CommentsList = []
        CurrentLineVar = None
        for Line in SectionString:
            LineContent = Line[0]
            LineNo = Line[1]
            if LineContent.strip() == '':
                CommentsList = []
                continue
            if LineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                CommentsList.append(Line)
                continue
            elif LineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                CommentsList.append((LineContent[LineContent.find(DT.TAB_COMMENT_SPLIT):], LineNo))
                LineContent = LineContent[:LineContent.find(DT.TAB_COMMENT_SPLIT)]
            if LineContent != '':
                (Name, Value) = MacroParser((LineContent, LineNo), FileName, DT.MODEL_EFI_GUID, self.FileLocalMacros)
                if Name is not None:
                    SectionMacros[Name] = Value
                    CommentsList = []
                    ValueList = []
                    continue
                TokenList = GetSplitValueList(LineContent, DT.TAB_VALUE_SPLIT, 1)
                ValueList[0:len(TokenList)] = TokenList
                ValueList = [InfExpandMacro(Value, (FileName, LineContent, LineNo), self.FileLocalMacros, SectionMacros, True) for Value in ValueList]
                CurrentLineVar = (LineContent, LineNo, FileName)
            if len(ValueList) >= 1:
                GuidList.append((ValueList, CommentsList, CurrentLineVar))
                CommentsList = []
                ValueList = []
            continue
        ArchList = []
        LineIndex = -1
        for Item in self.LastSectionHeaderContent:
            LineIndex = Item[3]
            if Item[1] not in ArchList:
                ArchList.append(Item[1])
        if not InfSectionObject.SetGuid(GuidList, Arch=ArchList):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[Guid]', File=FileName, Line=LineIndex)

    def InfPpiParser(self, SectionString, InfSectionObject, FileName):
        if False:
            return 10
        SectionMacros = {}
        ValueList = []
        PpiList = []
        CommentsList = []
        CurrentLineVar = None
        for Line in SectionString:
            LineContent = Line[0]
            LineNo = Line[1]
            if LineContent.strip() == '':
                CommentsList = []
                continue
            if LineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                CommentsList.append(Line)
                continue
            elif LineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                CommentsList.append((LineContent[LineContent.find(DT.TAB_COMMENT_SPLIT):], LineNo))
                LineContent = LineContent[:LineContent.find(DT.TAB_COMMENT_SPLIT)]
            if LineContent != '':
                (Name, Value) = MacroParser((LineContent, LineNo), FileName, DT.MODEL_EFI_PPI, self.FileLocalMacros)
                if Name is not None:
                    SectionMacros[Name] = Value
                    ValueList = []
                    CommentsList = []
                    continue
                TokenList = GetSplitValueList(LineContent, DT.TAB_VALUE_SPLIT, 1)
                ValueList[0:len(TokenList)] = TokenList
                ValueList = [InfExpandMacro(Value, (FileName, LineContent, LineNo), self.FileLocalMacros, SectionMacros) for Value in ValueList]
                CurrentLineVar = (LineContent, LineNo, FileName)
            if len(ValueList) >= 1:
                PpiList.append((ValueList, CommentsList, CurrentLineVar))
                ValueList = []
                CommentsList = []
            continue
        ArchList = []
        LineIndex = -1
        for Item in self.LastSectionHeaderContent:
            LineIndex = Item[3]
            if Item[1] not in ArchList:
                ArchList.append(Item[1])
        if not InfSectionObject.SetPpi(PpiList, Arch=ArchList):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[Ppis]', File=FileName, Line=LineIndex)

    def InfUserExtensionParser(self, SectionString, InfSectionObject, FileName):
        if False:
            for i in range(10):
                print('nop')
        UserExtensionContent = ''
        for Line in SectionString:
            LineContent = Line[0]
            UserExtensionContent += LineContent + DT.END_OF_LINE
            continue
        IdContentList = []
        LastItem = ''
        SectionLineNo = None
        for Item in self.LastSectionHeaderContent:
            UserId = Item[1]
            IdString = Item[2]
            Arch = Item[3]
            SectionLineNo = Item[4]
            if not IsValidArch(Arch):
                Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % Arch, File=GlobalData.gINF_MODULE_NAME, Line=SectionLineNo, ExtraData=None)
            if (UserId, IdString, Arch) not in IdContentList:
                if not IsValidUserId(UserId):
                    Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_UE_SECTION_USER_ID_ERROR % Item[1], File=GlobalData.gINF_MODULE_NAME, Line=SectionLineNo, ExtraData=None)
                if not IsValidIdString(IdString):
                    Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_UE_SECTION_ID_STRING_ERROR % IdString, File=GlobalData.gINF_MODULE_NAME, Line=SectionLineNo, ExtraData=None)
                IdContentList.append((UserId, IdString, Arch))
            else:
                Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_UE_SECTION_DUPLICATE_ERROR % IdString, File=GlobalData.gINF_MODULE_NAME, Line=SectionLineNo, ExtraData=None)
            LastItem = Item
        if not InfSectionObject.SetUserExtension(UserExtensionContent, IdContent=IdContentList, LineNo=SectionLineNo):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[UserExtension]', File=FileName, Line=LastItem[4])

    def InfProtocolParser(self, SectionString, InfSectionObject, FileName):
        if False:
            while True:
                i = 10
        SectionMacros = {}
        ValueList = []
        ProtocolList = []
        CommentsList = []
        CurrentLineVar = None
        for Line in SectionString:
            LineContent = Line[0]
            LineNo = Line[1]
            if LineContent.strip() == '':
                CommentsList = []
                continue
            if LineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                CommentsList.append(Line)
                continue
            elif LineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                CommentsList.append((LineContent[LineContent.find(DT.TAB_COMMENT_SPLIT):], LineNo))
                LineContent = LineContent[:LineContent.find(DT.TAB_COMMENT_SPLIT)]
            if LineContent != '':
                (Name, Value) = MacroParser((LineContent, LineNo), FileName, DT.MODEL_EFI_PROTOCOL, self.FileLocalMacros)
                if Name is not None:
                    SectionMacros[Name] = Value
                    ValueList = []
                    CommentsList = []
                    continue
                TokenList = GetSplitValueList(LineContent, DT.TAB_VALUE_SPLIT, 1)
                ValueList[0:len(TokenList)] = TokenList
                ValueList = [InfExpandMacro(Value, (FileName, LineContent, LineNo), self.FileLocalMacros, SectionMacros) for Value in ValueList]
                CurrentLineVar = (LineContent, LineNo, FileName)
            if len(ValueList) >= 1:
                ProtocolList.append((ValueList, CommentsList, CurrentLineVar))
                ValueList = []
                CommentsList = []
            continue
        ArchList = []
        LineIndex = -1
        for Item in self.LastSectionHeaderContent:
            LineIndex = Item[3]
            if Item[1] not in ArchList:
                ArchList.append(Item[1])
        if not InfSectionObject.SetProtocol(ProtocolList, Arch=ArchList):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[Protocol]', File=FileName, Line=LineIndex)