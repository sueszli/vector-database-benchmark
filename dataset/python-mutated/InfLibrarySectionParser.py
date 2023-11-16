"""
InfLibrarySectionParser
"""
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger.ToolError import FORMAT_INVALID
from Parser.InfParserMisc import InfExpandMacro
from Library import DataType as DT
from Library.Parsing import MacroParser
from Library.Misc import GetSplitValueList
from Object.Parser.InfCommonObject import InfLineCommentObject
from Library import GlobalData
from Parser.InfParserMisc import IsLibInstanceInfo
from Parser.InfAsBuiltProcess import GetLibInstanceInfo
from Parser.InfParserMisc import InfParserSectionRoot

class InfLibrarySectionParser(InfParserSectionRoot):

    def InfLibraryParser(self, SectionString, InfSectionObject, FileName):
        if False:
            while True:
                i = 10
        if not GlobalData.gIS_BINARY_INF:
            SectionMacros = {}
            ValueList = []
            LibraryList = []
            LibStillCommentFalg = False
            LibHeaderComments = []
            LibLineComment = None
            for Line in SectionString:
                LibLineContent = Line[0]
                LibLineNo = Line[1]
                if LibLineContent.strip() == '':
                    continue
                if LibLineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                    if LibStillCommentFalg:
                        LibHeaderComments.append(Line)
                        continue
                    else:
                        LibHeaderComments = []
                        LibHeaderComments.append(Line)
                        LibStillCommentFalg = True
                        continue
                else:
                    LibStillCommentFalg = False
                if len(LibHeaderComments) >= 1:
                    LibLineComment = InfLineCommentObject()
                    LineCommentContent = ''
                    for Item in LibHeaderComments:
                        LineCommentContent += Item[0] + DT.END_OF_LINE
                    LibLineComment.SetHeaderComments(LineCommentContent)
                if LibLineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                    LibTailComments = LibLineContent[LibLineContent.find(DT.TAB_COMMENT_SPLIT):]
                    LibLineContent = LibLineContent[:LibLineContent.find(DT.TAB_COMMENT_SPLIT)]
                    if LibLineComment is None:
                        LibLineComment = InfLineCommentObject()
                    LibLineComment.SetTailComments(LibTailComments)
                (Name, Value) = MacroParser((LibLineContent, LibLineNo), FileName, DT.MODEL_EFI_LIBRARY_CLASS, self.FileLocalMacros)
                if Name is not None:
                    SectionMacros[Name] = Value
                    LibLineComment = None
                    LibHeaderComments = []
                    continue
                TokenList = GetSplitValueList(LibLineContent, DT.TAB_VALUE_SPLIT, 1)
                ValueList[0:len(TokenList)] = TokenList
                ValueList = [InfExpandMacro(Value, (FileName, LibLineContent, LibLineNo), self.FileLocalMacros, SectionMacros, True) for Value in ValueList]
                LibraryList.append((ValueList, LibLineComment, (LibLineContent, LibLineNo, FileName)))
                ValueList = []
                LibLineComment = None
                LibTailComments = ''
                LibHeaderComments = []
                continue
            KeyList = []
            for Item in self.LastSectionHeaderContent:
                if (Item[1], Item[2]) not in KeyList:
                    KeyList.append((Item[1], Item[2]))
            if not InfSectionObject.SetLibraryClasses(LibraryList, KeyList=KeyList):
                Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[Library]', File=FileName, Line=Item[3])
        else:
            self.InfAsBuiltLibraryParser(SectionString, InfSectionObject, FileName)

    def InfAsBuiltLibraryParser(self, SectionString, InfSectionObject, FileName):
        if False:
            while True:
                i = 10
        LibraryList = []
        LibInsFlag = False
        for Line in SectionString:
            LineContent = Line[0]
            LineNo = Line[1]
            if LineContent.strip() == '':
                LibInsFlag = False
                continue
            if not LineContent.strip().startswith('#'):
                Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_LIB_CONTATIN_ASBUILD_AND_COMMON, File=FileName, Line=LineNo, ExtraData=LineContent)
            if IsLibInstanceInfo(LineContent):
                LibInsFlag = True
                continue
            if LibInsFlag:
                (LibGuid, LibVer) = GetLibInstanceInfo(LineContent, GlobalData.gWORKSPACE, LineNo, FileName)
                if LibVer == '':
                    LibVer = '0'
                if LibGuid != '':
                    if (LibGuid, LibVer) not in LibraryList:
                        LibraryList.append((LibGuid, LibVer))
        KeyList = []
        Item = ['', '', '']
        for Item in self.LastSectionHeaderContent:
            if (Item[1], Item[2]) not in KeyList:
                KeyList.append((Item[1], Item[2]))
        if not InfSectionObject.SetLibraryClasses(LibraryList, KeyList=KeyList):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[Library]', File=FileName, Line=Item[3])