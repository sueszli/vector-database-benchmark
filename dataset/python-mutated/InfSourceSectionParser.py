"""
InfSourceSectionParser
"""
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger.ToolError import FORMAT_INVALID
from Parser.InfParserMisc import InfExpandMacro
from Library import DataType as DT
from Library.Parsing import MacroParser
from Library.Misc import GetSplitValueList
from Object.Parser.InfCommonObject import InfLineCommentObject
from Parser.InfParserMisc import InfParserSectionRoot

class InfSourceSectionParser(InfParserSectionRoot):

    def InfSourceParser(self, SectionString, InfSectionObject, FileName):
        if False:
            return 10
        SectionMacros = {}
        ValueList = []
        SourceList = []
        StillCommentFalg = False
        HeaderComments = []
        LineComment = None
        SectionContent = ''
        for Line in SectionString:
            SrcLineContent = Line[0]
            SrcLineNo = Line[1]
            if SrcLineContent.strip() == '':
                continue
            if SrcLineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                if StillCommentFalg:
                    HeaderComments.append(Line)
                    SectionContent += SrcLineContent + DT.END_OF_LINE
                    continue
                else:
                    HeaderComments = []
                    HeaderComments.append(Line)
                    StillCommentFalg = True
                    SectionContent += SrcLineContent + DT.END_OF_LINE
                    continue
            else:
                StillCommentFalg = False
            if len(HeaderComments) >= 1:
                LineComment = InfLineCommentObject()
                LineCommentContent = ''
                for Item in HeaderComments:
                    LineCommentContent += Item[0] + DT.END_OF_LINE
                LineComment.SetHeaderComments(LineCommentContent)
            if SrcLineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                TailComments = SrcLineContent[SrcLineContent.find(DT.TAB_COMMENT_SPLIT):]
                SrcLineContent = SrcLineContent[:SrcLineContent.find(DT.TAB_COMMENT_SPLIT)]
                if LineComment is None:
                    LineComment = InfLineCommentObject()
                LineComment.SetTailComments(TailComments)
            (Name, Value) = MacroParser((SrcLineContent, SrcLineNo), FileName, DT.MODEL_EFI_SOURCE_FILE, self.FileLocalMacros)
            if Name is not None:
                SectionMacros[Name] = Value
                LineComment = None
                HeaderComments = []
                continue
            SrcLineContent = InfExpandMacro(SrcLineContent, (FileName, SrcLineContent, SrcLineNo), self.FileLocalMacros, SectionMacros)
            TokenList = GetSplitValueList(SrcLineContent, DT.TAB_VALUE_SPLIT, 4)
            ValueList[0:len(TokenList)] = TokenList
            SectionContent += SrcLineContent + DT.END_OF_LINE
            SourceList.append((ValueList, LineComment, (SrcLineContent, SrcLineNo, FileName)))
            ValueList = []
            LineComment = None
            TailComments = ''
            HeaderComments = []
            continue
        ArchList = []
        for Item in self.LastSectionHeaderContent:
            if Item[1] not in ArchList:
                ArchList.append(Item[1])
                InfSectionObject.SetSupArchList(Item[1])
        InfSectionObject.SetAllContent(SectionContent)
        if not InfSectionObject.SetSources(SourceList, Arch=ArchList):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[Sources]', File=FileName, Line=Item[3])