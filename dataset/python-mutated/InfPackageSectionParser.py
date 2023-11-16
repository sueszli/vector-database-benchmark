"""
InfPackageSectionParser
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

class InfPackageSectionParser(InfParserSectionRoot):

    def InfPackageParser(self, SectionString, InfSectionObject, FileName):
        if False:
            i = 10
            return i + 15
        SectionMacros = {}
        ValueList = []
        PackageList = []
        StillCommentFalg = False
        HeaderComments = []
        LineComment = None
        for Line in SectionString:
            PkgLineContent = Line[0]
            PkgLineNo = Line[1]
            if PkgLineContent.strip() == '':
                continue
            if PkgLineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                if StillCommentFalg:
                    HeaderComments.append(Line)
                    continue
                else:
                    HeaderComments = []
                    HeaderComments.append(Line)
                    StillCommentFalg = True
                    continue
            else:
                StillCommentFalg = False
            if len(HeaderComments) >= 1:
                LineComment = InfLineCommentObject()
                LineCommentContent = ''
                for Item in HeaderComments:
                    LineCommentContent += Item[0] + DT.END_OF_LINE
                LineComment.SetHeaderComments(LineCommentContent)
            if PkgLineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                TailComments = PkgLineContent[PkgLineContent.find(DT.TAB_COMMENT_SPLIT):]
                PkgLineContent = PkgLineContent[:PkgLineContent.find(DT.TAB_COMMENT_SPLIT)]
                if LineComment is None:
                    LineComment = InfLineCommentObject()
                LineComment.SetTailComments(TailComments)
            (Name, Value) = MacroParser((PkgLineContent, PkgLineNo), FileName, DT.MODEL_META_DATA_PACKAGE, self.FileLocalMacros)
            if Name is not None:
                SectionMacros[Name] = Value
                LineComment = None
                HeaderComments = []
                continue
            TokenList = GetSplitValueList(PkgLineContent, DT.TAB_VALUE_SPLIT, 1)
            ValueList[0:len(TokenList)] = TokenList
            ValueList = [InfExpandMacro(Value, (FileName, PkgLineContent, PkgLineNo), self.FileLocalMacros, SectionMacros, True) for Value in ValueList]
            PackageList.append((ValueList, LineComment, (PkgLineContent, PkgLineNo, FileName)))
            ValueList = []
            LineComment = None
            TailComments = ''
            HeaderComments = []
            continue
        ArchList = []
        for Item in self.LastSectionHeaderContent:
            if Item[1] not in ArchList:
                ArchList.append(Item[1])
        if not InfSectionObject.SetPackages(PackageList, Arch=ArchList):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[Packages]', File=FileName, Line=Item[3])