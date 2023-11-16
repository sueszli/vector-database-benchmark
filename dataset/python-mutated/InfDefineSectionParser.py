"""
InfDefineSectionParser
"""
import re
from Library import DataType as DT
from Library import GlobalData
from Library.Parsing import MacroParser
from Library.Misc import GetSplitValueList
from Library.ParserValidate import IsValidArch
from Object.Parser.InfCommonObject import InfLineCommentObject
from Object.Parser.InfDefineObject import InfDefMember
from Parser.InfParserMisc import InfExpandMacro
from Object.Parser.InfMisc import ErrorInInf
from Logger import StringTable as ST
from Parser.InfParserMisc import InfParserSectionRoot

def GetValidateArchList(LineContent):
    if False:
        i = 10
        return i + 15
    TempArch = ''
    ArchList = []
    ValidateAcrhPatten = re.compile('^\\s*#\\s*VALID_ARCHITECTURES\\s*=\\s*.*$', re.DOTALL)
    if ValidateAcrhPatten.match(LineContent):
        TempArch = GetSplitValueList(LineContent, DT.TAB_EQUAL_SPLIT, 1)[1]
        TempArch = GetSplitValueList(TempArch, '(', 1)[0]
        ArchList = re.split('\\s+', TempArch)
        NewArchList = []
        for Arch in ArchList:
            if IsValidArch(Arch):
                NewArchList.append(Arch)
        ArchList = NewArchList
    return ArchList

class InfDefinSectionParser(InfParserSectionRoot):

    def InfDefineParser(self, SectionString, InfSectionObject, FileName, SectionComment):
        if False:
            while True:
                i = 10
        if SectionComment:
            pass
        StillCommentFalg = False
        HeaderComments = []
        SectionContent = ''
        ArchList = []
        _ContentList = []
        _ValueList = []
        self.FileLocalMacros['WORKSPACE'] = GlobalData.gWORKSPACE
        for Line in SectionString:
            LineContent = Line[0]
            LineNo = Line[1]
            TailComments = ''
            LineComment = None
            LineInfo = ['', -1, '']
            LineInfo[0] = FileName
            LineInfo[1] = LineNo
            LineInfo[2] = LineContent
            if LineContent.strip() == '':
                continue
            if not ArchList:
                ArchList = GetValidateArchList(LineContent)
            if LineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                if StillCommentFalg:
                    HeaderComments.append(Line)
                    SectionContent += LineContent + DT.END_OF_LINE
                    continue
                else:
                    HeaderComments = []
                    HeaderComments.append(Line)
                    StillCommentFalg = True
                    SectionContent += LineContent + DT.END_OF_LINE
                    continue
            else:
                StillCommentFalg = False
            if len(HeaderComments) >= 1:
                LineComment = InfLineCommentObject()
                LineCommentContent = ''
                for Item in HeaderComments:
                    LineCommentContent += Item[0] + DT.END_OF_LINE
                LineComment.SetHeaderComments(LineCommentContent)
            if LineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                TailComments = LineContent[LineContent.find(DT.TAB_COMMENT_SPLIT):]
                LineContent = LineContent[:LineContent.find(DT.TAB_COMMENT_SPLIT)]
                if LineComment is None:
                    LineComment = InfLineCommentObject()
                LineComment.SetTailComments(TailComments)
            (Name, Value) = MacroParser((LineContent, LineNo), FileName, DT.MODEL_META_DATA_HEADER, self.FileLocalMacros)
            if Name is not None:
                self.FileLocalMacros[Name] = Value
                continue
            LineContent = InfExpandMacro(LineContent, (FileName, LineContent, LineNo), self.FileLocalMacros, None, True)
            SectionContent += LineContent + DT.END_OF_LINE
            TokenList = GetSplitValueList(LineContent, DT.TAB_EQUAL_SPLIT, 1)
            if len(TokenList) < 2:
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_NO_VALUE, LineInfo=LineInfo)
            _ValueList[0:len(TokenList)] = TokenList
            if not _ValueList[0]:
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_NO_NAME, LineInfo=LineInfo)
            if not _ValueList[1]:
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_NO_VALUE, LineInfo=LineInfo)
            (Name, Value) = (_ValueList[0], _ValueList[1])
            InfDefMemberObj = InfDefMember(Name, Value)
            if LineComment is not None:
                InfDefMemberObj.Comments.SetHeaderComments(LineComment.GetHeaderComments())
                InfDefMemberObj.Comments.SetTailComments(LineComment.GetTailComments())
            InfDefMemberObj.CurrentLine.SetFileName(self.FullPath)
            InfDefMemberObj.CurrentLine.SetLineString(LineContent)
            InfDefMemberObj.CurrentLine.SetLineNo(LineNo)
            _ContentList.append(InfDefMemberObj)
            HeaderComments = []
            TailComments = ''
        if not ArchList:
            ArchList = ['COMMON']
        InfSectionObject.SetAllContent(SectionContent)
        InfSectionObject.SetDefines(_ContentList, Arch=ArchList)