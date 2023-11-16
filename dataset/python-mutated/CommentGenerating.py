"""
CommentGenerating
"""
from Library.StringUtils import GetSplitValueList
from Library.DataType import TAB_SPACE_SPLIT
from Library.DataType import TAB_INF_GUIDTYPE_VAR
from Library.DataType import USAGE_ITEM_NOTIFY
from Library.DataType import ITEM_UNDEFINED
from Library.DataType import TAB_HEADER_COMMENT
from Library.DataType import TAB_BINARY_HEADER_COMMENT
from Library.DataType import TAB_COMMENT_SPLIT
from Library.DataType import TAB_SPECIAL_COMMENT
from Library.DataType import END_OF_LINE
from Library.DataType import TAB_COMMENT_EDK1_SPLIT
from Library.DataType import TAB_COMMENT_EDK1_START
from Library.DataType import TAB_COMMENT_EDK1_END
from Library.DataType import TAB_STAR
from Library.DataType import TAB_PCD_PROMPT
from Library.UniClassObject import ConvertSpecialUnicodes
from Library.Misc import GetLocalValue

def GenTailCommentLines(TailCommentLines, LeadingSpaceNum=0):
    if False:
        i = 10
        return i + 15
    TailCommentLines = TailCommentLines.rstrip(END_OF_LINE)
    CommentStr = TAB_SPACE_SPLIT * 2 + TAB_SPECIAL_COMMENT + TAB_SPACE_SPLIT + (END_OF_LINE + LeadingSpaceNum * TAB_SPACE_SPLIT + TAB_SPACE_SPLIT * 2 + TAB_SPECIAL_COMMENT + TAB_SPACE_SPLIT).join(GetSplitValueList(TailCommentLines, END_OF_LINE))
    return CommentStr

def GenGenericComment(CommentLines):
    if False:
        for i in range(10):
            print('nop')
    if not CommentLines:
        return ''
    CommentLines = CommentLines.rstrip(END_OF_LINE)
    CommentStr = TAB_SPECIAL_COMMENT + TAB_SPACE_SPLIT + (END_OF_LINE + TAB_COMMENT_SPLIT + TAB_SPACE_SPLIT).join(GetSplitValueList(CommentLines, END_OF_LINE)) + END_OF_LINE
    return CommentStr

def GenGenericCommentF(CommentLines, NumOfPound=1, IsPrompt=False, IsInfLibraryClass=False):
    if False:
        return 10
    if not CommentLines:
        return ''
    if CommentLines.endswith(END_OF_LINE):
        CommentLines = CommentLines[:-1]
    CommentStr = ''
    if IsPrompt:
        CommentStr += TAB_COMMENT_SPLIT * NumOfPound + TAB_SPACE_SPLIT + TAB_PCD_PROMPT + TAB_SPACE_SPLIT + CommentLines.replace(END_OF_LINE, '') + END_OF_LINE
    else:
        CommentLineList = GetSplitValueList(CommentLines, END_OF_LINE)
        FindLibraryClass = False
        for Line in CommentLineList:
            if IsInfLibraryClass and Line.find(u'@libraryclass ') > -1:
                FindLibraryClass = True
            if Line == '':
                CommentStr += TAB_COMMENT_SPLIT * NumOfPound + END_OF_LINE
            elif FindLibraryClass and Line.find(u'@libraryclass ') > -1:
                CommentStr += TAB_COMMENT_SPLIT * NumOfPound + TAB_SPACE_SPLIT + Line + END_OF_LINE
            elif FindLibraryClass:
                CommentStr += TAB_COMMENT_SPLIT * NumOfPound + TAB_SPACE_SPLIT * 16 + Line + END_OF_LINE
            else:
                CommentStr += TAB_COMMENT_SPLIT * NumOfPound + TAB_SPACE_SPLIT + Line + END_OF_LINE
    return CommentStr

def GenHeaderCommentSection(Abstract, Description, Copyright, License, IsBinaryHeader=False, CommChar=TAB_COMMENT_SPLIT):
    if False:
        i = 10
        return i + 15
    Content = ''
    Abstract = ConvertSpecialUnicodes(Abstract)
    Description = ConvertSpecialUnicodes(Description)
    if IsBinaryHeader:
        Content += CommChar * 2 + TAB_SPACE_SPLIT + TAB_BINARY_HEADER_COMMENT + '\r\n'
    elif CommChar == TAB_COMMENT_EDK1_SPLIT:
        Content += CommChar + TAB_SPACE_SPLIT + TAB_COMMENT_EDK1_START + TAB_STAR + TAB_SPACE_SPLIT + TAB_HEADER_COMMENT + '\r\n'
    else:
        Content += CommChar * 2 + TAB_SPACE_SPLIT + TAB_HEADER_COMMENT + '\r\n'
    if Abstract:
        Abstract = Abstract.rstrip('\r\n')
        Content += CommChar + TAB_SPACE_SPLIT + ('\r\n' + CommChar + TAB_SPACE_SPLIT).join(GetSplitValueList(Abstract, '\n'))
        Content += '\r\n' + CommChar + '\r\n'
    else:
        Content += CommChar + '\r\n'
    if Description:
        Description = Description.rstrip('\r\n')
        Content += CommChar + TAB_SPACE_SPLIT + ('\r\n' + CommChar + TAB_SPACE_SPLIT).join(GetSplitValueList(Description, '\n'))
        Content += '\r\n' + CommChar + '\r\n'
    if Copyright:
        Copyright = Copyright.rstrip('\r\n')
        Content += CommChar + TAB_SPACE_SPLIT + ('\r\n' + CommChar + TAB_SPACE_SPLIT).join(GetSplitValueList(Copyright, '\n'))
        Content += '\r\n' + CommChar + '\r\n'
    if License:
        License = License.rstrip('\r\n')
        Content += CommChar + TAB_SPACE_SPLIT + ('\r\n' + CommChar + TAB_SPACE_SPLIT).join(GetSplitValueList(License, '\n'))
        Content += '\r\n' + CommChar + '\r\n'
    if CommChar == TAB_COMMENT_EDK1_SPLIT:
        Content += CommChar + TAB_SPACE_SPLIT + TAB_STAR + TAB_COMMENT_EDK1_END + '\r\n'
    else:
        Content += CommChar * 2 + '\r\n'
    return Content

def GenInfPcdTailComment(Usage, TailCommentText):
    if False:
        for i in range(10):
            print('nop')
    if Usage == ITEM_UNDEFINED and (not TailCommentText):
        return ''
    CommentLine = TAB_SPACE_SPLIT.join([Usage, TailCommentText])
    return GenTailCommentLines(CommentLine)

def GenInfProtocolPPITailComment(Usage, Notify, TailCommentText):
    if False:
        i = 10
        return i + 15
    if not Notify and Usage == ITEM_UNDEFINED and (not TailCommentText):
        return ''
    if Notify:
        CommentLine = USAGE_ITEM_NOTIFY + ' ## '
    else:
        CommentLine = ''
    CommentLine += TAB_SPACE_SPLIT.join([Usage, TailCommentText])
    return GenTailCommentLines(CommentLine)

def GenInfGuidTailComment(Usage, GuidTypeList, VariableName, TailCommentText):
    if False:
        print('Hello World!')
    GuidType = GuidTypeList[0]
    if Usage == ITEM_UNDEFINED and GuidType == ITEM_UNDEFINED and (not TailCommentText):
        return ''
    FirstLine = Usage + ' ## ' + GuidType
    if GuidType == TAB_INF_GUIDTYPE_VAR:
        FirstLine += ':' + VariableName
    CommentLine = TAB_SPACE_SPLIT.join([FirstLine, TailCommentText])
    return GenTailCommentLines(CommentLine)

def GenDecTailComment(SupModuleList):
    if False:
        return 10
    CommentLine = TAB_SPACE_SPLIT.join(SupModuleList)
    return GenTailCommentLines(CommentLine)

def _GetHelpStr(HelpTextObjList):
    if False:
        return 10
    ValueList = []
    for HelpObj in HelpTextObjList:
        ValueList.append((HelpObj.GetLang(), HelpObj.GetString()))
    return GetLocalValue(ValueList, True)