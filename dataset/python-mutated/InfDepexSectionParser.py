"""
InfDepexSectionParser
"""
import re
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger.ToolError import FORMAT_INVALID
from Parser.InfParserMisc import InfExpandMacro
from Library import DataType as DT
from Library.Misc import GetSplitValueList
from Parser.InfParserMisc import InfParserSectionRoot

class InfDepexSectionParser(InfParserSectionRoot):

    def InfDepexParser(self, SectionString, InfSectionObject, FileName):
        if False:
            for i in range(10):
                print('nop')
        DepexContent = []
        DepexComment = []
        ValueList = []
        for Line in SectionString:
            LineContent = Line[0]
            LineNo = Line[1]
            if LineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                DepexComment.append((LineContent, LineNo))
                continue
            LineContent = InfExpandMacro(LineContent, (FileName, LineContent, Line[1]), self.FileLocalMacros, None, True)
            CommentCount = LineContent.find(DT.TAB_COMMENT_SPLIT)
            if CommentCount > -1:
                DepexComment.append((LineContent[CommentCount:], LineNo))
                LineContent = LineContent[:CommentCount - 1]
            CommentCount = -1
            DepexContent.append((LineContent, LineNo))
            TokenList = GetSplitValueList(LineContent, DT.TAB_COMMENT_SPLIT)
            ValueList[0:len(TokenList)] = TokenList
        KeyList = []
        LastItem = ''
        for Item in self.LastSectionHeaderContent:
            LastItem = Item
            if (Item[1], Item[2], Item[3]) not in KeyList:
                KeyList.append((Item[1], Item[2], Item[3]))
        NewCommentList = []
        FormatCommentLn = -1
        ReFormatComment = re.compile('#(?:\\s*)\\[(.*?)\\](?:.*)', re.DOTALL)
        for CommentItem in DepexComment:
            CommentContent = CommentItem[0]
            if ReFormatComment.match(CommentContent) is not None:
                FormatCommentLn = CommentItem[1] + 1
                continue
            if CommentItem[1] != FormatCommentLn:
                NewCommentList.append(CommentContent)
            else:
                FormatCommentLn = CommentItem[1] + 1
        if not InfSectionObject.SetDepex(DepexContent, KeyList=KeyList, CommentList=NewCommentList):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[Depex]', File=FileName, Line=LastItem[3])