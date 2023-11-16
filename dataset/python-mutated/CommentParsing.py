"""
CommentParsing
"""
import re
from Library.StringUtils import GetSplitValueList
from Library.StringUtils import CleanString2
from Library.DataType import HEADER_COMMENT_NOT_STARTED
from Library.DataType import TAB_COMMENT_SPLIT
from Library.DataType import HEADER_COMMENT_LICENSE
from Library.DataType import HEADER_COMMENT_ABSTRACT
from Library.DataType import HEADER_COMMENT_COPYRIGHT
from Library.DataType import HEADER_COMMENT_DESCRIPTION
from Library.DataType import TAB_SPACE_SPLIT
from Library.DataType import TAB_COMMA_SPLIT
from Library.DataType import SUP_MODULE_LIST
from Library.DataType import TAB_VALUE_SPLIT
from Library.DataType import TAB_PCD_VALIDRANGE
from Library.DataType import TAB_PCD_VALIDLIST
from Library.DataType import TAB_PCD_EXPRESSION
from Library.DataType import TAB_PCD_PROMPT
from Library.DataType import TAB_CAPHEX_START
from Library.DataType import TAB_HEX_START
from Library.DataType import PCD_ERR_CODE_MAX_SIZE
from Library.ExpressionValidate import IsValidRangeExpr
from Library.ExpressionValidate import IsValidListExpr
from Library.ExpressionValidate import IsValidLogicalExpr
from Object.POM.CommonObject import TextObject
from Object.POM.CommonObject import PcdErrorObject
import Logger.Log as Logger
from Logger.ToolError import FORMAT_INVALID
from Logger.ToolError import FORMAT_NOT_SUPPORTED
from Logger import StringTable as ST

def ParseHeaderCommentSection(CommentList, FileName=None, IsBinaryHeader=False):
    if False:
        for i in range(10):
            print('nop')
    Abstract = ''
    Description = ''
    Copyright = ''
    License = ''
    EndOfLine = '\n'
    if IsBinaryHeader:
        STR_HEADER_COMMENT_START = '@BinaryHeader'
    else:
        STR_HEADER_COMMENT_START = '@file'
    HeaderCommentStage = HEADER_COMMENT_NOT_STARTED
    Last = 0
    for Index in range(len(CommentList) - 1, 0, -1):
        Line = CommentList[Index][0]
        if _IsCopyrightLine(Line):
            Last = Index
            break
    for Item in CommentList:
        Line = Item[0]
        LineNo = Item[1]
        if not Line.startswith(TAB_COMMENT_SPLIT) and Line:
            Logger.Error('\nUPT', FORMAT_INVALID, ST.ERR_INVALID_COMMENT_FORMAT, FileName, Item[1])
        Comment = CleanString2(Line)[1]
        Comment = Comment.strip()
        if not Comment and HeaderCommentStage not in [HEADER_COMMENT_LICENSE, HEADER_COMMENT_DESCRIPTION, HEADER_COMMENT_ABSTRACT]:
            continue
        if HeaderCommentStage == HEADER_COMMENT_NOT_STARTED:
            if Comment.startswith(STR_HEADER_COMMENT_START):
                HeaderCommentStage = HEADER_COMMENT_ABSTRACT
            else:
                License += Comment + EndOfLine
        elif HeaderCommentStage == HEADER_COMMENT_ABSTRACT:
            if not Comment:
                HeaderCommentStage = HEADER_COMMENT_DESCRIPTION
            elif _IsCopyrightLine(Comment):
                (Result, ErrMsg) = _ValidateCopyright(Comment)
                ValidateCopyright(Result, ST.WRN_INVALID_COPYRIGHT, FileName, LineNo, ErrMsg)
                Copyright += Comment + EndOfLine
                HeaderCommentStage = HEADER_COMMENT_COPYRIGHT
            else:
                Abstract += Comment + EndOfLine
                HeaderCommentStage = HEADER_COMMENT_DESCRIPTION
        elif HeaderCommentStage == HEADER_COMMENT_DESCRIPTION:
            if _IsCopyrightLine(Comment):
                (Result, ErrMsg) = _ValidateCopyright(Comment)
                ValidateCopyright(Result, ST.WRN_INVALID_COPYRIGHT, FileName, LineNo, ErrMsg)
                Copyright += Comment + EndOfLine
                HeaderCommentStage = HEADER_COMMENT_COPYRIGHT
            else:
                Description += Comment + EndOfLine
        elif HeaderCommentStage == HEADER_COMMENT_COPYRIGHT:
            if _IsCopyrightLine(Comment):
                (Result, ErrMsg) = _ValidateCopyright(Comment)
                ValidateCopyright(Result, ST.WRN_INVALID_COPYRIGHT, FileName, LineNo, ErrMsg)
                Copyright += Comment + EndOfLine
            elif LineNo > Last:
                if License:
                    License += EndOfLine
                License += Comment + EndOfLine
                HeaderCommentStage = HEADER_COMMENT_LICENSE
        else:
            if not Comment and (not License):
                continue
            License += Comment + EndOfLine
    return (Abstract.strip(), Description.strip(), Copyright.strip(), License.strip())

def _IsCopyrightLine(LineContent):
    if False:
        print('Hello World!')
    LineContent = LineContent.upper()
    Result = False
    ReIsCopyrightRe = re.compile('(^|\\s)COPYRIGHT *\\(', re.DOTALL)
    if ReIsCopyrightRe.search(LineContent):
        Result = True
    return Result

def ParseGenericComment(GenericComment, ContainerFile=None, SkipTag=None):
    if False:
        i = 10
        return i + 15
    if ContainerFile:
        pass
    HelpTxt = None
    HelpStr = ''
    for Item in GenericComment:
        CommentLine = Item[0]
        Comment = CleanString2(CommentLine)[1]
        if SkipTag is not None and Comment.startswith(SkipTag):
            Comment = Comment.replace(SkipTag, '', 1)
        HelpStr += Comment + '\n'
    if HelpStr:
        HelpTxt = TextObject()
        if HelpStr.endswith('\n') and (not HelpStr.endswith('\n\n')) and (HelpStr != '\n'):
            HelpStr = HelpStr[:-1]
        HelpTxt.SetString(HelpStr)
    return HelpTxt

def ParsePcdErrorCode(Value=None, ContainerFile=None, LineNum=None):
    if False:
        print('Hello World!')
    try:
        if Value.strip().startswith((TAB_HEX_START, TAB_CAPHEX_START)):
            Base = 16
        else:
            Base = 10
        ErrorCode = int(Value, Base)
        if ErrorCode > PCD_ERR_CODE_MAX_SIZE or ErrorCode < 0:
            Logger.Error('Parser', FORMAT_NOT_SUPPORTED, 'The format %s of ErrorCode is not valid, should be UNIT32 type or long type' % Value, File=ContainerFile, Line=LineNum)
        ErrorCode = '0x%x' % ErrorCode
        return ErrorCode
    except ValueError as XStr:
        if XStr:
            pass
        Logger.Error('Parser', FORMAT_NOT_SUPPORTED, 'The format %s of ErrorCode is not valid, should be UNIT32 type or long type' % Value, File=ContainerFile, Line=LineNum)

def ParseDecPcdGenericComment(GenericComment, ContainerFile, TokenSpaceGuidCName, CName, MacroReplaceDict):
    if False:
        for i in range(10):
            print('nop')
    HelpStr = ''
    PromptStr = ''
    PcdErr = None
    PcdErrList = []
    ValidValueNum = 0
    ValidRangeNum = 0
    ExpressionNum = 0
    for (CommentLine, LineNum) in GenericComment:
        Comment = CleanString2(CommentLine)[1]
        MACRO_PATTERN = '[\t\\s]*\\$\\([A-Z][_A-Z0-9]*\\)'
        MatchedStrs = re.findall(MACRO_PATTERN, Comment)
        for MatchedStr in MatchedStrs:
            if MatchedStr:
                Macro = MatchedStr.strip().lstrip('$(').rstrip(')').strip()
                if Macro in MacroReplaceDict:
                    Comment = Comment.replace(MatchedStr, MacroReplaceDict[Macro])
        if Comment.startswith(TAB_PCD_VALIDRANGE):
            if ValidValueNum > 0 or ExpressionNum > 0:
                Logger.Error('Parser', FORMAT_NOT_SUPPORTED, ST.WRN_MULTI_PCD_RANGES, File=ContainerFile, Line=LineNum)
            else:
                PcdErr = PcdErrorObject()
                PcdErr.SetTokenSpaceGuidCName(TokenSpaceGuidCName)
                PcdErr.SetCName(CName)
                PcdErr.SetFileLine(Comment)
                PcdErr.SetLineNum(LineNum)
                ValidRangeNum += 1
            ValidRange = Comment.replace(TAB_PCD_VALIDRANGE, '', 1).strip()
            (Valid, Cause) = _CheckRangeExpression(ValidRange)
            if Valid:
                ValueList = ValidRange.split(TAB_VALUE_SPLIT)
                if len(ValueList) > 1:
                    PcdErr.SetValidValueRange(TAB_VALUE_SPLIT.join(ValueList[1:]).strip())
                    PcdErr.SetErrorNumber(ParsePcdErrorCode(ValueList[0], ContainerFile, LineNum))
                else:
                    PcdErr.SetValidValueRange(ValidRange)
                PcdErrList.append(PcdErr)
            else:
                Logger.Error('Parser', FORMAT_NOT_SUPPORTED, Cause, ContainerFile, LineNum)
        elif Comment.startswith(TAB_PCD_VALIDLIST):
            if ValidRangeNum > 0 or ExpressionNum > 0:
                Logger.Error('Parser', FORMAT_NOT_SUPPORTED, ST.WRN_MULTI_PCD_RANGES, File=ContainerFile, Line=LineNum)
            elif ValidValueNum > 0:
                Logger.Error('Parser', FORMAT_NOT_SUPPORTED, ST.WRN_MULTI_PCD_VALIDVALUE, File=ContainerFile, Line=LineNum)
            else:
                PcdErr = PcdErrorObject()
                PcdErr.SetTokenSpaceGuidCName(TokenSpaceGuidCName)
                PcdErr.SetCName(CName)
                PcdErr.SetFileLine(Comment)
                PcdErr.SetLineNum(LineNum)
                ValidValueNum += 1
                ValidValueExpr = Comment.replace(TAB_PCD_VALIDLIST, '', 1).strip()
            (Valid, Cause) = _CheckListExpression(ValidValueExpr)
            if Valid:
                ValidValue = Comment.replace(TAB_PCD_VALIDLIST, '', 1).replace(TAB_COMMA_SPLIT, TAB_SPACE_SPLIT)
                ValueList = ValidValue.split(TAB_VALUE_SPLIT)
                if len(ValueList) > 1:
                    PcdErr.SetValidValue(TAB_VALUE_SPLIT.join(ValueList[1:]).strip())
                    PcdErr.SetErrorNumber(ParsePcdErrorCode(ValueList[0], ContainerFile, LineNum))
                else:
                    PcdErr.SetValidValue(ValidValue)
                PcdErrList.append(PcdErr)
            else:
                Logger.Error('Parser', FORMAT_NOT_SUPPORTED, Cause, ContainerFile, LineNum)
        elif Comment.startswith(TAB_PCD_EXPRESSION):
            if ValidRangeNum > 0 or ValidValueNum > 0:
                Logger.Error('Parser', FORMAT_NOT_SUPPORTED, ST.WRN_MULTI_PCD_RANGES, File=ContainerFile, Line=LineNum)
            else:
                PcdErr = PcdErrorObject()
                PcdErr.SetTokenSpaceGuidCName(TokenSpaceGuidCName)
                PcdErr.SetCName(CName)
                PcdErr.SetFileLine(Comment)
                PcdErr.SetLineNum(LineNum)
                ExpressionNum += 1
            Expression = Comment.replace(TAB_PCD_EXPRESSION, '', 1).strip()
            (Valid, Cause) = _CheckExpression(Expression)
            if Valid:
                ValueList = Expression.split(TAB_VALUE_SPLIT)
                if len(ValueList) > 1:
                    PcdErr.SetExpression(TAB_VALUE_SPLIT.join(ValueList[1:]).strip())
                    PcdErr.SetErrorNumber(ParsePcdErrorCode(ValueList[0], ContainerFile, LineNum))
                else:
                    PcdErr.SetExpression(Expression)
                PcdErrList.append(PcdErr)
            else:
                Logger.Error('Parser', FORMAT_NOT_SUPPORTED, Cause, ContainerFile, LineNum)
        elif Comment.startswith(TAB_PCD_PROMPT):
            if PromptStr:
                Logger.Error('Parser', FORMAT_NOT_SUPPORTED, ST.WRN_MULTI_PCD_PROMPT, File=ContainerFile, Line=LineNum)
            PromptStr = Comment.replace(TAB_PCD_PROMPT, '', 1).strip()
        elif Comment:
            HelpStr += Comment + '\n'
    if HelpStr.endswith('\n'):
        if HelpStr != '\n' and (not HelpStr.endswith('\n\n')):
            HelpStr = HelpStr[:-1]
    return (HelpStr, PcdErrList, PromptStr)

def ParseDecPcdTailComment(TailCommentList, ContainerFile):
    if False:
        for i in range(10):
            print('nop')
    assert len(TailCommentList) == 1
    TailComment = TailCommentList[0][0]
    LineNum = TailCommentList[0][1]
    Comment = TailComment.lstrip(' #')
    ReFindFirstWordRe = re.compile('^([^ #]*)', re.DOTALL)
    MatchObject = ReFindFirstWordRe.match(Comment)
    if not (MatchObject and MatchObject.group(1) in SUP_MODULE_LIST):
        return (None, Comment)
    if Comment.find(TAB_COMMENT_SPLIT) == -1:
        Comment += TAB_COMMENT_SPLIT
    (SupMode, HelpStr) = GetSplitValueList(Comment, TAB_COMMENT_SPLIT, 1)
    SupModuleList = []
    for Mod in GetSplitValueList(SupMode, TAB_SPACE_SPLIT):
        if not Mod:
            continue
        elif Mod not in SUP_MODULE_LIST:
            Logger.Error('UPT', FORMAT_INVALID, ST.WRN_INVALID_MODULE_TYPE % Mod, ContainerFile, LineNum)
        else:
            SupModuleList.append(Mod)
    return (SupModuleList, HelpStr)

def _CheckListExpression(Expression):
    if False:
        while True:
            i = 10
    ListExpr = ''
    if TAB_VALUE_SPLIT in Expression:
        ListExpr = Expression[Expression.find(TAB_VALUE_SPLIT) + 1:]
    else:
        ListExpr = Expression
    return IsValidListExpr(ListExpr)

def _CheckExpression(Expression):
    if False:
        print('Hello World!')
    Expr = ''
    if TAB_VALUE_SPLIT in Expression:
        Expr = Expression[Expression.find(TAB_VALUE_SPLIT) + 1:]
    else:
        Expr = Expression
    return IsValidLogicalExpr(Expr, True)

def _CheckRangeExpression(Expression):
    if False:
        return 10
    RangeExpr = ''
    if TAB_VALUE_SPLIT in Expression:
        RangeExpr = Expression[Expression.find(TAB_VALUE_SPLIT) + 1:]
    else:
        RangeExpr = Expression
    return IsValidRangeExpr(RangeExpr)

def ValidateCopyright(Result, ErrType, FileName, LineNo, ErrMsg):
    if False:
        for i in range(10):
            print('nop')
    if not Result:
        Logger.Warn('\nUPT', ErrType, FileName, LineNo, ErrMsg)

def _ValidateCopyright(Line):
    if False:
        i = 10
        return i + 15
    if Line:
        pass
    Result = True
    ErrMsg = ''
    return (Result, ErrMsg)

def GenerateTokenList(Comment):
    if False:
        i = 10
        return i + 15
    ReplacedComment = None
    while Comment != ReplacedComment:
        ReplacedComment = Comment
        Comment = Comment.replace('##', '#').replace('  ', ' ').replace(' ', '#').strip('# ')
    return Comment.split('#')

def ParseComment(Comment, UsageTokens, TypeTokens, RemoveTokens, ParseVariable):
    if False:
        while True:
            i = 10
    Usage = None
    Type = None
    String = None
    Comment = Comment[0]
    NumTokens = 2
    if ParseVariable:
        List = Comment.split(':', 1)
        if len(List) > 1:
            SubList = GenerateTokenList(List[0].strip())
            if len(SubList) in [1, 2] and SubList[-1] == 'Variable':
                if List[1].strip().find('L"') == 0:
                    Comment = List[0].strip() + ':' + List[1].strip()
        End = -1
        Start = Comment.find('Variable:L"')
        if Start >= 0:
            String = Comment[Start + 9:]
            End = String[2:].find('"')
        else:
            Start = Comment.find('L"')
            if Start >= 0:
                String = Comment[Start:]
                End = String[2:].find('"')
        if End >= 0:
            SubList = GenerateTokenList(Comment[:Start])
            if len(SubList) < 2:
                Comment = Comment[:Start] + String[End + 3:]
                String = String[:End + 3]
                Type = 'Variable'
                NumTokens = 1
    HelpText = Comment
    List = GenerateTokenList(Comment)
    for Token in List[0:NumTokens]:
        if Usage is None and Token in UsageTokens:
            Usage = UsageTokens[Token]
            HelpText = HelpText.replace(Token, '')
    if Usage is not None or not ParseVariable:
        for Token in List[0:NumTokens]:
            if Type is None and Token in TypeTokens:
                Type = TypeTokens[Token]
                HelpText = HelpText.replace(Token, '')
            if Usage is not None:
                for Token in List[0:NumTokens]:
                    if Token in RemoveTokens:
                        HelpText = HelpText.replace(Token, '')
    if Usage is None:
        Usage = 'UNDEFINED'
    if Type is None:
        Type = 'UNDEFINED'
    if Type != 'Variable':
        String = None
    HelpText = HelpText.lstrip('# ')
    if HelpText == '':
        HelpText = None
    return (Usage, Type, String, HelpText)