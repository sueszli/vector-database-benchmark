from __future__ import absolute_import
import Common.LongFilePathOs as os
from CommonDataClass.DataClass import *
from Ecc.EccToolError import *
from Common.MultipleWorkspace import MultipleWorkspace as mws
from Ecc import EccGlobalData
import re

def GetIncludeListOfFile(WorkSpace, Filepath, Db):
    if False:
        print('Hello World!')
    IncludeList = []
    Filepath = os.path.normpath(Filepath)
    SqlCommand = "\n                select Value1, FullPath from Inf, File where Inf.Model = %s and Inf.BelongsToFile in(\n                    select distinct B.BelongsToFile from File as A left join Inf as B\n                        where A.ID = B.BelongsToFile and B.Model = %s and (A.Path || '%s' || B.Value1) = '%s')\n                        and Inf.BelongsToFile = File.ID" % (MODEL_META_DATA_PACKAGE, MODEL_EFI_SOURCE_FILE, '\\', Filepath)
    RecordSet = Db.TblFile.Exec(SqlCommand)
    for Record in RecordSet:
        DecFullPath = os.path.normpath(mws.join(WorkSpace, Record[0]))
        InfFullPath = os.path.normpath(mws.join(WorkSpace, Record[1]))
        (DecPath, DecName) = os.path.split(DecFullPath)
        (InfPath, InfName) = os.path.split(InfFullPath)
        SqlCommand = "select Value1 from Dec where BelongsToFile =\n                           (select ID from File where FullPath = '%s') and Model = %s" % (DecFullPath, MODEL_EFI_INCLUDE)
        NewRecordSet = Db.TblDec.Exec(SqlCommand)
        if InfPath not in IncludeList:
            IncludeList.append(InfPath)
        for NewRecord in NewRecordSet:
            IncludePath = os.path.normpath(os.path.join(DecPath, NewRecord[0]))
            if IncludePath not in IncludeList:
                IncludeList.append(IncludePath)
    return IncludeList

def GetFileList(FileModel, Db):
    if False:
        for i in range(10):
            print('nop')
    FileList = []
    SqlCommand = 'select FullPath from File where Model = %s' % str(FileModel)
    RecordSet = Db.TblFile.Exec(SqlCommand)
    for Record in RecordSet:
        FileList.append(Record[0])
    return FileList

def GetTableList(FileModelList, Table, Db):
    if False:
        print('Hello World!')
    TableList = []
    SqlCommand = 'select ID from File where Model in %s' % str(FileModelList)
    RecordSet = Db.TblFile.Exec(SqlCommand)
    for Record in RecordSet:
        TableName = Table + str(Record[0])
        TableList.append(TableName)
    return TableList

def ParseHeaderCommentSection(CommentList, FileName=None):
    if False:
        while True:
            i = 10
    Abstract = ''
    Description = ''
    Copyright = ''
    License = ''
    EndOfLine = '\n'
    STR_HEADER_COMMENT_START = '@file'
    HEADER_COMMENT_NOT_STARTED = -1
    HEADER_COMMENT_STARTED = 0
    HEADER_COMMENT_FILE = 1
    HEADER_COMMENT_ABSTRACT = 2
    HEADER_COMMENT_DESCRIPTION = 3
    HEADER_COMMENT_COPYRIGHT = 4
    HEADER_COMMENT_LICENSE = 5
    HEADER_COMMENT_END = 6
    Last = 0
    HeaderCommentStage = HEADER_COMMENT_NOT_STARTED
    for Index in range(len(CommentList) - 1, 0, -1):
        Line = CommentList[Index][0]
        if _IsCopyrightLine(Line):
            Last = Index
            break
    for Item in CommentList:
        Line = Item[0]
        LineNo = Item[1]
        if not Line.startswith('#') and Line:
            SqlStatement = " select ID from File where FullPath like '%s'" % FileName
            ResultSet = EccGlobalData.gDb.TblFile.Exec(SqlStatement)
            for Result in ResultSet:
                Msg = 'Comment must start with #'
                EccGlobalData.gDb.TblReport.Insert(ERROR_DOXYGEN_CHECK_FILE_HEADER, Msg, 'File', Result[0])
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
                Abstract = ''
                HeaderCommentStage = HEADER_COMMENT_DESCRIPTION
            elif _IsCopyrightLine(Comment):
                Copyright += Comment + EndOfLine
                HeaderCommentStage = HEADER_COMMENT_COPYRIGHT
            else:
                Abstract += Comment + EndOfLine
                HeaderCommentStage = HEADER_COMMENT_DESCRIPTION
        elif HeaderCommentStage == HEADER_COMMENT_DESCRIPTION:
            if _IsCopyrightLine(Comment):
                Copyright += Comment + EndOfLine
                HeaderCommentStage = HEADER_COMMENT_COPYRIGHT
            else:
                Description += Comment + EndOfLine
        elif HeaderCommentStage == HEADER_COMMENT_COPYRIGHT:
            if _IsCopyrightLine(Comment):
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
    if not Copyright.strip():
        SqlStatement = " select ID from File where FullPath like '%s'" % FileName
        ResultSet = EccGlobalData.gDb.TblFile.Exec(SqlStatement)
        for Result in ResultSet:
            Msg = 'Header comment section must have copyright information'
            EccGlobalData.gDb.TblReport.Insert(ERROR_DOXYGEN_CHECK_FILE_HEADER, Msg, 'File', Result[0])
    if not License.strip():
        SqlStatement = " select ID from File where FullPath like '%s'" % FileName
        ResultSet = EccGlobalData.gDb.TblFile.Exec(SqlStatement)
        for Result in ResultSet:
            Msg = 'Header comment section must have license information'
            EccGlobalData.gDb.TblReport.Insert(ERROR_DOXYGEN_CHECK_FILE_HEADER, Msg, 'File', Result[0])
    if not Abstract.strip() or Abstract.find('Component description file') > -1:
        SqlStatement = " select ID from File where FullPath like '%s'" % FileName
        ResultSet = EccGlobalData.gDb.TblFile.Exec(SqlStatement)
        for Result in ResultSet:
            Msg = 'Header comment section must have Abstract information.'
            EccGlobalData.gDb.TblReport.Insert(ERROR_DOXYGEN_CHECK_FILE_HEADER, Msg, 'File', Result[0])
    return (Abstract.strip(), Description.strip(), Copyright.strip(), License.strip())

def _IsCopyrightLine(LineContent):
    if False:
        while True:
            i = 10
    LineContent = LineContent.upper()
    Result = False
    ReIsCopyrightRe = re.compile('(^|\\s)COPYRIGHT *\\(', re.DOTALL)
    ReIsCopyrightTypeB = re.compile('(^|\\s)\\(C\\)\\s*COPYRIGHT', re.DOTALL)
    if ReIsCopyrightRe.search(LineContent) or ReIsCopyrightTypeB.search(LineContent):
        Result = True
    return Result

def CleanString2(Line, CommentCharacter='#', AllowCppStyleComment=False):
    if False:
        print('Hello World!')
    Line = Line.strip()
    if AllowCppStyleComment:
        Line = Line.replace('//', CommentCharacter)
    LineParts = Line.split(CommentCharacter, 1)
    Line = LineParts[0].strip()
    if len(LineParts) > 1:
        Comment = LineParts[1].strip()
        Start = 0
        End = len(Comment)
        while Start < End and Comment.startswith(CommentCharacter, Start, End):
            Start += 1
        while End >= 0 and Comment.endswith(CommentCharacter, Start, End):
            End -= 1
        Comment = Comment[Start:End]
        Comment = Comment.strip()
    else:
        Comment = ''
    return (Line, Comment)