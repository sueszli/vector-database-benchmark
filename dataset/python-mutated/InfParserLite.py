from __future__ import print_function
from __future__ import absolute_import
import Common.LongFilePathOs as os
import Common.EdkLogger as EdkLogger
from Common.DataType import *
from CommonDataClass.DataClass import *
from Eot.Identification import Identification
from Common.StringUtils import *
from Eot.Parser import *
from Eot import Database
from Eot import EotGlobalData

class EdkInfParser(object):

    def __init__(self, Filename=None, Database=None, SourceFileList=None):
        if False:
            i = 10
            return i + 15
        self.Identification = Identification()
        self.Sources = []
        self.Macros = {}
        self.Cur = Database.Cur
        self.TblFile = Database.TblFile
        self.TblInf = Database.TblInf
        self.FileID = -1
        if Filename is not None:
            self.LoadInfFile(Filename)
        if SourceFileList:
            for Item in SourceFileList:
                self.TblInf.Insert(MODEL_EFI_SOURCE_FILE, Item, '', '', '', '', 'COMMON', -1, self.FileID, -1, -1, -1, -1, 0)

    def LoadInfFile(self, Filename=None):
        if False:
            print('Hello World!')
        Filename = NormPath(Filename)
        self.Identification.FileFullPath = Filename
        (self.Identification.FileRelativePath, self.Identification.FileName) = os.path.split(Filename)
        self.FileID = self.TblFile.InsertFile(Filename, MODEL_FILE_INF)
        self.ParseInf(PreProcess(Filename, False), self.Identification.FileRelativePath, Filename)

    def ParserSource(self, CurrentSection, SectionItemList, ArchList, ThirdList):
        if False:
            return 10
        for Index in range(0, len(ArchList)):
            Arch = ArchList[Index]
            Third = ThirdList[Index]
            if Arch == '':
                Arch = TAB_ARCH_COMMON
            for Item in SectionItemList:
                if CurrentSection.upper() == 'defines'.upper():
                    (Name, Value) = AddToSelfMacro(self.Macros, Item[0])
                    self.TblInf.Insert(MODEL_META_DATA_HEADER, Name, Value, Third, '', '', Arch, -1, self.FileID, Item[1], -1, Item[1], -1, 0)

    def ParseInf(self, Lines=[], FileRelativePath='', Filename=''):
        if False:
            while True:
                i = 10
        (IfDefList, SectionItemList, CurrentSection, ArchList, ThirdList, IncludeFiles) = ([], [], TAB_UNKNOWN, [], [], [])
        LineNo = 0
        for Line in Lines:
            LineNo = LineNo + 1
            if Line == '':
                continue
            if Line.startswith(TAB_SECTION_START) and Line.endswith(TAB_SECTION_END):
                self.ParserSource(CurrentSection, SectionItemList, ArchList, ThirdList)
                SectionItemList = []
                ArchList = []
                ThirdList = []
                CurrentSection = ''
                LineList = GetSplitValueList(Line[len(TAB_SECTION_START):len(Line) - len(TAB_SECTION_END)], TAB_COMMA_SPLIT)
                for Item in LineList:
                    ItemList = GetSplitValueList(Item, TAB_SPLIT)
                    if CurrentSection == '':
                        CurrentSection = ItemList[0]
                    elif CurrentSection != ItemList[0]:
                        EdkLogger.error('Parser', PARSER_ERROR, "Different section names '%s' and '%s' are found in one section definition, this is not allowed." % (CurrentSection, ItemList[0]), File=Filename, Line=LineNo)
                    ItemList.append('')
                    ItemList.append('')
                    if len(ItemList) > 5:
                        RaiseParserError(Line, CurrentSection, Filename, '', LineNo)
                    else:
                        ArchList.append(ItemList[1].upper())
                        ThirdList.append(ItemList[2])
                continue
            SectionItemList.append([Line, LineNo])
        self.ParserSource(CurrentSection, SectionItemList, ArchList, ThirdList)