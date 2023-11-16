from __future__ import print_function
import os
from Object.Parser.InfCommonObject import CurrentLine
from Object.Parser.InfCommonObject import InfLineCommentObject
from Object.Parser.InfBinaryObject import InfBinariesObject
import Logger.Log as Logger
import Library.GlobalData as Global
SectionStringsCommonItem1 = '\nGUID\n'
SectionStringsCommonItem2 = '\nGUID | Test/Test.guid\n'
SectionStringsCommonItem3 = '\nGUID | Test/Test.guid | DEBUG\n'
SectionStringsCommonItem4 = '\nGUID | Test/Test.guid | $(TARGET)\n'
SectionStringsCommonItem5 = '\nDEFINE BINARY_FILE_PATH = Test\nGUID | $(BINARY_FILE_PATH)/Test.guid | $(TARGET)\n'
SectionStringsCommonItem6 = '\nGUID | Test/Test.guid | DEBUG | *\n'
SectionStringsCommonItem7 = '\nGUID | Test/Test.guid | DEBUG | MSFT\n'
SectionStringsCommonItem8 = '\nGUID | Test/Test.guid | DEBUG | MSFT | TEST\n'
SectionStringsCommonItem9 = '\nGUID | Test/Test.guid | DEBUG | MSFT | TEST | TRUE\n'
SectionStringsCommonItem10 = '\nGUID | Test/Test.guid | DEBUG | MSFT | TEST | TRUE | OVERFLOW\n'
SectionStringsVerItem1 = '\nVER\n'
SectionStringsVerItem2 = '\nVER | Test/Test.ver | * | TRUE | OverFlow\n'
SectionStringsVerItem3 = '\nVER | Test/Test.ver\n'
SectionStringsVerItem4 = '\nVER | Test/Test.ver | DEBUG\n'
SectionStringsVerItem5 = '\nVER | Test/Test.ver | DEBUG | TRUE\n'
SectionStringsVerItem6 = '\nVER | Test/Test.ver | * | TRUE\nVER | Test/Test2.ver | * | TRUE\n'
SectionStringsVerItem7 = '\nVER | Test/Test.ver | * | TRUE\nVER | Test/Test2.ver | * | FALSE\n'
SectionStringsUiItem1 = '\nUI | Test/Test.ui | * | TRUE\nUI | Test/Test2.ui | * | TRUE\n'
SectionStringsUiItem2 = '\nUI | Test/Test.ui | * | TRUE\nSEC_UI | Test/Test2.ui | * | TRUE\n'
SectionStringsUiItem3 = '\nUI | Test/Test.ui | * | TRUE\nUI | Test/Test2.ui | * | FALSE\n'
SectionStringsUiItem4 = '\nUI\n'
SectionStringsUiItem5 = '\nUI | Test/Test.ui | * | TRUE | OverFlow\n'
SectionStringsUiItem6 = '\nUI | Test/Test.ui\n'
SectionStringsUiItem7 = '\nUI | Test/Test.ui | DEBUG\n'
SectionStringsUiItem8 = '\nUI | Test/Test.ui | DEBUG | TRUE\n'
gFileName = 'BinarySectionTest.inf'

def StringToSectionString(String):
    if False:
        while True:
            i = 10
    Lines = String.split('\n')
    LineNo = 0
    SectionString = []
    for Line in Lines:
        if Line.strip() == '':
            continue
        SectionString.append((Line, LineNo, ''))
        LineNo = LineNo + 1
    return SectionString

def PrepareTest(String):
    if False:
        i = 10
        return i + 15
    SectionString = StringToSectionString(String)
    ItemList = []
    for Item in SectionString:
        ValueList = Item[0].split('|')
        for count in range(len(ValueList)):
            ValueList[count] = ValueList[count].strip()
        if len(ValueList) >= 2:
            FileName = os.path.normpath(os.path.realpath(ValueList[1].strip()))
            try:
                TempFile = open(FileName, 'w')
                TempFile.close()
            except:
                print('File Create Error')
        CurrentLine = CurrentLine()
        CurrentLine.SetFileName('Test')
        CurrentLine.SetLineString(Item[0])
        CurrentLine.SetLineNo(Item[1])
        InfLineCommentObject = InfLineCommentObject()
        ItemList.append((ValueList, InfLineCommentObject, CurrentLine))
    return ItemList
if __name__ == '__main__':
    Logger.Initialize()
    InfBinariesInstance = InfBinariesObject()
    ArchList = ['COMMON']
    Global.gINF_MODULE_DIR = os.getcwd()
    AllPassedFlag = True
    UiStringList = [SectionStringsUiItem1, SectionStringsUiItem2, SectionStringsUiItem3, SectionStringsUiItem4, SectionStringsUiItem5, SectionStringsUiItem6, SectionStringsUiItem7, SectionStringsUiItem8]
    for Item in UiStringList:
        Ui = PrepareTest(Item)
        if Item == SectionStringsUiItem4 or Item == SectionStringsUiItem5:
            try:
                InfBinariesInstance.SetBinary(Ui=Ui, ArchList=ArchList)
            except Logger.FatalError:
                pass
        else:
            try:
                InfBinariesInstance.SetBinary(Ui=Ui, ArchList=ArchList)
            except:
                AllPassedFlag = False
    VerStringList = [SectionStringsVerItem1, SectionStringsVerItem2, SectionStringsVerItem3, SectionStringsVerItem4, SectionStringsVerItem5, SectionStringsVerItem6, SectionStringsVerItem7]
    for Item in VerStringList:
        Ver = PrepareTest(Item)
        if Item == SectionStringsVerItem1 or Item == SectionStringsVerItem2:
            try:
                InfBinariesInstance.SetBinary(Ver=Ver, ArchList=ArchList)
            except:
                pass
        else:
            try:
                InfBinariesInstance.SetBinary(Ver=Ver, ArchList=ArchList)
            except:
                AllPassedFlag = False
    CommonStringList = [SectionStringsCommonItem1, SectionStringsCommonItem2, SectionStringsCommonItem3, SectionStringsCommonItem4, SectionStringsCommonItem5, SectionStringsCommonItem6, SectionStringsCommonItem7, SectionStringsCommonItem8, SectionStringsCommonItem9, SectionStringsCommonItem10]
    for Item in CommonStringList:
        CommonBin = PrepareTest(Item)
        if Item == SectionStringsCommonItem10 or Item == SectionStringsCommonItem1:
            try:
                InfBinariesInstance.SetBinary(CommonBinary=CommonBin, ArchList=ArchList)
            except:
                pass
        else:
            try:
                InfBinariesInstance.SetBinary(Ver=Ver, ArchList=ArchList)
            except:
                print('Test Failed!')
                AllPassedFlag = False
    if AllPassedFlag:
        print('All tests passed...')
    else:
        print('Some unit test failed!')