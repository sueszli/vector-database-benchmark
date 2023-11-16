import os
import unittest
from Logger.Log import FatalError
from Parser.DecParser import Dec, _DecDefine, _DecLibraryclass, _DecPcd, _DecGuid, FileContent, _DecBase, CleanString
from Object.Parser.DecObject import _DecComments

class CleanStringTestCase(unittest.TestCase):

    def testCleanString(self):
        if False:
            for i in range(10):
                print('nop')
        (Line, Comment) = CleanString('')
        self.assertEqual(Line, '')
        self.assertEqual(Comment, '')
        (Line, Comment) = CleanString('line without comment')
        self.assertEqual(Line, 'line without comment')
        self.assertEqual(Comment, '')
        (Line, Comment) = CleanString('# pure comment')
        self.assertEqual(Line, '')
        self.assertEqual(Comment, '# pure comment')
        (Line, Comment) = CleanString('line # and comment')
        self.assertEqual(Line, 'line')
        self.assertEqual(Comment, '# and comment')

    def testCleanStringCpp(self):
        if False:
            print('Hello World!')
        (Line, Comment) = CleanString('line // and comment', AllowCppStyleComment=True)
        self.assertEqual(Line, 'line')
        self.assertEqual(Comment, '# and comment')

class MacroParserTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.dec = _DecBase(FileContent('dummy', []))

    def testCorrectMacro(self):
        if False:
            print('Hello World!')
        self.dec._MacroParser('DEFINE MACRO1 = test1')
        self.failIf('MACRO1' not in self.dec._LocalMacro)
        self.assertEqual(self.dec._LocalMacro['MACRO1'], 'test1')

    def testErrorMacro1(self):
        if False:
            while True:
                i = 10
        self.assertRaises(FatalError, self.dec._MacroParser, 'DEFINE not_upper_case = test2')

    def testErrorMacro2(self):
        if False:
            return 10
        self.assertRaises(FatalError, self.dec._MacroParser, 'DEFINE ')

class TryBackSlashTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        Content = ['test no backslash', 'test with backslash \\', 'continue second line', '\\', 'line with backslash \\', '']
        self.dec = _DecBase(FileContent('dummy', Content))

    def testBackSlash(self):
        if False:
            print('Hello World!')
        (ConcatLine, CommentList) = self.dec._TryBackSlash(self.dec._RawData.GetNextLine(), [])
        self.assertEqual(ConcatLine, 'test no backslash')
        self.assertEqual(CommentList, [])
        (ConcatLine, CommentList) = self.dec._TryBackSlash(self.dec._RawData.GetNextLine(), [])
        self.assertEqual(CommentList, [])
        self.assertEqual(ConcatLine, 'test with backslash continue second line')
        self.assertRaises(FatalError, self.dec._TryBackSlash, self.dec._RawData.GetNextLine(), [])
        self.assertRaises(FatalError, self.dec._TryBackSlash, self.dec._RawData.GetNextLine(), [])

class DataItem(_DecComments):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        _DecComments.__init__(self)
        self.String = ''

class Data(_DecComments):

    def __init__(self):
        if False:
            return 10
        _DecComments.__init__(self)
        self.ItemList = []

class TestInner(_DecBase):

    def __init__(self, RawData):
        if False:
            i = 10
            return i + 15
        _DecBase.__init__(self, RawData)
        self.ItemObject = Data()

    def _StopCurrentParsing(self, Line):
        if False:
            i = 10
            return i + 15
        return Line == '[TOP]'

    def _ParseItem(self):
        if False:
            print('Hello World!')
        Item = DataItem()
        Item.String = self._RawData.CurrentLine
        self.ItemObject.ItemList.append(Item)
        return Item

    def _TailCommentStrategy(self, Comment):
        if False:
            while True:
                i = 10
        return Comment.find('@comment') != -1

class TestTop(_DecBase):

    def __init__(self, RawData):
        if False:
            for i in range(10):
                print('nop')
        _DecBase.__init__(self, RawData)
        self.ItemObject = []

    def _StopCurrentParsing(self, Line):
        if False:
            while True:
                i = 10
        return False

    def _ParseItem(self):
        if False:
            for i in range(10):
                print('nop')
        TestParser = TestInner(self._RawData)
        TestParser.Parse()
        self.ItemObject.append(TestParser.ItemObject)
        return TestParser.ItemObject

class ParseTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def testParse(self):
        if False:
            i = 10
            return i + 15
        Content = '# Top comment\n        [TOP]\n          # sub1 head comment\n          (test item has both head and tail comment) # sub1 tail comment\n          # sub2 head comment\n          (test item has head and special tail comment)\n          # @comment test TailCommentStrategy branch\n\n          (test item has no comment)\n\n        # test NextLine branch\n        [TOP]\n          sub-item\n        '
        dec = TestTop(FileContent('dummy', Content.splitlines()))
        dec.Parse()
        self.assertEqual(len(dec.ItemObject), 2)
        data = dec.ItemObject[0]
        self.assertEqual(data._HeadComment[0][0], '# Top comment')
        self.assertEqual(data._HeadComment[0][1], 1)
        self.assertEqual(len(data.ItemList), 3)
        dataitem = data.ItemList[0]
        self.assertEqual(dataitem.String, '(test item has both head and tail comment)')
        self.assertEqual(dataitem._HeadComment[0][0], '# sub1 head comment')
        self.assertEqual(dataitem._TailComment[0][0], '# sub1 tail comment')
        self.assertEqual(dataitem._HeadComment[0][1], 3)
        self.assertEqual(dataitem._TailComment[0][1], 4)
        dataitem = data.ItemList[1]
        self.assertEqual(dataitem.String, '(test item has head and special tail comment)')
        self.assertEqual(dataitem._HeadComment[0][0], '# sub2 head comment')
        self.assertEqual(dataitem._TailComment[0][0], '# @comment test TailCommentStrategy branch')
        self.assertEqual(dataitem._HeadComment[0][1], 5)
        self.assertEqual(dataitem._TailComment[0][1], 7)
        dataitem = data.ItemList[2]
        self.assertEqual(dataitem.String, '(test item has no comment)')
        self.assertEqual(dataitem._HeadComment, [])
        self.assertEqual(dataitem._TailComment, [])
        data = dec.ItemObject[1]
        self.assertEqual(data._HeadComment[0][0], '# test NextLine branch')
        self.assertEqual(data._HeadComment[0][1], 11)
        self.assertEqual(len(data.ItemList), 1)
        dataitem = data.ItemList[0]
        self.assertEqual(dataitem.String, 'sub-item')
        self.assertEqual(dataitem._HeadComment, [])
        self.assertEqual(dataitem._TailComment, [])

class DecDefineTestCase(unittest.TestCase):

    def GetObj(self, Content):
        if False:
            return 10
        Obj = _DecDefine(FileContent('dummy', Content.splitlines()))
        Obj._RawData.CurrentLine = Obj._RawData.GetNextLine()
        return Obj

    def testDecDefine(self):
        if False:
            print('Hello World!')
        item = self.GetObj('PACKAGE_NAME = MdePkg')._ParseItem()
        self.assertEqual(item.Key, 'PACKAGE_NAME')
        self.assertEqual(item.Value, 'MdePkg')

    def testDecDefine1(self):
        if False:
            print('Hello World!')
        obj = self.GetObj('PACKAGE_NAME')
        self.assertRaises(FatalError, obj._ParseItem)

    def testDecDefine2(self):
        if False:
            print('Hello World!')
        obj = self.GetObj('unknown_key = ')
        self.assertRaises(FatalError, obj._ParseItem)

    def testDecDefine3(self):
        if False:
            while True:
                i = 10
        obj = self.GetObj('PACKAGE_NAME = ')
        self.assertRaises(FatalError, obj._ParseItem)

class DecLibraryTestCase(unittest.TestCase):

    def GetObj(self, Content):
        if False:
            for i in range(10):
                print('nop')
        Obj = _DecLibraryclass(FileContent('dummy', Content.splitlines()))
        Obj._RawData.CurrentLine = Obj._RawData.GetNextLine()
        return Obj

    def testNoInc(self):
        if False:
            print('Hello World!')
        obj = self.GetObj('UefiRuntimeLib')
        self.assertRaises(FatalError, obj._ParseItem)

    def testEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        obj = self.GetObj(' | ')
        self.assertRaises(FatalError, obj._ParseItem)

    def testLibclassNaming(self):
        if False:
            print('Hello World!')
        obj = self.GetObj('lowercase_efiRuntimeLib|Include/Library/UefiRuntimeLib.h')
        self.assertRaises(FatalError, obj._ParseItem)

    def testLibclassExt(self):
        if False:
            while True:
                i = 10
        obj = self.GetObj('RuntimeLib|Include/Library/UefiRuntimeLib.no_h')
        self.assertRaises(FatalError, obj._ParseItem)

    def testLibclassRelative(self):
        if False:
            for i in range(10):
                print('nop')
        obj = self.GetObj('RuntimeLib|Include/../UefiRuntimeLib.h')
        self.assertRaises(FatalError, obj._ParseItem)

class DecPcdTestCase(unittest.TestCase):

    def GetObj(self, Content):
        if False:
            i = 10
            return i + 15
        Obj = _DecPcd(FileContent('dummy', Content.splitlines()))
        Obj._RawData.CurrentLine = Obj._RawData.GetNextLine()
        Obj._RawData.CurrentScope = [('PcdsFeatureFlag'.upper(), 'COMMON')]
        return Obj

    def testOK(self):
        if False:
            for i in range(10):
                print('nop')
        item = self.GetObj('gEfiMdePkgTokenSpaceGuid.PcdComponentNameDisable|FALSE|BOOLEAN|0x0000000d')._ParseItem()
        self.assertEqual(item.TokenSpaceGuidCName, 'gEfiMdePkgTokenSpaceGuid')
        self.assertEqual(item.TokenCName, 'PcdComponentNameDisable')
        self.assertEqual(item.DefaultValue, 'FALSE')
        self.assertEqual(item.DatumType, 'BOOLEAN')
        self.assertEqual(item.TokenValue, '0x0000000d')

    def testNoCvar(self):
        if False:
            while True:
                i = 10
        obj = self.GetObj('123ai.PcdComponentNameDisable|FALSE|BOOLEAN|0x0000000d')
        self.assertRaises(FatalError, obj._ParseItem)

    def testSplit(self):
        if False:
            while True:
                i = 10
        obj = self.GetObj('gEfiMdePkgTokenSpaceGuid.PcdComponentNameDisable FALSE|BOOLEAN|0x0000000d')
        self.assertRaises(FatalError, obj._ParseItem)
        obj = self.GetObj('gEfiMdePkgTokenSpaceGuid.PcdComponentNameDisable|FALSE|BOOLEAN|0x0000000d | abc')
        self.assertRaises(FatalError, obj._ParseItem)

    def testUnknownType(self):
        if False:
            return 10
        obj = self.GetObj('gEfiMdePkgTokenSpaceGuid.PcdComponentNameDisable|FALSE|unknown|0x0000000d')
        self.assertRaises(FatalError, obj._ParseItem)

    def testVoid(self):
        if False:
            print('Hello World!')
        obj = self.GetObj('gEfiMdePkgTokenSpaceGuid.PcdComponentNameDisable|abc|VOID*|0x0000000d')
        self.assertRaises(FatalError, obj._ParseItem)

    def testUINT(self):
        if False:
            while True:
                i = 10
        obj = self.GetObj('gEfiMdePkgTokenSpaceGuid.PcdComponentNameDisable|0xabc|UINT8|0x0000000d')
        self.assertRaises(FatalError, obj._ParseItem)

class DecIncludeTestCase(unittest.TestCase):
    pass

class DecGuidTestCase(unittest.TestCase):

    def GetObj(self, Content):
        if False:
            i = 10
            return i + 15
        Obj = _DecGuid(FileContent('dummy', Content.splitlines()))
        Obj._RawData.CurrentLine = Obj._RawData.GetNextLine()
        Obj._RawData.CurrentScope = [('guids'.upper(), 'COMMON')]
        return Obj

    def testCValue(self):
        if False:
            print('Hello World!')
        item = self.GetObj('gEfiIpSecProtocolGuid={ 0xdfb386f7, 0xe100, 0x43ad, {0x9c, 0x9a, 0xed, 0x90, 0xd0, 0x8a, 0x5e, 0x12 }}')._ParseItem()
        self.assertEqual(item.GuidCName, 'gEfiIpSecProtocolGuid')
        self.assertEqual(item.GuidCValue, '{ 0xdfb386f7, 0xe100, 0x43ad, {0x9c, 0x9a, 0xed, 0x90, 0xd0, 0x8a, 0x5e, 0x12 }}')

    def testGuidString(self):
        if False:
            while True:
                i = 10
        item = self.GetObj('gEfiIpSecProtocolGuid=1E73767F-8F52-4603-AEB4-F29B510B6766')._ParseItem()
        self.assertEqual(item.GuidCName, 'gEfiIpSecProtocolGuid')
        self.assertEqual(item.GuidCValue, '1E73767F-8F52-4603-AEB4-F29B510B6766')

    def testNoValue1(self):
        if False:
            while True:
                i = 10
        obj = self.GetObj('gEfiIpSecProtocolGuid')
        self.assertRaises(FatalError, obj._ParseItem)

    def testNoValue2(self):
        if False:
            return 10
        obj = self.GetObj('gEfiIpSecProtocolGuid=')
        self.assertRaises(FatalError, obj._ParseItem)

    def testNoName(self):
        if False:
            while True:
                i = 10
        obj = self.GetObj('=')
        self.assertRaises(FatalError, obj._ParseItem)

class DecDecInitTestCase(unittest.TestCase):

    def testNoDecFile(self):
        if False:
            while True:
                i = 10
        self.assertRaises(FatalError, Dec, 'No_Such_File')

class TmpFile:

    def __init__(self, File):
        if False:
            while True:
                i = 10
        self.File = File

    def Write(self, Content):
        if False:
            while True:
                i = 10
        try:
            FileObj = open(self.File, 'w')
            FileObj.write(Content)
            FileObj.close()
        except:
            pass

    def Remove(self):
        if False:
            i = 10
            return i + 15
        try:
            os.remove(self.File)
        except:
            pass

class DecUESectionTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.File = TmpFile('test.dec')
        self.File.Write('[userextensions.intel."myid"]\n[userextensions.intel."myid".IA32]\n[userextensions.intel."myid".IA32,]\n[userextensions.intel."myid]\n')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.File.Remove()

    def testUserExtentionHeader(self):
        if False:
            while True:
                i = 10
        dec = Dec('test.dec', False)
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        dec._UserExtentionSectionParser()
        self.assertEqual(len(dec._RawData.CurrentScope), 1)
        self.assertEqual(dec._RawData.CurrentScope[0][0], 'userextensions'.upper())
        self.assertEqual(dec._RawData.CurrentScope[0][1], 'intel')
        self.assertEqual(dec._RawData.CurrentScope[0][2], '"myid"')
        self.assertEqual(dec._RawData.CurrentScope[0][3], 'COMMON')
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        dec._UserExtentionSectionParser()
        self.assertEqual(len(dec._RawData.CurrentScope), 1)
        self.assertEqual(dec._RawData.CurrentScope[0][0], 'userextensions'.upper())
        self.assertEqual(dec._RawData.CurrentScope[0][1], 'intel')
        self.assertEqual(dec._RawData.CurrentScope[0][2], '"myid"')
        self.assertEqual(dec._RawData.CurrentScope[0][3], 'IA32')
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        self.assertRaises(FatalError, dec._UserExtentionSectionParser)
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        self.assertRaises(FatalError, dec._UserExtentionSectionParser)

class DecSectionTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.File = TmpFile('test.dec')
        self.File.Write('[no section start or end\n[,] # empty sub-section\n[unknow_section_name]\n[Includes.IA32.other] # no third one\n[PcdsFeatureFlag, PcdsFixedAtBuild] # feature flag PCD must not be in the same section of other types of PCD\n[Includes.IA32, Includes.IA32]\n[Includes, Includes.IA32] # common cannot be with other arch\n[Includes.IA32, PcdsFeatureFlag] # different section name\n')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.File.Remove()

    def testSectionHeader(self):
        if False:
            while True:
                i = 10
        dec = Dec('test.dec', False)
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        self.assertRaises(FatalError, dec._SectionHeaderParser)
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        self.assertRaises(FatalError, dec._SectionHeaderParser)
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        self.assertRaises(FatalError, dec._SectionHeaderParser)
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        self.assertRaises(FatalError, dec._SectionHeaderParser)
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        self.assertRaises(FatalError, dec._SectionHeaderParser)
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        dec._SectionHeaderParser()
        self.assertEqual(len(dec._RawData.CurrentScope), 1)
        self.assertEqual(dec._RawData.CurrentScope[0][0], 'Includes'.upper())
        self.assertEqual(dec._RawData.CurrentScope[0][1], 'IA32')
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        self.assertRaises(FatalError, dec._SectionHeaderParser)
        dec._RawData.CurrentLine = CleanString(dec._RawData.GetNextLine())[0]
        self.assertRaises(FatalError, dec._SectionHeaderParser)

class DecDecCommentTestCase(unittest.TestCase):

    def testDecHeadComment(self):
        if False:
            return 10
        File = TmpFile('test.dec')
        File.Write('# abc\n          ##')
        dec = Dec('test.dec', False)
        dec.ParseDecComment()
        self.assertEqual(len(dec._HeadComment), 2)
        self.assertEqual(dec._HeadComment[0][0], '# abc')
        self.assertEqual(dec._HeadComment[0][1], 1)
        self.assertEqual(dec._HeadComment[1][0], '##')
        self.assertEqual(dec._HeadComment[1][1], 2)
        File.Remove()

    def testNoDoubleComment(self):
        if False:
            print('Hello World!')
        File = TmpFile('test.dec')
        File.Write('# abc\n          #\n          [section_start]')
        dec = Dec('test.dec', False)
        dec.ParseDecComment()
        self.assertEqual(len(dec._HeadComment), 2)
        self.assertEqual(dec._HeadComment[0][0], '# abc')
        self.assertEqual(dec._HeadComment[0][1], 1)
        self.assertEqual(dec._HeadComment[1][0], '#')
        self.assertEqual(dec._HeadComment[1][1], 2)
        File.Remove()
if __name__ == '__main__':
    import Logger.Logger
    Logger.Logger.Initialize()
    unittest.main()