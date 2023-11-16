import unittest
import Logger.Log as Logger
from Library.CommentParsing import ParseHeaderCommentSection, ParseGenericComment, ParseDecPcdGenericComment, ParseDecPcdTailComment
from Library.CommentParsing import _IsCopyrightLine
from Library.StringUtils import GetSplitValueList
from Library.DataType import TAB_SPACE_SPLIT
from Library.DataType import TAB_LANGUAGE_EN_US

class ParseHeaderCommentSectionTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def testNormalCase1(self):
        if False:
            while True:
                i = 10
        TestCommentLines1 = '# License1\n        # License2\n        #\n        ## @file\n        # example abstract\n        #\n        # example description\n        #\n        # Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        #\n        # License3\n        #'
        CommentList = GetSplitValueList(TestCommentLines1, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (Abstract, Description, Copyright, License) = ParseHeaderCommentSection(TestCommentLinesList, 'PhonyFile')
        ExpectedAbstract = 'example abstract'
        self.assertEqual(Abstract, ExpectedAbstract)
        ExpectedDescription = 'example description'
        self.assertEqual(Description, ExpectedDescription)
        ExpectedCopyright = 'Copyright (c) 2007 - 2010, Intel Corporation. All rights reserved.<BR>'
        self.assertEqual(Copyright, ExpectedCopyright)
        ExpectedLicense = 'License1\nLicense2\n\nLicense3'
        self.assertEqual(License, ExpectedLicense)

    def testNormalCase2(self):
        if False:
            print('Hello World!')
        TestCommentLines2 = ' # License1\n        # License2\n        #\n        ## @file\n        # example abstract\n        #\n        # example description\n        #\n        #Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        #\n        ##'
        CommentList = GetSplitValueList(TestCommentLines2, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (Abstract, Description, Copyright, License) = ParseHeaderCommentSection(TestCommentLinesList, 'PhonyFile')
        ExpectedAbstract = 'example abstract'
        self.assertEqual(Abstract, ExpectedAbstract)
        ExpectedDescription = 'example description'
        self.assertEqual(Description, ExpectedDescription)
        ExpectedCopyright = 'Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>'
        self.assertEqual(Copyright, ExpectedCopyright)
        ExpectedLicense = 'License1\nLicense2'
        self.assertEqual(License, ExpectedLicense)

    def testNormalCase3(self):
        if False:
            for i in range(10):
                print('nop')
        TestCommentLines3 = ' # License1\n        # License2\n        #\n        ## @file\n        # Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        #\n        # License3 Line1\n        # License3 Line2\n        ##'
        CommentList = GetSplitValueList(TestCommentLines3, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (Abstract, Description, Copyright, License) = ParseHeaderCommentSection(TestCommentLinesList, 'PhonyFile')
        ExpectedAbstract = ''
        self.assertEqual(Abstract, ExpectedAbstract)
        ExpectedDescription = ''
        self.assertEqual(Description, ExpectedDescription)
        ExpectedCopyright = 'Copyright (c) 2007 - 2010, Intel Corporation. All rights reserved.<BR>'
        self.assertEqual(Copyright, ExpectedCopyright)
        ExpectedLicense = 'License1\nLicense2\n\nLicense3 Line1\nLicense3 Line2'
        self.assertEqual(License, ExpectedLicense)

    def testNormalCase4(self):
        if False:
            for i in range(10):
                print('nop')
        TestCommentLines = '\n        ## @file\n        # Abstract\n        #\n        # Description\n        #\n        # Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        #\n        # License\n        #\n        ##'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (Abstract, Description, Copyright, License) = ParseHeaderCommentSection(TestCommentLinesList, 'PhonyFile')
        ExpectedAbstract = 'Abstract'
        self.assertEqual(Abstract, ExpectedAbstract)
        ExpectedDescription = 'Description'
        self.assertEqual(Description, ExpectedDescription)
        ExpectedCopyright = 'Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>'
        self.assertEqual(Copyright, ExpectedCopyright)
        ExpectedLicense = 'License'
        self.assertEqual(License, ExpectedLicense)

    def testNormalCase5(self):
        if False:
            while True:
                i = 10
        TestCommentLines = '\n        ## @file\n        # Abstract\n        #\n        # Description\n        #\n        # Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        # other line\n        # Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        #\n        # License\n        #\n        ##'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (Abstract, Description, Copyright, License) = ParseHeaderCommentSection(TestCommentLinesList, 'PhonyFile')
        ExpectedAbstract = 'Abstract'
        self.assertEqual(Abstract, ExpectedAbstract)
        ExpectedDescription = 'Description'
        self.assertEqual(Description, ExpectedDescription)
        ExpectedCopyright = 'Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\nCopyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>'
        self.assertEqual(Copyright, ExpectedCopyright)
        ExpectedLicense = 'License'
        self.assertEqual(License, ExpectedLicense)

    def testNormalCase6(self):
        if False:
            print('Hello World!')
        TestCommentLines = '\n        ## @file\n        # Abstract\n        #\n        # Description\n        #\n        # Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        # Copyright (c) 2007 - 2010, FOO1 Corporation. All rights reserved.<BR>\n        # Copyright (c) 2007 - 2010, FOO2 Corporation. All rights reserved.<BR>\n        #\n        # License\n        #\n        ##'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (Abstract, Description, Copyright, License) = ParseHeaderCommentSection(TestCommentLinesList, 'PhonyFile')
        ExpectedAbstract = 'Abstract'
        self.assertEqual(Abstract, ExpectedAbstract)
        ExpectedDescription = 'Description'
        self.assertEqual(Description, ExpectedDescription)
        ExpectedCopyright = 'Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\nCopyright (c) 2007 - 2010, FOO1 Corporation. All rights reserved.<BR>\nCopyright (c) 2007 - 2010, FOO2 Corporation. All rights reserved.<BR>'
        self.assertEqual(Copyright, ExpectedCopyright)
        ExpectedLicense = 'License'
        self.assertEqual(License, ExpectedLicense)

    def testNormalCase7(self):
        if False:
            return 10
        TestCommentLines = '\n        ## @file\n        #\n        # Description\n        #\n        # Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        # Copyright (c) 2007 - 2010, FOO1 Corporation. All rights reserved.<BR>\n        # Copyright (c) 2007 - 2010, FOO2 Corporation. All rights reserved.<BR>\n        #\n        # License\n        #\n        ##'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (Abstract, Description, Copyright, License) = ParseHeaderCommentSection(TestCommentLinesList, 'PhonyFile')
        ExpectedAbstract = ''
        self.assertEqual(Abstract, ExpectedAbstract)
        ExpectedDescription = 'Description'
        self.assertEqual(Description, ExpectedDescription)
        ExpectedCopyright = 'Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\nCopyright (c) 2007 - 2010, FOO1 Corporation. All rights reserved.<BR>\nCopyright (c) 2007 - 2010, FOO2 Corporation. All rights reserved.<BR>'
        self.assertEqual(Copyright, ExpectedCopyright)
        ExpectedLicense = 'License'
        self.assertEqual(License, ExpectedLicense)

    def testNormalCase8(self):
        if False:
            while True:
                i = 10
        TestCommentLines = '\n        ## @file\n        # Abstact\n        #\n        # Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        #\n        # License\n        #\n        ##'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (Abstract, Description, Copyright, License) = ParseHeaderCommentSection(TestCommentLinesList, 'PhonyFile')
        ExpectedAbstract = 'Abstact'
        self.assertEqual(Abstract, ExpectedAbstract)
        ExpectedDescription = ''
        self.assertEqual(Description, ExpectedDescription)
        ExpectedCopyright = 'Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>'
        self.assertEqual(Copyright, ExpectedCopyright)
        ExpectedLicense = 'License'
        self.assertEqual(License, ExpectedLicense)

    def testErrorCase1(self):
        if False:
            for i in range(10):
                print('nop')
        TestCommentLines = '\n        ## @file\n        # Abstract\n        #\n        # Description\n        #\n        # License\n        #\n        ##'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        self.assertRaises(Logger.FatalError, ParseHeaderCommentSection, TestCommentLinesList, 'PhonyFile')

    def testErrorCase2(self):
        if False:
            for i in range(10):
                print('nop')
        TestCommentLines = '\n        ## @file\n        # Abstract\n        #\n        this is invalid line\n        # Description\n        #\n        # Copyright (c) 2007 - 2018, Intel Corporation. All rights reserved.<BR>\n        # License\n        #\n        ##'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        self.assertRaises(Logger.FatalError, ParseHeaderCommentSection, TestCommentLinesList, 'PhonyFile')

class ParseGenericCommentTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def testNormalCase1(self):
        if False:
            return 10
        TestCommentLines = '# hello world'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        HelptxtObj = ParseGenericComment(TestCommentLinesList, 'testNormalCase1')
        self.failIf(not HelptxtObj)
        self.assertEqual(HelptxtObj.GetString(), 'hello world')
        self.assertEqual(HelptxtObj.GetLang(), TAB_LANGUAGE_EN_US)

    def testNormalCase2(self):
        if False:
            for i in range(10):
                print('nop')
        TestCommentLines = '## hello world\n        # second line'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        HelptxtObj = ParseGenericComment(TestCommentLinesList, 'testNormalCase2')
        self.failIf(not HelptxtObj)
        self.assertEqual(HelptxtObj.GetString(), 'hello world\n' + 'second line')
        self.assertEqual(HelptxtObj.GetLang(), TAB_LANGUAGE_EN_US)

    def testNormalCase3(self):
        if False:
            return 10
        TestCommentLines = '## hello world\n        This is not comment line'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        HelptxtObj = ParseGenericComment(TestCommentLinesList, 'testNormalCase3')
        self.failIf(not HelptxtObj)
        self.assertEqual(HelptxtObj.GetString(), 'hello world\n\n')
        self.assertEqual(HelptxtObj.GetLang(), TAB_LANGUAGE_EN_US)

class ParseDecPcdGenericCommentTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def testNormalCase1(self):
        if False:
            return 10
        TestCommentLines = '## hello world\n        # second line'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (HelpTxt, PcdErr) = ParseDecPcdGenericComment(TestCommentLinesList, 'testNormalCase1')
        self.failIf(not HelpTxt)
        self.failIf(PcdErr)
        self.assertEqual(HelpTxt, 'hello world\n' + 'second line')

    def testNormalCase2(self):
        if False:
            for i in range(10):
                print('nop')
        TestCommentLines = '## hello world\n        # second line\n        # @ValidList 1, 2, 3\n        # other line'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (HelpTxt, PcdErr) = ParseDecPcdGenericComment(TestCommentLinesList, 'UnitTest')
        self.failIf(not HelpTxt)
        self.failIf(not PcdErr)
        self.assertEqual(HelpTxt, 'hello world\n' + 'second line\n' + 'other line')
        ExpectedList = GetSplitValueList('1 2 3', TAB_SPACE_SPLIT)
        ActualList = [item for item in GetSplitValueList(PcdErr.GetValidValue(), TAB_SPACE_SPLIT) if item]
        self.assertEqual(ExpectedList, ActualList)
        self.failIf(PcdErr.GetExpression())
        self.failIf(PcdErr.GetValidValueRange())

    def testNormalCase3(self):
        if False:
            i = 10
            return i + 15
        TestCommentLines = '## hello world\n        # second line\n        # @ValidRange LT 1 AND GT 2\n        # other line'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (HelpTxt, PcdErr) = ParseDecPcdGenericComment(TestCommentLinesList, 'UnitTest')
        self.failIf(not HelpTxt)
        self.failIf(not PcdErr)
        self.assertEqual(HelpTxt, 'hello world\n' + 'second line\n' + 'other line')
        self.assertEqual(PcdErr.GetValidValueRange().strip(), 'LT 1 AND GT 2')
        self.failIf(PcdErr.GetExpression())
        self.failIf(PcdErr.GetValidValue())

    def testNormalCase4(self):
        if False:
            print('Hello World!')
        TestCommentLines = '## hello world\n        # second line\n        # @Expression LT 1 AND GT 2\n        # other line'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (HelpTxt, PcdErr) = ParseDecPcdGenericComment(TestCommentLinesList, 'UnitTest')
        self.failIf(not HelpTxt)
        self.failIf(not PcdErr)
        self.assertEqual(HelpTxt, 'hello world\n' + 'second line\n' + 'other line')
        self.assertEqual(PcdErr.GetExpression().strip(), 'LT 1 AND GT 2')
        self.failIf(PcdErr.GetValidValueRange())
        self.failIf(PcdErr.GetValidValue())

    def testNormalCase5(self):
        if False:
            while True:
                i = 10
        TestCommentLines = '# @Expression LT 1 AND GT 2'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (HelpTxt, PcdErr) = ParseDecPcdGenericComment(TestCommentLinesList, 'UnitTest')
        self.failIf(HelpTxt)
        self.failIf(not PcdErr)
        self.assertEqual(PcdErr.GetExpression().strip(), 'LT 1 AND GT 2')
        self.failIf(PcdErr.GetValidValueRange())
        self.failIf(PcdErr.GetValidValue())

    def testNormalCase6(self):
        if False:
            i = 10
            return i + 15
        TestCommentLines = '#'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (HelpTxt, PcdErr) = ParseDecPcdGenericComment(TestCommentLinesList, 'UnitTest')
        self.assertEqual(HelpTxt, '\n')
        self.failIf(PcdErr)

    def testErrorCase1(self):
        if False:
            print('Hello World!')
        TestCommentLines = '## hello world\n        # second line\n        # @ValidList 1, 2, 3\n        # @Expression LT 1 AND GT 2\n        # other line'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        try:
            ParseDecPcdGenericComment(TestCommentLinesList, 'UnitTest')
        except Logger.FatalError:
            pass

class ParseDecPcdTailCommentTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def testNormalCase1(self):
        if False:
            i = 10
            return i + 15
        TestCommentLines = '## #hello world'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (SupModeList, HelpStr) = ParseDecPcdTailComment(TestCommentLinesList, 'UnitTest')
        self.failIf(not HelpStr)
        self.failIf(SupModeList)
        self.assertEqual(HelpStr, 'hello world')

    def testNormalCase2(self):
        if False:
            for i in range(10):
                print('nop')
        TestCommentLines = '## BASE #hello world'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (SupModeList, HelpStr) = ParseDecPcdTailComment(TestCommentLinesList, 'UnitTest')
        self.failIf(not HelpStr)
        self.failIf(not SupModeList)
        self.assertEqual(HelpStr, 'hello world')
        self.assertEqual(SupModeList, ['BASE'])

    def testNormalCase3(self):
        if False:
            return 10
        TestCommentLines = '## BASE  UEFI_APPLICATION #hello world'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (SupModeList, HelpStr) = ParseDecPcdTailComment(TestCommentLinesList, 'UnitTest')
        self.failIf(not HelpStr)
        self.failIf(not SupModeList)
        self.assertEqual(HelpStr, 'hello world')
        self.assertEqual(SupModeList, ['BASE', 'UEFI_APPLICATION'])

    def testNormalCase4(self):
        if False:
            print('Hello World!')
        TestCommentLines = '## BASE  UEFI_APPLICATION'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (SupModeList, HelpStr) = ParseDecPcdTailComment(TestCommentLinesList, 'UnitTest')
        self.failIf(HelpStr)
        self.failIf(not SupModeList)
        self.assertEqual(SupModeList, ['BASE', 'UEFI_APPLICATION'])

    def testNormalCase5(self):
        if False:
            i = 10
            return i + 15
        TestCommentLines = ' # 1 = 128MB, 2 = 256MB, 3 = MAX'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        (SupModeList, HelpStr) = ParseDecPcdTailComment(TestCommentLinesList, 'UnitTest')
        self.failIf(not HelpStr)
        self.assertEqual(HelpStr, '1 = 128MB, 2 = 256MB, 3 = MAX')
        self.failIf(SupModeList)

    def testErrorCase2(self):
        if False:
            while True:
                i = 10
        TestCommentLines = '## BASE INVALID_MODULE_TYPE #hello world'
        CommentList = GetSplitValueList(TestCommentLines, '\n')
        LineNum = 0
        TestCommentLinesList = []
        for Comment in CommentList:
            LineNum += 1
            TestCommentLinesList.append((Comment, LineNum))
        try:
            ParseDecPcdTailComment(TestCommentLinesList, 'UnitTest')
        except Logger.FatalError:
            pass

class _IsCopyrightLineTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def testCase1(self):
        if False:
            print('Hello World!')
        Line = 'this is a copyright ( line'
        Result = _IsCopyrightLine(Line)
        self.failIf(not Result)

    def testCase2(self):
        if False:
            print('Hello World!')
        Line = 'this is a Copyright ( line'
        Result = _IsCopyrightLine(Line)
        self.failIf(not Result)

    def testCase3(self):
        if False:
            while True:
                i = 10
        Line = 'this is not aCopyright ( line'
        Result = _IsCopyrightLine(Line)
        self.failIf(Result)

    def testCase4(self):
        if False:
            print('Hello World!')
        Line = 'this is Copyright( line'
        Result = _IsCopyrightLine(Line)
        self.failIf(not Result)

    def testCase5(self):
        if False:
            i = 10
            return i + 15
        Line = 'this is Copyright         (line'
        Result = _IsCopyrightLine(Line)
        self.failIf(not Result)

    def testCase6(self):
        if False:
            return 10
        Line = 'this is not Copyright line'
        Result = _IsCopyrightLine(Line)
        self.failIf(Result)

    def testCase7(self):
        if False:
            return 10
        Line = 'Copyright (c) line'
        Result = _IsCopyrightLine(Line)
        self.failIf(not Result)

    def testCase8(self):
        if False:
            for i in range(10):
                print('nop')
        Line = ' Copyright (c) line'
        Result = _IsCopyrightLine(Line)
        self.failIf(not Result)

    def testCase9(self):
        if False:
            for i in range(10):
                print('nop')
        Line = 'not a Copyright '
        Result = _IsCopyrightLine(Line)
        self.failIf(Result)
if __name__ == '__main__':
    Logger.Initialize()
    unittest.main()