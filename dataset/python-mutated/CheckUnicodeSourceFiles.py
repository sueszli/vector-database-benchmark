import os
import unittest
import codecs
import TestTools
from Common.Misc import PathClass
import AutoGen.UniClassObject as BtUni
from Common import EdkLogger
EdkLogger.InitializeForUnitTest()

class Tests(TestTools.BaseToolsTest):
    SampleData = u'\n        #langdef en-US "English"\n        #string STR_A #language en-US "STR_A for en-US"\n    '

    def EncodeToFile(self, encoding, string=None):
        if False:
            print('Hello World!')
        if string is None:
            string = self.SampleData
        if encoding is not None:
            data = codecs.encode(string, encoding)
        else:
            data = string
        path = 'input.uni'
        self.WriteTmpFile(path, data)
        return PathClass(self.GetTmpFilePath(path))

    def ErrorFailure(self, error, encoding, shouldPass):
        if False:
            while True:
                i = 10
        msg = error + ' should '
        if shouldPass:
            msg += 'not '
        msg += 'be generated for '
        msg += '%s data in a .uni file' % encoding
        self.fail(msg)

    def UnicodeErrorFailure(self, encoding, shouldPass):
        if False:
            i = 10
            return i + 15
        self.ErrorFailure('UnicodeError', encoding, shouldPass)

    def EdkErrorFailure(self, encoding, shouldPass):
        if False:
            while True:
                i = 10
        self.ErrorFailure('EdkLogger.FatalError', encoding, shouldPass)

    def CheckFile(self, encoding, shouldPass, string=None):
        if False:
            i = 10
            return i + 15
        path = self.EncodeToFile(encoding, string)
        try:
            BtUni.UniFileClassObject([path])
            if shouldPass:
                return
        except UnicodeError:
            if not shouldPass:
                return
            else:
                self.UnicodeErrorFailure(encoding, shouldPass)
        except EdkLogger.FatalError:
            if not shouldPass:
                return
            else:
                self.EdkErrorFailure(encoding, shouldPass)
        except Exception:
            pass
        self.EdkErrorFailure(encoding, shouldPass)

    def testUtf16InUniFile(self):
        if False:
            i = 10
            return i + 15
        self.CheckFile('utf_16', shouldPass=True)

    def testSupplementaryPlaneUnicodeCharInUtf16File(self):
        if False:
            for i in range(10):
                print('nop')
        data = u'\n            #langdef en-US "English"\n            #string STR_A #language en-US "CodePoint (𐌀) > 0xFFFF"\n        '
        self.CheckFile('utf_16', shouldPass=False, string=data)

    def testSurrogatePairUnicodeCharInUtf16File(self):
        if False:
            while True:
                i = 10
        data = codecs.BOM_UTF16_LE + b'//\x01\xd8 '
        self.CheckFile(encoding=None, shouldPass=False, string=data)

    def testValidUtf8File(self):
        if False:
            print('Hello World!')
        self.CheckFile(encoding='utf_8', shouldPass=True)

    def testValidUtf8FileWithBom(self):
        if False:
            print('Hello World!')
        data = codecs.BOM_UTF8 + codecs.encode(self.SampleData, 'utf_8')
        self.CheckFile(encoding=None, shouldPass=True, string=data)

    def test32bitUnicodeCharInUtf8File(self):
        if False:
            while True:
                i = 10
        data = u'\n            #langdef en-US "English"\n            #string STR_A #language en-US "CodePoint (𐌀) > 0xFFFF"\n        '
        self.CheckFile('utf_16', shouldPass=False, string=data)

    def test32bitUnicodeCharInUtf8File(self):
        if False:
            while True:
                i = 10
        data = u'\n            #langdef en-US "English"\n            #string STR_A #language en-US "CodePoint (𐌀) > 0xFFFF"\n        '
        self.CheckFile('utf_8', shouldPass=False, string=data)

    def test32bitUnicodeCharInUtf8Comment(self):
        if False:
            for i in range(10):
                print('nop')
        data = u'\n            // Even in comments, we reject non-UCS-2 chars: 𐌀\n            #langdef en-US "English"\n            #string STR_A #language en-US "A"\n        '
        self.CheckFile('utf_8', shouldPass=False, string=data)

    def testSurrogatePairUnicodeCharInUtf8File(self):
        if False:
            print('Hello World!')
        data = b'\xed\xa0\x81'
        self.CheckFile(encoding=None, shouldPass=False, string=data)

    def testSurrogatePairUnicodeCharInUtf8FileWithBom(self):
        if False:
            i = 10
            return i + 15
        data = codecs.BOM_UTF8 + b'\xed\xa0\x81'
        self.CheckFile(encoding=None, shouldPass=False, string=data)
TheTestSuite = TestTools.MakeTheTestSuite(locals())
if __name__ == '__main__':
    allTests = TheTestSuite()
    unittest.TextTestRunner().run(allTests)