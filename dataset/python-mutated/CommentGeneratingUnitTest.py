import os
import unittest
import Logger.Log as Logger
from GenMetaFile.GenInfFile import GenGuidSections
from GenMetaFile.GenInfFile import GenProtocolPPiSections
from GenMetaFile.GenInfFile import GenPcdSections
from GenMetaFile.GenInfFile import GenSpecialSections
from Library.CommentGenerating import GenGenericCommentF
from Library.CommentGenerating import _GetHelpStr
from Object.POM.CommonObject import TextObject
from Object.POM.CommonObject import GuidObject
from Object.POM.CommonObject import ProtocolObject
from Object.POM.CommonObject import PpiObject
from Object.POM.CommonObject import PcdObject
from Object.POM.ModuleObject import HobObject
from Library.StringUtils import GetSplitValueList
from Library.DataType import TAB_SPACE_SPLIT
from Library.DataType import TAB_LANGUAGE_EN_US
from Library.DataType import TAB_LANGUAGE_ENG
from Library.DataType import ITEM_UNDEFINED
from Library.DataType import TAB_INF_FEATURE_PCD
from Library import GlobalData
from Library.Misc import CreateDirectory

class _GetHelpStrTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass

    def testNormalCase1(self):
        if False:
            return 10
        HelpStr = 'Hello world'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang(TAB_LANGUAGE_EN_US)
        HelpTextObj.SetString(HelpStr)
        HelpTextList = [HelpTextObj]
        Result = _GetHelpStr(HelpTextList)
        self.assertEqual(Result, HelpStr)

    def testNormalCase2(self):
        if False:
            return 10
        HelpStr = 'Hello world'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang(TAB_LANGUAGE_ENG)
        HelpTextObj.SetString(HelpStr)
        HelpTextList = [HelpTextObj]
        ExpectedStr = 'Hello world1'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang(TAB_LANGUAGE_EN_US)
        HelpTextObj.SetString(ExpectedStr)
        HelpTextList.append(HelpTextObj)
        Result = _GetHelpStr(HelpTextList)
        self.assertEqual(Result, ExpectedStr)

    def testNormalCase3(self):
        if False:
            for i in range(10):
                print('nop')
        HelpStr = 'Hello world'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang('')
        HelpTextObj.SetString(HelpStr)
        HelpTextList = [HelpTextObj]
        ExpectedStr = 'Hello world1'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang(TAB_LANGUAGE_ENG)
        HelpTextObj.SetString(ExpectedStr)
        HelpTextList.append(HelpTextObj)
        Result = _GetHelpStr(HelpTextList)
        self.assertEqual(Result, ExpectedStr)

    def testNormalCase4(self):
        if False:
            return 10
        ExpectedStr = 'Hello world1'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang(TAB_LANGUAGE_ENG)
        HelpTextObj.SetString(ExpectedStr)
        HelpTextList = [HelpTextObj]
        HelpStr = 'Hello world'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang('')
        HelpTextObj.SetString(HelpStr)
        HelpTextList.append(HelpTextObj)
        Result = _GetHelpStr(HelpTextList)
        self.assertEqual(Result, ExpectedStr)

    def testNormalCase5(self):
        if False:
            i = 10
            return i + 15
        ExpectedStr = 'Hello world1'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang(TAB_LANGUAGE_EN_US)
        HelpTextObj.SetString(ExpectedStr)
        HelpTextList = [HelpTextObj]
        HelpStr = 'Hello unknown world'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang('')
        HelpTextObj.SetString(HelpStr)
        HelpTextList.append(HelpTextObj)
        HelpStr = 'Hello mysterious world'
        HelpTextObj = TextObject()
        HelpTextObj.SetLang('')
        HelpTextObj.SetString(HelpStr)
        HelpTextList.append(HelpTextObj)
        Result = _GetHelpStr(HelpTextList)
        self.assertEqual(Result, ExpectedStr)
        HelpTextList.sort()
        self.assertEqual(Result, ExpectedStr)
        HelpTextList.sort(reverse=True)
        self.assertEqual(Result, ExpectedStr)

class GenGuidSectionsTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def GuidFactory(self, CName, FFE, Usage, GuidType, VariableName, HelpStr):
        if False:
            for i in range(10):
                print('nop')
        Guid = GuidObject()
        Guid.SetCName(CName)
        Guid.SetFeatureFlag(FFE)
        Guid.SetGuidTypeList([GuidType])
        Guid.SetUsage(Usage)
        Guid.SetVariableName(VariableName)
        HelpTextObj = TextObject()
        HelpTextObj.SetLang('')
        HelpTextObj.SetString(HelpStr)
        Guid.SetHelpTextList([HelpTextObj])
        return Guid

    def testNormalCase1(self):
        if False:
            i = 10
            return i + 15
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        GuidType = 'Event'
        VariableName = ''
        HelpStr = 'Usage comment line 1'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'CONSUMES'
        GuidType = 'Variable'
        VariableName = ''
        HelpStr = 'Usage comment line 2'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\n## PRODUCES ## Event # Usage comment line 1\n## CONSUMES ## Variable: # Usage comment line 2\nGuid1|FFE1'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase2(self):
        if False:
            while True:
                i = 10
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        GuidType = 'Event'
        VariableName = ''
        HelpStr = 'Usage comment line 1'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        GuidType = 'UNDEFINED'
        VariableName = ''
        HelpStr = 'Generic comment line 1\n Generic comment line 2'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\n## PRODUCES ## Event # Usage comment line 1\n# Generic comment line 1\n# Generic comment line 2\nGuid1|FFE1'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase3(self):
        if False:
            print('Hello World!')
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        GuidType = 'UNDEFINED'
        VariableName = ''
        HelpStr = 'Generic comment'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        GuidType = 'Event'
        VariableName = ''
        HelpStr = 'Usage comment line 1'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\n# Generic comment\n## PRODUCES ## Event # Usage comment line 1\nGuid1|FFE1'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase5(self):
        if False:
            while True:
                i = 10
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        GuidType = 'UNDEFINED'
        VariableName = ''
        HelpStr = 'Generic comment line1 \n generic comment line 2'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\n# Generic comment line1\n# generic comment line 2\nGuid1|FFE1'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase6(self):
        if False:
            i = 10
            return i + 15
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        GuidType = 'Event'
        VariableName = ''
        HelpStr = 'Usage comment line 1\n Usage comment line 2'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\nGuid1|FFE1 ## PRODUCES ## Event # Usage comment line 1  Usage comment line 2\n'
        self.assertEqual(Result.strip(), Expected.strip())

    def testNormalCase7(self):
        if False:
            for i in range(10):
                print('nop')
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        GuidType = 'UNDEFINED'
        VariableName = ''
        HelpStr = 'Usage comment line 1'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\nGuid1|FFE1 # Usage comment line 1\n'
        self.assertEqual(Result.strip(), Expected.strip())

    def testNormalCase8(self):
        if False:
            return 10
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        GuidType = 'Event'
        VariableName = ''
        HelpStr = 'Usage comment line 1\n Usage comment line 2'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        GuidType = 'Event'
        VariableName = ''
        HelpStr = 'Usage comment line 3'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\n## PRODUCES ## Event # Usage comment line 1  Usage comment line 2\n## PRODUCES ## Event # Usage comment line 3\nGuid1|FFE1\n'
        self.assertEqual(Result.strip(), Expected.strip())

    def testNormalCase9(self):
        if False:
            i = 10
            return i + 15
        GuidList = []
        Result = GenGuidSections(GuidList)
        Expected = ''
        self.assertEqual(Result.strip(), Expected.strip())

    def testNormalCase10(self):
        if False:
            for i in range(10):
                print('nop')
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        GuidType = 'UNDEFINED'
        VariableName = ''
        HelpStr = ''
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\nGuid1|FFE1\n'
        self.assertEqual(Result.strip(), Expected.strip())

    def testNormalCase11(self):
        if False:
            i = 10
            return i + 15
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        GuidType = 'UNDEFINED'
        VariableName = ''
        HelpStr = 'general comment line 1'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        GuidType = 'Event'
        VariableName = ''
        HelpStr = 'Usage comment line 3'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        GuidType = 'UNDEFINED'
        VariableName = ''
        HelpStr = 'general comment line 2'
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\n# general comment line 1\n## PRODUCES ## Event # Usage comment line 3\n# general comment line 2\nGuid1|FFE1\n'
        self.assertEqual(Result.strip(), Expected.strip())

    def testNormalCase12(self):
        if False:
            return 10
        GuidList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        GuidType = 'GUID'
        VariableName = ''
        HelpStr = ''
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        GuidType = 'Event'
        VariableName = ''
        HelpStr = ''
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'CONSUMES'
        GuidType = 'Event'
        VariableName = ''
        HelpStr = ''
        Guid1 = self.GuidFactory(CName, FFE, Usage, GuidType, VariableName, HelpStr)
        GuidList.append(Guid1)
        Result = GenGuidSections(GuidList)
        Expected = '[Guids]\n## PRODUCES ## GUID\n## PRODUCES ## Event\n## CONSUMES ## Event\nGuid1|FFE1\n'
        self.assertEqual(Result.strip(), Expected.strip())

class GenProtocolPPiSectionsTest(unittest.TestCase):

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

    def ObjectFactory(self, CName, FFE, Usage, Notify, HelpStr, IsProtocol):
        if False:
            while True:
                i = 10
        if IsProtocol:
            Object = ProtocolObject()
        else:
            Object = PpiObject()
        Object.SetCName(CName)
        Object.SetFeatureFlag(FFE)
        Object.SetUsage(Usage)
        Object.SetNotify(Notify)
        HelpTextObj = TextObject()
        HelpTextObj.SetLang('')
        HelpTextObj.SetString(HelpStr)
        Object.SetHelpTextList([HelpTextObj])
        return Object

    def testNormalCase1(self):
        if False:
            while True:
                i = 10
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        Notify = True
        HelpStr = 'Help'
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## UNDEFINED ## NOTIFY # Help'
        self.assertEqual(Result.strip(), Expected)
        IsProtocol = False
        ObjectList = []
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Ppis]\nGuid1|FFE1 ## UNDEFINED ## NOTIFY # Help'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase2(self):
        if False:
            return 10
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        Notify = True
        HelpStr = ''
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## UNDEFINED ## NOTIFY'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase3(self):
        if False:
            while True:
                i = 10
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        Notify = False
        HelpStr = 'Help'
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## UNDEFINED # Help'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase4(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        Notify = False
        HelpStr = ''
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## UNDEFINED'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase5(self):
        if False:
            return 10
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        Notify = ''
        HelpStr = 'Help'
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 # Help'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase6(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'UNDEFINED'
        Notify = ''
        HelpStr = ''
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase7(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        Notify = True
        HelpStr = 'Help'
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## PRODUCES ## NOTIFY # Help'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase8(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        Notify = True
        HelpStr = ''
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## PRODUCES ## NOTIFY'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase9(self):
        if False:
            while True:
                i = 10
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        Notify = False
        HelpStr = 'Help'
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## PRODUCES # Help'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCaseA(self):
        if False:
            for i in range(10):
                print('nop')
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        Notify = False
        HelpStr = ''
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## PRODUCES'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCaseB(self):
        if False:
            print('Hello World!')
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        Notify = ''
        HelpStr = 'Help'
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## PRODUCES # Help'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCaseC(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        CName = 'Guid1'
        FFE = 'FFE1'
        Usage = 'PRODUCES'
        Notify = ''
        HelpStr = ''
        IsProtocol = True
        Object = self.ObjectFactory(CName, FFE, Usage, Notify, HelpStr, IsProtocol)
        ObjectList.append(Object)
        Result = GenProtocolPPiSections(ObjectList, IsProtocol)
        Expected = '[Protocols]\nGuid1|FFE1 ## PRODUCES'
        self.assertEqual(Result.strip(), Expected)

class GenPcdSectionsTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def ObjectFactory(self, ItemType, TSCName, CName, DValue, FFE, Usage, Str):
        if False:
            for i in range(10):
                print('nop')
        Object = PcdObject()
        HelpStr = Str
        Object.SetItemType(ItemType)
        Object.SetTokenSpaceGuidCName(TSCName)
        Object.SetCName(CName)
        Object.SetDefaultValue(DValue)
        Object.SetFeatureFlag(FFE)
        Object.SetValidUsage(Usage)
        HelpTextObj = TextObject()
        HelpTextObj.SetLang('')
        HelpTextObj.SetString(HelpStr)
        Object.SetHelpTextList([HelpTextObj])
        return Object

    def testNormalCase1(self):
        if False:
            while True:
                i = 10
        ObjectList = []
        ItemType = 'Pcd'
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'UNDEFINED'
        Str = 'Help'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '[Pcd]\n' + 'TSCName.CName|DValue|FFE # Help'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase2(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        ItemType = 'Pcd'
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'UNDEFINED'
        Str = ''
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '[Pcd]\nTSCName.CName|DValue|FFE'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase3(self):
        if False:
            return 10
        ObjectList = []
        ItemType = 'Pcd'
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'CONSUMES'
        Str = 'Help'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '[Pcd]\nTSCName.CName|DValue|FFE ## CONSUMES # Help'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase4(self):
        if False:
            for i in range(10):
                print('nop')
        ObjectList = []
        ItemType = 'Pcd'
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'CONSUMES'
        Str = ''
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '[Pcd]\nTSCName.CName|DValue|FFE ## CONSUMES'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase5(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        ItemType = 'Pcd'
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'CONSUMES'
        Str = 'commment line 1\ncomment line 2'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '[Pcd]\nTSCName.CName|DValue|FFE ## CONSUMES # commment line 1 comment line 2'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase6(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        ItemType = 'Pcd'
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'UNDEFINED'
        Str = 'commment line 1\ncomment line 2'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Usage = 'UNDEFINED'
        Str = 'commment line 3'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '[Pcd]\n# commment line 1\n# comment line 2\n# commment line 3\nTSCName.CName|DValue|FFE'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase7(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        ItemType = 'Pcd'
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'UNDEFINED'
        Str = 'commment line 1\ncomment line 2'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Usage = 'CONSUMES'
        Str = 'Foo'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Usage = 'UNDEFINED'
        Str = 'commment line 3'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '[Pcd]\n# commment line 1\n# comment line 2\n## CONSUMES # Foo\n# commment line 3\nTSCName.CName|DValue|FFE'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase8(self):
        if False:
            for i in range(10):
                print('nop')
        ObjectList = []
        ItemType = TAB_INF_FEATURE_PCD
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'CONSUMES'
        Str = 'commment line 1\ncomment line 2'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '[FeaturePcd]\n# commment line 1\n# comment line 2\nTSCName.CName|DValue|FFE'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase9(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        ItemType = TAB_INF_FEATURE_PCD
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'CONSUMES'
        Str = ''
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '[FeaturePcd]\nTSCName.CName|DValue|FFE'
        self.assertEqual(Result.strip(), Expected)

    def testNormalCase10(self):
        if False:
            i = 10
            return i + 15
        ObjectList = []
        ItemType = TAB_INF_FEATURE_PCD
        TSCName = 'TSCName'
        CName = 'CName'
        DValue = 'DValue'
        FFE = 'FFE'
        Usage = 'PRODUCES'
        Str = 'commment line 1\ncomment line 2'
        Object = self.ObjectFactory(ItemType, TSCName, CName, DValue, FFE, Usage, Str)
        ObjectList.append(Object)
        Result = GenPcdSections(ObjectList)
        Expected = '\n\n[FeaturePcd]\n# commment line 1\n# comment line 2\nTSCName.CName|DValue|FFE\n'
        self.assertEqual(Result, Expected)

class GenHobSectionsTest(unittest.TestCase):

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

    def ObjectFactory(self, SupArchList, Type, Usage, Str):
        if False:
            for i in range(10):
                print('nop')
        Object = HobObject()
        HelpStr = Str
        Object.SetHobType(Type)
        Object.SetUsage(Usage)
        Object.SetSupArchList(SupArchList)
        HelpTextObj = TextObject()
        HelpTextObj.SetLang('')
        HelpTextObj.SetString(HelpStr)
        Object.SetHelpTextList([HelpTextObj])
        return Object

    def testNormalCase1(self):
        if False:
            print('Hello World!')
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = 'Help'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# # Help\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase2(self):
        if False:
            for i in range(10):
                print('nop')
        ObjectList = []
        SupArchList = []
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = 'Help'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob]\n# ##\n# # Help\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase3(self):
        if False:
            for i in range(10):
                print('nop')
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = '\nComment Line 1\n\n'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# # Comment Line 1\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase4(self):
        if False:
            for i in range(10):
                print('nop')
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = '\nComment Line 1\n'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# # Comment Line 1\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase5(self):
        if False:
            return 10
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = 'Comment Line 1\n\n'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# # Comment Line 1\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase6(self):
        if False:
            while True:
                i = 10
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = ''
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase7(self):
        if False:
            return 10
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = '\nNew Stack HoB'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# # New Stack HoB\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase8(self):
        if False:
            while True:
                i = 10
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = '\nNew Stack HoB\n\nTail Comment'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# # New Stack HoB\n# #\n# # Tail Comment\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase9(self):
        if False:
            return 10
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = '\n\n'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# #\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase10(self):
        if False:
            for i in range(10):
                print('nop')
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = '\n'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# #\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase11(self):
        if False:
            while True:
                i = 10
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = '\n\n\n'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# #\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase12(self):
        if False:
            while True:
                i = 10
        ObjectList = []
        SupArchList = ['X64']
        Type = 'Foo'
        Usage = 'UNDEFINED'
        Str = '\n\n\n\n'
        Object = self.ObjectFactory(SupArchList, Type, Usage, Str)
        ObjectList.append(Object)
        Result = GenSpecialSections(ObjectList, 'Hob')
        Expected = '# [Hob.X64]\n# ##\n# #\n# #\n# #\n# Foo ## UNDEFINED\n#\n#\n'
        self.assertEqual(Result, Expected)

class GenGenericCommentFTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def testNormalCase1(self):
        if False:
            print('Hello World!')
        CommentLines = 'Comment Line 1'
        Result = GenGenericCommentF(CommentLines)
        Expected = '# Comment Line 1\n'
        self.assertEqual(Result, Expected)

    def testNormalCase2(self):
        if False:
            print('Hello World!')
        CommentLines = '\n'
        Result = GenGenericCommentF(CommentLines)
        Expected = '#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase3(self):
        if False:
            return 10
        CommentLines = '\n\n\n'
        Result = GenGenericCommentF(CommentLines)
        Expected = '#\n#\n#\n'
        self.assertEqual(Result, Expected)

    def testNormalCase4(self):
        if False:
            while True:
                i = 10
        CommentLines = 'coment line 1\n'
        Result = GenGenericCommentF(CommentLines)
        Expected = '# coment line 1\n'
        self.assertEqual(Result, Expected)

    def testNormalCase5(self):
        if False:
            return 10
        CommentLines = 'coment line 1\n coment line 2\n'
        Result = GenGenericCommentF(CommentLines)
        Expected = '# coment line 1\n# coment line 2\n'
        self.assertEqual(Result, Expected)
if __name__ == '__main__':
    Logger.Initialize()
    unittest.main()