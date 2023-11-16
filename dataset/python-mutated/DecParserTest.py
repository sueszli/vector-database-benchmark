from __future__ import print_function
import os
import unittest
from Parser.DecParserMisc import IsValidCArray, IsValidPcdDatum
from Parser.DecParser import Dec
from Library.ParserValidate import IsValidCFormatGuid

def TestToolFuncs():
    if False:
        i = 10
        return i + 15
    assert IsValidCArray('{0x1, 0x23}')
    assert not IsValidCArray('{0x1, 0x23, }')
    assert not IsValidCArray('{0x1, 0x2345}')
    assert not IsValidCArray('{0x1, 0x23, ')
    assert not IsValidCArray('{0x1, 0x2 3, }')
    assert IsValidPcdDatum('VOID*', '"test"')[0]
    assert IsValidPcdDatum('VOID*', 'L"test"')[0]
    assert IsValidPcdDatum('BOOLEAN', 'TRUE')[0]
    assert IsValidPcdDatum('BOOLEAN', 'FALSE')[0]
    assert IsValidPcdDatum('BOOLEAN', '0')[0]
    assert IsValidPcdDatum('BOOLEAN', '1')[0]
    assert IsValidPcdDatum('UINT8', '0xab')[0]
    assert not IsValidPcdDatum('UNKNOWNTYPE', '0xabc')[0]
    assert not IsValidPcdDatum('UINT8', 'not number')[0]
    assert IsValidCFormatGuid('{ 0xfa0b1735 , 0x87a0, 0x4193, {0xb2, 0x66 , 0x53, 0x8c , 0x38, 0xaf, 0x48, 0xce }}')
    assert not IsValidCFormatGuid('{ 0xfa0b1735 , 0x87a0, 0x4193, {0xb2, 0x66 , 0x53, 0x8c , 0x38, 0xaf, 0x48, 0xce }} 0xaa')

def TestTemplate(TestString, TestFunc):
    if False:
        print('Hello World!')
    Path = os.path.join(os.getcwd(), 'test.dec')
    Path = os.path.normpath(Path)
    try:
        f = open(Path, 'w')
        f.write(TestString)
        f.close()
    except:
        print('Can not create temporary file [%s]!' % Path)
        exit(-1)
    Ret = TestFunc(Path, TestString)
    os.remove(Path)
    return Ret

def TestOK(Path, TestString):
    if False:
        return 10
    try:
        Parser = Dec(Path)
    except:
        raise 'Bug!!! Correct syntax in DEC file, but exception raised!\n' + TestString
    return Parser

def TestError(Path, TestString):
    if False:
        for i in range(10):
            print('nop')
    try:
        Dec(Path)
    except:
        return True
    raise 'Bug!!! Wrong syntax in DEC file, but passed by DEC parser!!\n' + TestString

def TestDecDefine():
    if False:
        return 10
    TestString = '\n    [Defines]\n      DEC_SPECIFICATION              = 0x00010005\n      PACKAGE_NAME                   = MdePkg\n      PACKAGE_GUID                   = 1E73767F-8F52-4603-AEB4-F29B510B6766\n      PACKAGE_VERSION                = 1.02\n    '
    Parser = TestTemplate(TestString, TestOK)
    DefObj = Parser.GetDefineSectionObject()
    assert DefObj.GetPackageSpecification() == '0x00010005'
    assert DefObj.GetPackageName() == 'MdePkg'
    assert DefObj.GetPackageGuid() == '1E73767F-8F52-4603-AEB4-F29B510B6766'
    assert DefObj.GetPackageVersion() == '1.02'
    TestString = '\n    [Defines]\n      UNKNOW_KEY              = 0x00010005 # A unknown key\n    '
    assert TestTemplate(TestString, TestError)
    TestString = '\n    [Defines]\n      PACKAGE_GUID                   = F-8F52-4603-AEB4-F29B510B6766 # Error GUID\n    '
    assert TestTemplate(TestString, TestError)

def TestDecInclude():
    if False:
        for i in range(10):
            print('nop')
    TestString = '\n    [Defines]\n      DEC_SPECIFICATION              = 0x00010005\n      PACKAGE_NAME                   = MdePkg\n      PACKAGE_GUID                   = 1E73767F-8F52-4603-AEB4-F29B510B6766\n      PACKAGE_VERSION                = 1.02\n    [ \\\n    Includes]\n      Include\n    [Includes.IA32]\n      Include/Ia32\n    '
    try:
        os.makedirs('Include/Ia32')
    except:
        pass
    Parser = TestTemplate(TestString, TestOK)
    IncObj = Parser.GetIncludeSectionObject()
    Items = IncObj.GetIncludes()
    assert len(Items) == 1
    assert Items[0].File == 'Include'
    Items = IncObj.GetIncludes('IA32')
    assert len(Items) == 1
    assert Items[0].File == 'Include\\Ia32'
    TestString = '\n    [Defines]\n      DEC_SPECIFICATION              = 0x00010005\n      PACKAGE_NAME                   = MdePkg\n      PACKAGE_GUID                   = 1E73767F-8F52-4603-AEB4-F29B510B6766\n      PACKAGE_VERSION                = 1.02\n    [Includes]\n      Include_not_exist  # directory does not exist\n    '
    assert TestTemplate(TestString, TestError)
    os.removedirs('Include/Ia32')

def TestDecGuidPpiProtocol():
    if False:
        for i in range(10):
            print('nop')
    TestString = '\n    [Defines]\n      DEC_SPECIFICATION              = 0x00010005\n      PACKAGE_NAME                   = MdePkg\n      PACKAGE_GUID                   = 1E73767F-8F52-4603-AEB4-F29B510B6766\n      PACKAGE_VERSION                = 1.02\n    [Guids]\n      #\n      # GUID defined in UEFI2.1/UEFI2.0/EFI1.1\n      #\n      ## Include/Guid/GlobalVariable.h\n      gEfiGlobalVariableGuid         = { 0x8BE4DF61, 0x93CA, 0x11D2, { 0xAA, 0x0D, 0x00, 0xE0, 0x98, 0x03, 0x2B, 0x8C }}\n    [Protocols]\n      ## Include/Protocol/Bds.h\n      gEfiBdsArchProtocolGuid        = { 0x665E3FF6, 0x46CC, 0x11D4, { 0x9A, 0x38, 0x00, 0x90, 0x27, 0x3F, 0xC1, 0x4D }}\n    [Ppis]\n      ## Include/Ppi/MasterBootMode.h\n      gEfiPeiMasterBootModePpiGuid = { 0x7408d748, 0xfc8c, 0x4ee6, {0x92, 0x88, 0xc4, 0xbe, 0xc0, 0x92, 0xa4, 0x10 } }\n    '
    Parser = TestTemplate(TestString, TestOK)
    Obj = Parser.GetGuidSectionObject()
    Items = Obj.GetGuids()
    assert Obj.GetSectionName() == 'Guids'.upper()
    assert len(Items) == 1
    assert Items[0].GuidCName == 'gEfiGlobalVariableGuid'
    assert Items[0].GuidCValue == '{ 0x8BE4DF61, 0x93CA, 0x11D2, { 0xAA, 0x0D, 0x00, 0xE0, 0x98, 0x03, 0x2B, 0x8C }}'
    Obj = Parser.GetProtocolSectionObject()
    Items = Obj.GetProtocols()
    assert Obj.GetSectionName() == 'Protocols'.upper()
    assert len(Items) == 1
    assert Items[0].GuidCName == 'gEfiBdsArchProtocolGuid'
    assert Items[0].GuidCValue == '{ 0x665E3FF6, 0x46CC, 0x11D4, { 0x9A, 0x38, 0x00, 0x90, 0x27, 0x3F, 0xC1, 0x4D }}'
    Obj = Parser.GetPpiSectionObject()
    Items = Obj.GetPpis()
    assert Obj.GetSectionName() == 'Ppis'.upper()
    assert len(Items) == 1
    assert Items[0].GuidCName == 'gEfiPeiMasterBootModePpiGuid'
    assert Items[0].GuidCValue == '{ 0x7408d748, 0xfc8c, 0x4ee6, {0x92, 0x88, 0xc4, 0xbe, 0xc0, 0x92, 0xa4, 0x10 } }'

def TestDecPcd():
    if False:
        return 10
    TestString = '\n    [Defines]\n      DEC_SPECIFICATION              = 0x00010005\n      PACKAGE_NAME                   = MdePkg\n      PACKAGE_GUID                   = 1E73767F-8F52-4603-AEB4-F29B510B6766\n      PACKAGE_VERSION                = 1.02\n    [PcdsFeatureFlag]\n      ## If TRUE, the component name protocol will not be installed.\n      gEfiMdePkgTokenSpaceGuid.PcdComponentNameDisable|FALSE|BOOLEAN|0x0000000d\n\n    [PcdsFixedAtBuild]\n      ## Indicates the maximum length of unicode string\n      gEfiMdePkgTokenSpaceGuid.PcdMaximumUnicodeStringLength|1000000|UINT32|0x00000001\n\n    [PcdsFixedAtBuild.IPF]\n      ## The base address of IO port space for IA64 arch\n      gEfiMdePkgTokenSpaceGuid.PcdIoBlockBaseAddressForIpf|0x0ffffc000000|UINT64|0x0000000f\n\n    [PcdsFixedAtBuild,PcdsPatchableInModule]\n      ## This flag is used to control the printout of DebugLib\n      gEfiMdePkgTokenSpaceGuid.PcdDebugPrintErrorLevel|0x80000000|UINT32|0x00000006\n\n    [PcdsFixedAtBuild,PcdsPatchableInModule,PcdsDynamic]\n      ## This value is used to set the base address of pci express hierarchy\n      gEfiMdePkgTokenSpaceGuid.PcdPciExpressBaseAddress|0xE0000000|UINT64|0x0000000a\n    '
    Parser = TestTemplate(TestString, TestOK)
    Obj = Parser.GetPcdSectionObject()
    Items = Obj.GetPcds('PcdsFeatureFlag', 'COMMON')
    assert len(Items) == 1
    assert Items[0].TokenSpaceGuidCName == 'gEfiMdePkgTokenSpaceGuid'
    assert Items[0].TokenCName == 'PcdComponentNameDisable'
    assert Items[0].DefaultValue == 'FALSE'
    assert Items[0].DatumType == 'BOOLEAN'
    assert Items[0].TokenValue == '0x0000000d'
    Items = Obj.GetPcdsByType('PcdsFixedAtBuild')
    assert len(Items) == 4
    assert len(Obj.GetPcdsByType('PcdsPatchableInModule')) == 2

def TestDecUserExtension():
    if False:
        print('Hello World!')
    TestString = '\n    [Defines]\n      DEC_SPECIFICATION              = 0x00010005\n      PACKAGE_NAME                   = MdePkg\n      PACKAGE_GUID                   = 1E73767F-8F52-4603-AEB4-F29B510B6766\n      PACKAGE_VERSION                = 1.02\n    [UserExtensions.MyID."TestString".IA32]\n      Some Strings...\n    '
    Parser = TestTemplate(TestString, TestOK)
    Obj = Parser.GetUserExtensionSectionObject()
    Items = Obj.GetAllUserExtensions()
    assert len(Items) == 1
    assert Items[0].UserString == 'Some Strings...'
    assert len(Items[0].ArchAndModuleType) == 1
    assert ['MyID', '"TestString"', 'IA32'] in Items[0].ArchAndModuleType
if __name__ == '__main__':
    import Logger.Logger
    Logger.Logger.Initialize()
    unittest.FunctionTestCase(TestToolFuncs).runTest()
    unittest.FunctionTestCase(TestDecDefine).runTest()
    unittest.FunctionTestCase(TestDecInclude).runTest()
    unittest.FunctionTestCase(TestDecGuidPpiProtocol).runTest()
    unittest.FunctionTestCase(TestDecPcd).runTest()
    unittest.FunctionTestCase(TestDecUserExtension).runTest()
    print('All tests passed...')