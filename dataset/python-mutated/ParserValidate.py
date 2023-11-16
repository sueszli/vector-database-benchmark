"""
ParserValidate
"""
import os.path
import re
import platform
from Library.DataType import MODULE_LIST
from Library.DataType import COMPONENT_TYPE_LIST
from Library.DataType import PCD_USAGE_TYPE_LIST_OF_MODULE
from Library.DataType import TAB_SPACE_SPLIT
from Library.StringUtils import GetSplitValueList
from Library.ExpressionValidate import IsValidBareCString
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Common.MultipleWorkspace import MultipleWorkspace as mws

def __HexDigit(TempChar):
    if False:
        while True:
            i = 10
    if TempChar >= 'a' and TempChar <= 'f' or (TempChar >= 'A' and TempChar <= 'F') or (TempChar >= '0' and TempChar <= '9'):
        return True
    else:
        return False

def IsValidHex(HexStr):
    if False:
        for i in range(10):
            print('nop')
    if not HexStr.upper().startswith('0X'):
        return False
    CharList = [c for c in HexStr[2:] if not __HexDigit(c)]
    if len(CharList) == 0:
        return True
    else:
        return False

def IsValidBoolType(BoolString):
    if False:
        while True:
            i = 10
    if BoolString == 'TRUE' or BoolString == 'True' or BoolString == 'true' or (BoolString == '0x1') or (BoolString == '0x01'):
        return True
    elif BoolString == 'FALSE' or BoolString == 'False' or BoolString == 'false' or (BoolString == '0x0') or (BoolString == '0x00'):
        return True
    else:
        return False

def IsValidInfMoudleTypeList(ModuleTypeList):
    if False:
        i = 10
        return i + 15
    for ModuleType in ModuleTypeList:
        return IsValidInfMoudleType(ModuleType)

def IsValidInfMoudleType(ModuleType):
    if False:
        i = 10
        return i + 15
    if ModuleType in MODULE_LIST:
        return True
    else:
        return False

def IsValidInfComponentType(ComponentType):
    if False:
        i = 10
        return i + 15
    if ComponentType.upper() in COMPONENT_TYPE_LIST:
        return True
    else:
        return False

def IsValidToolFamily(ToolFamily):
    if False:
        return 10
    ReIsValidFamily = re.compile('^[A-Z]+[A-Za-z0-9]{0,}$', re.DOTALL)
    if ReIsValidFamily.match(ToolFamily) is None:
        return False
    return True

def IsValidToolTagName(TagName):
    if False:
        return 10
    if TagName.strip() == '':
        return True
    if TagName.strip() == '*':
        return True
    if not IsValidWord(TagName):
        return False
    return True

def IsValidArch(Arch):
    if False:
        for i in range(10):
            print('nop')
    if Arch == 'common':
        return True
    ReIsValidArch = re.compile('^[a-zA-Z]+[a-zA-Z0-9]{0,}$', re.DOTALL)
    if ReIsValidArch.match(Arch) is None:
        return False
    return True

def IsValidFamily(Family):
    if False:
        return 10
    Family = Family.strip()
    if Family == '*':
        return True
    if Family == '':
        return True
    ReIsValidFamily = re.compile('^[A-Z]+[A-Za-z0-9]{0,}$', re.DOTALL)
    if ReIsValidFamily.match(Family) is None:
        return False
    return True

def IsValidBuildOptionName(BuildOptionName):
    if False:
        while True:
            i = 10
    if not BuildOptionName:
        return False
    ToolOptionList = GetSplitValueList(BuildOptionName, '_', 4)
    if len(ToolOptionList) != 5:
        return False
    ReIsValidBuildOption1 = re.compile('^\\s*(\\*)|([A-Z][a-zA-Z0-9]*)$')
    ReIsValidBuildOption2 = re.compile('^\\s*(\\*)|([a-zA-Z][a-zA-Z0-9]*)$')
    if ReIsValidBuildOption1.match(ToolOptionList[0]) is None:
        return False
    if ReIsValidBuildOption1.match(ToolOptionList[1]) is None:
        return False
    if ReIsValidBuildOption2.match(ToolOptionList[2]) is None:
        return False
    if ToolOptionList[3] == '*' and ToolOptionList[4] not in ['FAMILY', 'DLL', 'DPATH']:
        return False
    return True

def IsValidToken(ReString, Token):
    if False:
        print('Hello World!')
    Match = re.compile(ReString).match(Token)
    return Match and Match.start() == 0 and (Match.end() == len(Token))

def IsValidPath(Path, Root):
    if False:
        i = 10
        return i + 15
    Path = Path.strip()
    OrigPath = Path.replace('\\', '/')
    Path = os.path.normpath(Path).replace('\\', '/')
    Root = os.path.normpath(Root).replace('\\', '/')
    FullPath = mws.join(Root, Path)
    if not os.path.exists(FullPath):
        return False
    if os.path.isabs(Path):
        if not Path.startswith(Root):
            return False
        return True
    for Rel in ['/', './', '../']:
        if OrigPath.startswith(Rel):
            return False
    for Rel in ['//', '/./', '/../']:
        if Rel in OrigPath:
            return False
    for Rel in ['/.', '/..', '/']:
        if OrigPath.endswith(Rel):
            return False
    Path = Path.rstrip('/')
    for Word in Path.split('/'):
        if not IsValidWord(Word):
            return False
    return True

def IsValidInstallPath(Path):
    if False:
        print('Hello World!')
    if platform.platform().find('Windows') >= 0:
        if os.path.isabs(Path):
            return False
    else:
        if Path[1:2] == ':':
            return False
        if os.path.isabs(Path):
            return False
    if Path.startswith('.'):
        return False
    if Path.find('..') != -1:
        return False
    return True

def IsValidCFormatGuid(Guid):
    if False:
        print('Hello World!')
    List = ['{', 10, ',', 6, ',', 6, ',{', 4, ',', 4, ',', 4, ',', 4, ',', 4, ',', 4, ',', 4, ',', 4, '}}']
    Index = 0
    Value = ''
    SepValue = ''
    for Char in Guid:
        if Char not in '{},\t ':
            Value += Char
            continue
        if Value:
            try:
                if not SepValue or SepValue != List[Index]:
                    return False
                Index += 1
                SepValue = ''
                if not Value.startswith('0x') and (not Value.startswith('0X')):
                    return False
                if not isinstance(List[Index], type(1)) or len(Value) > List[Index] or len(Value) < 3:
                    return False
                int(Value, 16)
            except BaseException:
                return False
            Value = ''
            Index += 1
        if Char in '{},':
            SepValue += Char
    return SepValue == '}}' and Value == ''

def IsValidPcdType(PcdTypeString):
    if False:
        print('Hello World!')
    if PcdTypeString.upper() in PCD_USAGE_TYPE_LIST_OF_MODULE:
        return True
    else:
        return False

def IsValidWord(Word):
    if False:
        while True:
            i = 10
    if not Word:
        return False
    if not Word[0].isalnum() and (not Word[0] == '_') and (not Word[0].isdigit()):
        return False
    LastChar = ''
    for Char in Word[1:]:
        if not Char.isalpha() and (not Char.isdigit()) and (Char != '-') and (Char != '_') and (Char != '.'):
            return False
        if Char == '.' and LastChar == '.':
            return False
        LastChar = Char
    return True

def IsValidSimpleWord(Word):
    if False:
        while True:
            i = 10
    ReIsValidSimpleWord = re.compile('^[0-9A-Za-z][0-9A-Za-z\\-_]*$', re.DOTALL)
    Word = Word.strip()
    if not Word:
        return False
    if not ReIsValidSimpleWord.match(Word):
        return False
    return True

def IsValidDecVersion(Word):
    if False:
        print('Hello World!')
    if Word.find('.') > -1:
        ReIsValidDecVersion = re.compile('[0-9]+\\.?[0-9]+$')
    else:
        ReIsValidDecVersion = re.compile('[0-9]+$')
    if ReIsValidDecVersion.match(Word) is None:
        return False
    return True

def IsValidHexVersion(Word):
    if False:
        while True:
            i = 10
    ReIsValidHexVersion = re.compile('[0][xX][0-9A-Fa-f]{8}$', re.DOTALL)
    if ReIsValidHexVersion.match(Word) is None:
        return False
    return True

def IsValidBuildNumber(Word):
    if False:
        while True:
            i = 10
    ReIsValieBuildNumber = re.compile('[0-9]{1,4}$', re.DOTALL)
    if ReIsValieBuildNumber.match(Word) is None:
        return False
    return True

def IsValidDepex(Word):
    if False:
        return 10
    Index = Word.upper().find('PUSH')
    if Index > -1:
        return IsValidCFormatGuid(Word[Index + 4:].strip())
    ReIsValidCName = re.compile('^[A-Za-z_][0-9A-Za-z_\\s\\.]*$', re.DOTALL)
    if ReIsValidCName.match(Word) is None:
        return False
    return True

def IsValidNormalizedString(String):
    if False:
        print('Hello World!')
    if String == '':
        return True
    for Char in String:
        if Char == '\t':
            return False
    StringList = GetSplitValueList(String, TAB_SPACE_SPLIT)
    for Item in StringList:
        if not Item:
            continue
        if not IsValidWord(Item):
            return False
    return True

def IsValidIdString(String):
    if False:
        while True:
            i = 10
    if IsValidSimpleWord(String.strip()):
        return True
    if String.strip().startswith('"') and String.strip().endswith('"'):
        String = String[1:-1]
        if String.strip() == '':
            return True
        if IsValidNormalizedString(String):
            return True
    return False

def IsValidVersionString(VersionString):
    if False:
        while True:
            i = 10
    VersionString = VersionString.strip()
    for Char in VersionString:
        if not (Char >= 33 and Char <= 126):
            return False
    return True

def IsValidPcdValue(PcdValue):
    if False:
        print('Hello World!')
    for Char in PcdValue:
        if Char == '\n' or Char == '\t' or Char == '\x0c':
            return False
    if IsValidFeatureFlagExp(PcdValue, True)[0]:
        return True
    if IsValidHex(PcdValue):
        return True
    ReIsValidIntegerSingle = re.compile('^\\s*[0-9]\\s*$', re.DOTALL)
    if ReIsValidIntegerSingle.match(PcdValue) is not None:
        return True
    ReIsValidIntegerMulti = re.compile('^\\s*[1-9][0-9]+\\s*$', re.DOTALL)
    if ReIsValidIntegerMulti.match(PcdValue) is not None:
        return True
    ReIsValidStringType = re.compile('^\\s*[\\"L].*[\\"]\\s*$')
    if ReIsValidStringType.match(PcdValue):
        IsTrue = False
        if PcdValue.strip().startswith('L"'):
            StringValue = PcdValue.strip().lstrip('L"').rstrip('"')
            if IsValidBareCString(StringValue):
                IsTrue = True
        elif PcdValue.strip().startswith('"'):
            StringValue = PcdValue.strip().lstrip('"').rstrip('"')
            if IsValidBareCString(StringValue):
                IsTrue = True
        if IsTrue:
            return IsTrue
    if IsValidCFormatGuid(PcdValue):
        return True
    ReIsValidByteHex = re.compile('^\\s*0x[0-9a-fA-F]{1,2}\\s*$', re.DOTALL)
    if PcdValue.strip().startswith('{') and PcdValue.strip().endswith('}'):
        StringValue = PcdValue.strip().lstrip('{').rstrip('}')
        ValueList = StringValue.split(',')
        AllValidFlag = True
        for ValueItem in ValueList:
            if not ReIsValidByteHex.match(ValueItem.strip()):
                AllValidFlag = False
        if AllValidFlag:
            return True
    AllValidFlag = True
    ValueList = PcdValue.split(',')
    for ValueItem in ValueList:
        if not ReIsValidByteHex.match(ValueItem.strip()):
            AllValidFlag = False
    if AllValidFlag:
        return True
    return False

def IsValidCVariableName(CName):
    if False:
        i = 10
        return i + 15
    ReIsValidCName = re.compile('^[A-Za-z_][0-9A-Za-z_]*$', re.DOTALL)
    if ReIsValidCName.match(CName) is None:
        return False
    return True

def IsValidIdentifier(Ident):
    if False:
        print('Hello World!')
    ReIdent = re.compile('^[A-Za-z_][0-9A-Za-z_]*$', re.DOTALL)
    if ReIdent.match(Ident) is None:
        return False
    return True

def IsValidDecVersionVal(Ver):
    if False:
        for i in range(10):
            print('nop')
    ReVersion = re.compile('[0-9]+(\\.[0-9]{1,2})$')
    if ReVersion.match(Ver) is None:
        return False
    return True

def IsValidLibName(LibName):
    if False:
        for i in range(10):
            print('nop')
    if LibName == 'NULL':
        return False
    ReLibName = re.compile('^[A-Z]+[a-zA-Z0-9]*$')
    if not ReLibName.match(LibName):
        return False
    return True

def IsValidUserId(UserId):
    if False:
        print('Hello World!')
    UserId = UserId.strip()
    Quoted = False
    if UserId.startswith('"') and UserId.endswith('"'):
        Quoted = True
        UserId = UserId[1:-1]
    if not UserId or not UserId[0].isalpha():
        return False
    for Char in UserId[1:]:
        if not Char.isalnum() and (not Char in '_.'):
            return False
        if Char == '.' and (not Quoted):
            return False
    return True

def CheckUTF16FileHeader(File):
    if False:
        for i in range(10):
            print('nop')
    FileIn = open(File, 'rb').read(2)
    if FileIn != b'\xff\xfe':
        return False
    return True