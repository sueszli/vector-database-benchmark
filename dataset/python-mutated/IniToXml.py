"""
IniToXml
"""
import os.path
import re
from time import strftime
from time import localtime
import Logger.Log as Logger
from Logger.ToolError import UPT_INI_PARSE_ERROR
from Logger.ToolError import FILE_NOT_FOUND
from Library.Xml.XmlRoutines import CreateXmlElement
from Library.DataType import TAB_VALUE_SPLIT
from Library.DataType import TAB_EQUAL_SPLIT
from Library.DataType import TAB_SECTION_START
from Library.DataType import TAB_SECTION_END
from Logger import StringTable as ST
from Library.StringUtils import ConvertSpecialChar
from Library.ParserValidate import IsValidPath
from Library import GlobalData

def IniParseError(Error, File, Line):
    if False:
        while True:
            i = 10
    Logger.Error('UPT', UPT_INI_PARSE_ERROR, File=File, Line=Line, ExtraData=Error)

def __ValidatePath(Path, Root):
    if False:
        return 10
    Path = Path.strip()
    if os.path.isabs(Path) or not IsValidPath(Path, Root):
        return (False, ST.ERR_FILELIST_LOCATION % (Root, Path))
    return (True, '')

def ValidateMiscFile(Filename):
    if False:
        for i in range(10):
            print('nop')
    Root = GlobalData.gWORKSPACE
    return __ValidatePath(Filename, Root)

def ValidateToolsFile(Filename):
    if False:
        while True:
            i = 10
    (Valid, Cause) = (False, '')
    if not Valid and 'EDK_TOOLS_PATH' in os.environ:
        (Valid, Cause) = __ValidatePath(Filename, os.environ['EDK_TOOLS_PATH'])
    if not Valid:
        (Valid, Cause) = __ValidatePath(Filename, GlobalData.gWORKSPACE)
    return (Valid, Cause)

def ParseFileList(Line, Map, CurrentKey, PathFunc):
    if False:
        i = 10
        return i + 15
    FileList = ['', {}]
    TokenList = Line.split(TAB_VALUE_SPLIT)
    if len(TokenList) > 0:
        Path = TokenList[0].strip().replace('\\', '/')
        if not Path:
            return (False, ST.ERR_WRONG_FILELIST_FORMAT)
        (Valid, Cause) = PathFunc(Path)
        if not Valid:
            return (Valid, Cause)
        FileList[0] = TokenList[0].strip()
        for Token in TokenList[1:]:
            Attr = Token.split(TAB_EQUAL_SPLIT)
            if len(Attr) != 2 or not Attr[0].strip() or (not Attr[1].strip()):
                return (False, ST.ERR_WRONG_FILELIST_FORMAT)
            Key = Attr[0].strip()
            Val = Attr[1].strip()
            if Key not in ['OS', 'Executable']:
                return (False, ST.ERR_UNKNOWN_FILELIST_ATTR % Key)
            if Key == 'OS' and Val not in ['Win32', 'Win64', 'Linux32', 'Linux64', 'OS/X32', 'OS/X64', 'GenericWin', 'GenericNix']:
                return (False, ST.ERR_FILELIST_ATTR % 'OS')
            elif Key == 'Executable' and Val not in ['true', 'false']:
                return (False, ST.ERR_FILELIST_ATTR % 'Executable')
            FileList[1][Key] = Val
        Map[CurrentKey].append(FileList)
    return (True, '')

def CreateHeaderXml(DistMap, Root):
    if False:
        print('Hello World!')
    Element1 = CreateXmlElement('Name', DistMap['Name'], [], [['BaseName', DistMap['BaseName']]])
    Element2 = CreateXmlElement('GUID', DistMap['GUID'], [], [['Version', DistMap['Version']]])
    AttributeList = [['ReadOnly', DistMap['ReadOnly']], ['RePackage', DistMap['RePackage']]]
    NodeList = [Element1, Element2, ['Vendor', DistMap['Vendor']], ['Date', DistMap['Date']], ['Copyright', DistMap['Copyright']], ['License', DistMap['License']], ['Abstract', DistMap['Abstract']], ['Description', DistMap['Description']], ['Signature', DistMap['Signature']], ['XmlSpecification', DistMap['XmlSpecification']]]
    Root.appendChild(CreateXmlElement('DistributionHeader', '', NodeList, AttributeList))

def CreateToolsXml(Map, Root, Tag):
    if False:
        for i in range(10):
            print('nop')
    for Key in Map:
        if len(Map[Key]) > 0:
            break
    else:
        return
    NodeList = [['Name', Map['Name']], ['Copyright', Map['Copyright']], ['License', Map['License']], ['Abstract', Map['Abstract']], ['Description', Map['Description']]]
    HeaderNode = CreateXmlElement('Header', '', NodeList, [])
    NodeList = [HeaderNode]
    for File in Map['FileList']:
        AttrList = []
        for Key in File[1]:
            AttrList.append([Key, File[1][Key]])
        NodeList.append(CreateXmlElement('Filename', File[0], [], AttrList))
    Root.appendChild(CreateXmlElement(Tag, '', NodeList, []))

def ValidateValues(Key, Value, SectionName):
    if False:
        while True:
            i = 10
    if SectionName == 'DistributionHeader':
        (Valid, Cause) = ValidateRegValues(Key, Value)
        if not Valid:
            return (Valid, Cause)
        Valid = __ValidateDistHeader(Key, Value)
        if not Valid:
            return (Valid, ST.ERR_VALUE_INVALID % (Key, SectionName))
    else:
        Valid = __ValidateOtherHeader(Key, Value)
        if not Valid:
            return (Valid, ST.ERR_VALUE_INVALID % (Key, SectionName))
    return (True, '')

def ValidateRegValues(Key, Value):
    if False:
        i = 10
        return i + 15
    ValidateMap = {'ReadOnly': ('true|false', ST.ERR_BOOLEAN_VALUE % (Key, Value)), 'RePackage': ('true|false', ST.ERR_BOOLEAN_VALUE % (Key, Value)), 'GUID': ('[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', ST.ERR_GUID_VALUE % Value), 'Version': ('[0-9]+(\\.[0-9]+)?', ST.ERR_VERSION_VALUE % (Key, Value)), 'XmlSpecification': ('1\\.1', ST.ERR_VERSION_XMLSPEC % Value)}
    if Key not in ValidateMap:
        return (True, '')
    Elem = ValidateMap[Key]
    Match = re.compile(Elem[0]).match(Value)
    if Match and Match.start() == 0 and (Match.end() == len(Value)):
        return (True, '')
    return (False, Elem[1])

def __ValidateDistHeaderName(Name):
    if False:
        i = 10
        return i + 15
    if len(Name) < 1:
        return False
    for Char in Name:
        if ord(Char) < 32 or ord(Char) >= 127:
            return False
    return True

def __ValidateDistHeaderBaseName(BaseName):
    if False:
        i = 10
        return i + 15
    if not BaseName:
        return False
    if not BaseName[0].isalnum() and BaseName[0] != '_':
        return False
    for Char in BaseName[1:]:
        if not Char.isalnum() and Char not in '-_':
            return False
    return True

def __ValidateDistHeaderAbstract(Abstract):
    if False:
        i = 10
        return i + 15
    return '\t' not in Abstract and len(Abstract.splitlines()) == 1

def __ValidateOtherHeaderAbstract(Abstract):
    if False:
        print('Hello World!')
    return __ValidateDistHeaderAbstract(Abstract)

def __ValidateDistHeader(Key, Value):
    if False:
        print('Hello World!')
    ValidateMap = {'Name': __ValidateDistHeaderName, 'BaseName': __ValidateDistHeaderBaseName, 'Abstract': __ValidateDistHeaderAbstract, 'Vendor': __ValidateDistHeaderAbstract}
    return not (Value and Key in ValidateMap and (not ValidateMap[Key](Value)))

def __ValidateOtherHeader(Key, Value):
    if False:
        i = 10
        return i + 15
    ValidateMap = {'Name': __ValidateDistHeaderName, 'Abstract': __ValidateOtherHeaderAbstract}
    return not (Value and Key in ValidateMap and (not ValidateMap[Key](Value)))

def IniToXml(IniFile):
    if False:
        while True:
            i = 10
    if not os.path.exists(IniFile):
        Logger.Error('UPT', FILE_NOT_FOUND, ST.ERR_TEMPLATE_NOTFOUND % IniFile)
    DistMap = {'ReadOnly': '', 'RePackage': '', 'Name': '', 'BaseName': '', 'GUID': '', 'Version': '', 'Vendor': '', 'Date': '', 'Copyright': '', 'License': '', 'Abstract': '', 'Description': '', 'Signature': '', 'XmlSpecification': ''}
    ToolsMap = {'Name': '', 'Copyright': '', 'License': '', 'Abstract': '', 'Description': '', 'FileList': []}
    MiscMap = {'Name': '', 'Copyright': '', 'License': '', 'Abstract': '', 'Description': '', 'FileList': []}
    SectionMap = {'DistributionHeader': DistMap, 'ToolsHeader': ToolsMap, 'MiscellaneousFilesHeader': MiscMap}
    PathValidator = {'ToolsHeader': ValidateToolsFile, 'MiscellaneousFilesHeader': ValidateMiscFile}
    ParsedSection = []
    SectionName = ''
    CurrentKey = ''
    PreMap = None
    Map = None
    FileContent = ConvertSpecialChar(open(IniFile, 'r').readlines())
    LastIndex = 0
    for Index in range(0, len(FileContent)):
        LastIndex = Index
        Line = FileContent[Index].strip()
        if Line == '' or Line.startswith(';'):
            continue
        if Line[0] == TAB_SECTION_START and Line[-1] == TAB_SECTION_END:
            CurrentKey = ''
            SectionName = Line[1:-1].strip()
            if SectionName not in SectionMap:
                IniParseError(ST.ERR_SECTION_NAME_INVALID % SectionName, IniFile, Index + 1)
            if SectionName in ParsedSection:
                IniParseError(ST.ERR_SECTION_REDEFINE % SectionName, IniFile, Index + 1)
            else:
                ParsedSection.append(SectionName)
            Map = SectionMap[SectionName]
            continue
        if not Map:
            IniParseError(ST.ERR_SECTION_NAME_NONE, IniFile, Index + 1)
        TokenList = Line.split(TAB_EQUAL_SPLIT, 1)
        TempKey = TokenList[0].strip()
        if len(TokenList) < 2 or TempKey not in Map:
            if CurrentKey == '':
                IniParseError(ST.ERR_KEYWORD_INVALID % TempKey, IniFile, Index + 1)
            elif CurrentKey == 'FileList':
                (Valid, Cause) = ParseFileList(Line, Map, CurrentKey, PathValidator[SectionName])
                if not Valid:
                    IniParseError(Cause, IniFile, Index + 1)
            else:
                Map[CurrentKey] = ''.join([Map[CurrentKey], '\n', Line])
                (Valid, Cause) = ValidateValues(CurrentKey, Map[CurrentKey], SectionName)
                if not Valid:
                    IniParseError(Cause, IniFile, Index + 1)
            continue
        if TokenList[1].strip() == '':
            IniParseError(ST.ERR_EMPTY_VALUE, IniFile, Index + 1)
        CurrentKey = TempKey
        if Map[CurrentKey]:
            IniParseError(ST.ERR_KEYWORD_REDEFINE % CurrentKey, IniFile, Index + 1)
        if id(Map) != id(PreMap) and Map['Copyright']:
            PreMap = Map
            Copyright = Map['Copyright'].lower()
            Pos = Copyright.find('copyright')
            if Pos == -1:
                IniParseError(ST.ERR_COPYRIGHT_CONTENT, IniFile, Index)
            if not Copyright[Pos + len('copyright'):].lstrip(' ').startswith('('):
                IniParseError(ST.ERR_COPYRIGHT_CONTENT, IniFile, Index)
        if CurrentKey == 'FileList':
            (Valid, Cause) = ParseFileList(TokenList[1], Map, CurrentKey, PathValidator[SectionName])
            if not Valid:
                IniParseError(Cause, IniFile, Index + 1)
        else:
            Map[CurrentKey] = TokenList[1].strip()
            (Valid, Cause) = ValidateValues(CurrentKey, Map[CurrentKey], SectionName)
            if not Valid:
                IniParseError(Cause, IniFile, Index + 1)
    if id(Map) != id(PreMap) and Map['Copyright'] and ('copyright' not in Map['Copyright'].lower()):
        IniParseError(ST.ERR_COPYRIGHT_CONTENT, IniFile, LastIndex)
    CheckMdtKeys(DistMap, IniFile, LastIndex, (('ToolsHeader', ToolsMap), ('MiscellaneousFilesHeader', MiscMap)))
    return CreateXml(DistMap, ToolsMap, MiscMap, IniFile)

def CheckMdtKeys(DistMap, IniFile, LastIndex, Maps):
    if False:
        for i in range(10):
            print('nop')
    MdtDistKeys = ['Name', 'GUID', 'Version', 'Vendor', 'Copyright', 'License', 'Abstract', 'XmlSpecification']
    for Key in MdtDistKeys:
        if Key not in DistMap or DistMap[Key] == '':
            IniParseError(ST.ERR_KEYWORD_MANDATORY % Key, IniFile, LastIndex + 1)
    if '.' not in DistMap['Version']:
        DistMap['Version'] = DistMap['Version'] + '.0'
    DistMap['Date'] = str(strftime('%Y-%m-%dT%H:%M:%S', localtime()))
    for Item in Maps:
        Map = Item[1]
        NonEmptyKey = 0
        for Key in Map:
            if Map[Key]:
                NonEmptyKey += 1
        if NonEmptyKey > 0 and (not Map['FileList']):
            IniParseError(ST.ERR_KEYWORD_MANDATORY % (Item[0] + '.FileList'), IniFile, LastIndex + 1)
        if NonEmptyKey > 0 and (not Map['Name']):
            IniParseError(ST.ERR_KEYWORD_MANDATORY % (Item[0] + '.Name'), IniFile, LastIndex + 1)

def CreateXml(DistMap, ToolsMap, MiscMap, IniFile):
    if False:
        i = 10
        return i + 15
    Attrs = [['xmlns', 'http://www.uefi.org/2011/1.1'], ['xmlns:xsi', 'http:/www.w3.org/2001/XMLSchema-instance']]
    Root = CreateXmlElement('DistributionPackage', '', [], Attrs)
    CreateHeaderXml(DistMap, Root)
    CreateToolsXml(ToolsMap, Root, 'Tools')
    CreateToolsXml(MiscMap, Root, 'MiscellaneousFiles')
    FileAndExt = IniFile.rsplit('.', 1)
    if len(FileAndExt) > 1:
        FileName = FileAndExt[0] + '.xml'
    else:
        FileName = IniFile + '.xml'
    File = open(FileName, 'w')
    try:
        File.write(Root.toprettyxml(indent='  '))
    finally:
        File.close()
    return FileName