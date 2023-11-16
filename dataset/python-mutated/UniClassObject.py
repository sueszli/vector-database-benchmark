from __future__ import print_function
import Common.LongFilePathOs as os, codecs, re
import shlex
import Common.EdkLogger as EdkLogger
from io import BytesIO
from Common.BuildToolError import *
from Common.StringUtils import GetLineNo
from Common.Misc import PathClass
from Common.LongFilePathSupport import LongFilePath
from Common.GlobalData import *
UNICODE_WIDE_CHAR = u'\\wide'
UNICODE_NARROW_CHAR = u'\\narrow'
UNICODE_NON_BREAKING_CHAR = u'\\nbr'
UNICODE_UNICODE_CR = '\r'
UNICODE_UNICODE_LF = '\n'
NARROW_CHAR = u'\ufff0'
WIDE_CHAR = u'\ufff1'
NON_BREAKING_CHAR = u'\ufff2'
CR = u'\r'
LF = u'\n'
NULL = u'\x00'
TAB = u'\t'
BACK_SLASH_PLACEHOLDER = u'\x06'
gIncludePattern = re.compile('^#include +["<]+([^"< >]+)[>"]+$', re.MULTILINE | re.UNICODE)

def UniToHexList(Uni):
    if False:
        print('Hello World!')
    List = []
    for Item in Uni:
        Temp = '%04X' % ord(Item)
        List.append('0x' + Temp[2:4])
        List.append('0x' + Temp[0:2])
    return List
LangConvTable = {'eng': 'en', 'fra': 'fr', 'aar': 'aa', 'abk': 'ab', 'ave': 'ae', 'afr': 'af', 'aka': 'ak', 'amh': 'am', 'arg': 'an', 'ara': 'ar', 'asm': 'as', 'ava': 'av', 'aym': 'ay', 'aze': 'az', 'bak': 'ba', 'bel': 'be', 'bul': 'bg', 'bih': 'bh', 'bis': 'bi', 'bam': 'bm', 'ben': 'bn', 'bod': 'bo', 'bre': 'br', 'bos': 'bs', 'cat': 'ca', 'che': 'ce', 'cha': 'ch', 'cos': 'co', 'cre': 'cr', 'ces': 'cs', 'chu': 'cu', 'chv': 'cv', 'cym': 'cy', 'dan': 'da', 'deu': 'de', 'div': 'dv', 'dzo': 'dz', 'ewe': 'ee', 'ell': 'el', 'epo': 'eo', 'spa': 'es', 'est': 'et', 'eus': 'eu', 'fas': 'fa', 'ful': 'ff', 'fin': 'fi', 'fij': 'fj', 'fao': 'fo', 'fry': 'fy', 'gle': 'ga', 'gla': 'gd', 'glg': 'gl', 'grn': 'gn', 'guj': 'gu', 'glv': 'gv', 'hau': 'ha', 'heb': 'he', 'hin': 'hi', 'hmo': 'ho', 'hrv': 'hr', 'hat': 'ht', 'hun': 'hu', 'hye': 'hy', 'her': 'hz', 'ina': 'ia', 'ind': 'id', 'ile': 'ie', 'ibo': 'ig', 'iii': 'ii', 'ipk': 'ik', 'ido': 'io', 'isl': 'is', 'ita': 'it', 'iku': 'iu', 'jpn': 'ja', 'jav': 'jv', 'kat': 'ka', 'kon': 'kg', 'kik': 'ki', 'kua': 'kj', 'kaz': 'kk', 'kal': 'kl', 'khm': 'km', 'kan': 'kn', 'kor': 'ko', 'kau': 'kr', 'kas': 'ks', 'kur': 'ku', 'kom': 'kv', 'cor': 'kw', 'kir': 'ky', 'lat': 'la', 'ltz': 'lb', 'lug': 'lg', 'lim': 'li', 'lin': 'ln', 'lao': 'lo', 'lit': 'lt', 'lub': 'lu', 'lav': 'lv', 'mlg': 'mg', 'mah': 'mh', 'mri': 'mi', 'mkd': 'mk', 'mal': 'ml', 'mon': 'mn', 'mar': 'mr', 'msa': 'ms', 'mlt': 'mt', 'mya': 'my', 'nau': 'na', 'nob': 'nb', 'nde': 'nd', 'nep': 'ne', 'ndo': 'ng', 'nld': 'nl', 'nno': 'nn', 'nor': 'no', 'nbl': 'nr', 'nav': 'nv', 'nya': 'ny', 'oci': 'oc', 'oji': 'oj', 'orm': 'om', 'ori': 'or', 'oss': 'os', 'pan': 'pa', 'pli': 'pi', 'pol': 'pl', 'pus': 'ps', 'por': 'pt', 'que': 'qu', 'roh': 'rm', 'run': 'rn', 'ron': 'ro', 'rus': 'ru', 'kin': 'rw', 'san': 'sa', 'srd': 'sc', 'snd': 'sd', 'sme': 'se', 'sag': 'sg', 'sin': 'si', 'slk': 'sk', 'slv': 'sl', 'smo': 'sm', 'sna': 'sn', 'som': 'so', 'sqi': 'sq', 'srp': 'sr', 'ssw': 'ss', 'sot': 'st', 'sun': 'su', 'swe': 'sv', 'swa': 'sw', 'tam': 'ta', 'tel': 'te', 'tgk': 'tg', 'tha': 'th', 'tir': 'ti', 'tuk': 'tk', 'tgl': 'tl', 'tsn': 'tn', 'ton': 'to', 'tur': 'tr', 'tso': 'ts', 'tat': 'tt', 'twi': 'tw', 'tah': 'ty', 'uig': 'ug', 'ukr': 'uk', 'urd': 'ur', 'uzb': 'uz', 'ven': 've', 'vie': 'vi', 'vol': 'vo', 'wln': 'wa', 'wol': 'wo', 'xho': 'xh', 'yid': 'yi', 'yor': 'yo', 'zha': 'za', 'zho': 'zh', 'zul': 'zu'}

def GetLanguageCode(LangName, IsCompatibleMode, File):
    if False:
        i = 10
        return i + 15
    length = len(LangName)
    if IsCompatibleMode:
        if length == 3 and LangName.isalpha():
            TempLangName = LangConvTable.get(LangName.lower())
            if TempLangName is not None:
                return TempLangName
            return LangName
        else:
            EdkLogger.error('Unicode File Parser', FORMAT_INVALID, 'Invalid ISO 639-2 language code : %s' % LangName, File)
    if (LangName[0] == 'X' or LangName[0] == 'x') and LangName[1] == '-':
        return LangName
    if length == 2:
        if LangName.isalpha():
            return LangName
    elif length == 3:
        if LangName.isalpha() and LangConvTable.get(LangName.lower()) is None:
            return LangName
    elif length == 5:
        if LangName[0:2].isalpha() and LangName[2] == '-':
            return LangName
    elif length >= 6:
        if LangName[0:2].isalpha() and LangName[2] == '-':
            return LangName
        if LangName[0:3].isalpha() and LangConvTable.get(LangName.lower()) is None and (LangName[3] == '-'):
            return LangName
    EdkLogger.error('Unicode File Parser', FORMAT_INVALID, 'Invalid RFC 4646 language code : %s' % LangName, File)

class Ucs2Codec(codecs.Codec):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__utf16 = codecs.lookup('utf-16')

    def encode(self, input, errors='strict'):
        if False:
            for i in range(10):
                print('nop')
        for Char in input:
            CodePoint = ord(Char)
            if CodePoint >= 55296 and CodePoint <= 57343:
                raise ValueError('Code Point is in range reserved for ' + 'UTF-16 surrogate pairs')
            elif CodePoint > 65535:
                raise ValueError('Code Point too large to encode in UCS-2')
        return self.__utf16.encode(input)
TheUcs2Codec = Ucs2Codec()

def Ucs2Search(name):
    if False:
        for i in range(10):
            print('nop')
    if name in ['ucs-2', 'ucs_2']:
        return codecs.CodecInfo(name=name, encode=TheUcs2Codec.encode, decode=TheUcs2Codec.decode)
    else:
        return None
codecs.register(Ucs2Search)

class StringDefClassObject(object):

    def __init__(self, Name=None, Value=None, Referenced=False, Token=None, UseOtherLangDef=''):
        if False:
            i = 10
            return i + 15
        self.StringName = ''
        self.StringNameByteList = []
        self.StringValue = ''
        self.StringValueByteList = ''
        self.Token = 0
        self.Referenced = Referenced
        self.UseOtherLangDef = UseOtherLangDef
        self.Length = 0
        if Name is not None:
            self.StringName = Name
            self.StringNameByteList = UniToHexList(Name)
        if Value is not None:
            self.StringValue = Value + u'\x00'
            self.StringValueByteList = UniToHexList(self.StringValue)
            self.Length = len(self.StringValueByteList)
        if Token is not None:
            self.Token = Token

    def __str__(self):
        if False:
            while True:
                i = 10
        return repr(self.StringName) + ' ' + repr(self.Token) + ' ' + repr(self.Referenced) + ' ' + repr(self.StringValue) + ' ' + repr(self.UseOtherLangDef)

    def UpdateValue(self, Value=None):
        if False:
            i = 10
            return i + 15
        if Value is not None:
            self.StringValue = Value + u'\x00'
            self.StringValueByteList = UniToHexList(self.StringValue)
            self.Length = len(self.StringValueByteList)

def StripComments(Line):
    if False:
        for i in range(10):
            print('nop')
    Comment = u'//'
    CommentPos = Line.find(Comment)
    while CommentPos >= 0:
        if Line.count(u'"', 0, CommentPos) - Line.count(u'\\"', 0, CommentPos) & 1 == 1:
            CommentPos = Line.find(Comment, CommentPos + 1)
        else:
            return Line[:CommentPos].strip()
    return Line.strip()

class UniFileClassObject(object):

    def __init__(self, FileList=[], IsCompatibleMode=False, IncludePathList=[]):
        if False:
            print('Hello World!')
        self.FileList = FileList
        self.Token = 2
        self.LanguageDef = []
        self.OrderedStringList = {}
        self.OrderedStringDict = {}
        self.OrderedStringListByToken = {}
        self.IsCompatibleMode = IsCompatibleMode
        self.IncludePathList = IncludePathList
        if len(self.FileList) > 0:
            self.LoadUniFiles(FileList)

    def GetLangDef(self, File, Line):
        if False:
            for i in range(10):
                print('nop')
        Lang = shlex.split(Line.split(u'//')[0])
        if len(Lang) != 3:
            try:
                FileIn = UniFileClassObject.OpenUniFile(LongFilePath(File.Path))
            except UnicodeError as X:
                EdkLogger.error('build', FILE_READ_FAILURE, 'File read failure: %s' % str(X), ExtraData=File)
            except:
                EdkLogger.error('build', FILE_OPEN_FAILURE, ExtraData=File)
            LineNo = GetLineNo(FileIn, Line, False)
            EdkLogger.error('Unicode File Parser', PARSER_ERROR, 'Wrong language definition', ExtraData='%s\n\t*Correct format is like \'#langdef en-US "English"\'' % Line, File=File, Line=LineNo)
        else:
            LangName = GetLanguageCode(Lang[1], self.IsCompatibleMode, self.File)
            LangPrintName = Lang[2]
        IsLangInDef = False
        for Item in self.LanguageDef:
            if Item[0] == LangName:
                IsLangInDef = True
                break
        if not IsLangInDef:
            self.LanguageDef.append([LangName, LangPrintName])
        self.AddStringToList(u'$LANGUAGE_NAME', LangName, LangName, 0, True, Index=0)
        self.AddStringToList(u'$PRINTABLE_LANGUAGE_NAME', LangName, LangPrintName, 1, True, Index=1)
        if not IsLangInDef:
            FirstLangName = self.LanguageDef[0][0]
            if LangName != FirstLangName:
                for Index in range(2, len(self.OrderedStringList[FirstLangName])):
                    Item = self.OrderedStringList[FirstLangName][Index]
                    if Item.UseOtherLangDef != '':
                        OtherLang = Item.UseOtherLangDef
                    else:
                        OtherLang = FirstLangName
                    self.OrderedStringList[LangName].append(StringDefClassObject(Item.StringName, '', Item.Referenced, Item.Token, OtherLang))
                    self.OrderedStringDict[LangName][Item.StringName] = len(self.OrderedStringList[LangName]) - 1
        return True

    @staticmethod
    def OpenUniFile(FileName):
        if False:
            i = 10
            return i + 15
        try:
            UniFile = open(FileName, mode='rb')
            FileIn = UniFile.read()
            UniFile.close()
        except:
            EdkLogger.Error('build', FILE_OPEN_FAILURE, ExtraData=File)
        Encoding = 'utf-8'
        if FileIn.startswith(codecs.BOM_UTF16_BE) or FileIn.startswith(codecs.BOM_UTF16_LE):
            Encoding = 'utf-16'
        UniFileClassObject.VerifyUcs2Data(FileIn, FileName, Encoding)
        UniFile = BytesIO(FileIn)
        Info = codecs.lookup(Encoding)
        (Reader, Writer) = (Info.streamreader, Info.streamwriter)
        return codecs.StreamReaderWriter(UniFile, Reader, Writer)

    @staticmethod
    def VerifyUcs2Data(FileIn, FileName, Encoding):
        if False:
            return 10
        Ucs2Info = codecs.lookup('ucs-2')
        try:
            FileDecoded = codecs.decode(FileIn, Encoding)
            Ucs2Info.encode(FileDecoded)
        except:
            UniFile = BytesIO(FileIn)
            Info = codecs.lookup(Encoding)
            (Reader, Writer) = (Info.streamreader, Info.streamwriter)
            File = codecs.StreamReaderWriter(UniFile, Reader, Writer)
            LineNumber = 0
            ErrMsg = lambda Encoding, LineNumber: '%s contains invalid %s characters on line %d.' % (FileName, Encoding, LineNumber)
            while True:
                LineNumber = LineNumber + 1
                try:
                    Line = File.readline()
                    if Line == '':
                        EdkLogger.error('Unicode File Parser', PARSER_ERROR, ErrMsg(Encoding, LineNumber))
                    Ucs2Info.encode(Line)
                except:
                    EdkLogger.error('Unicode File Parser', PARSER_ERROR, ErrMsg('UCS-2', LineNumber))

    def GetStringObject(self, Item):
        if False:
            print('Hello World!')
        Language = ''
        Value = ''
        Name = Item.split()[1]
        if Name != '':
            MatchString = gIdentifierPattern.match(Name)
            if MatchString is None:
                EdkLogger.error('Unicode File Parser', FORMAT_INVALID, 'The string token name %s defined in UNI file %s contains the invalid character.' % (Name, self.File))
        LanguageList = Item.split(u'#language ')
        for IndexI in range(len(LanguageList)):
            if IndexI == 0:
                continue
            else:
                Language = LanguageList[IndexI].split()[0]
                Value = LanguageList[IndexI][LanguageList[IndexI].find(u'"') + len(u'"'):LanguageList[IndexI].rfind(u'"')]
                Language = GetLanguageCode(Language, self.IsCompatibleMode, self.File)
                self.AddStringToList(Name, Language, Value)

    def GetIncludeFile(self, Item, Dir):
        if False:
            for i in range(10):
                print('nop')
        FileName = Item[Item.find(u'#include ') + len(u'#include '):Item.find(u' ', len(u'#include '))][1:-1]
        self.LoadUniFile(FileName)

    def PreProcess(self, File):
        if False:
            print('Hello World!')
        try:
            FileIn = UniFileClassObject.OpenUniFile(LongFilePath(File.Path))
        except UnicodeError as X:
            EdkLogger.error('build', FILE_READ_FAILURE, 'File read failure: %s' % str(X), ExtraData=File.Path)
        except OSError:
            EdkLogger.error('Unicode File Parser', FILE_NOT_FOUND, ExtraData=File.Path)
        except:
            EdkLogger.error('build', FILE_OPEN_FAILURE, ExtraData=File.Path)
        Lines = []
        for Line in FileIn:
            Line = Line.strip()
            Line = Line.replace(u'\\\\', BACK_SLASH_PLACEHOLDER)
            Line = StripComments(Line)
            if len(Line) == 0:
                continue
            Line = Line.replace(u'/langdef', u'#langdef')
            Line = Line.replace(u'/string', u'#string')
            Line = Line.replace(u'/language', u'#language')
            Line = Line.replace(u'/include', u'#include')
            Line = Line.replace(UNICODE_WIDE_CHAR, WIDE_CHAR)
            Line = Line.replace(UNICODE_NARROW_CHAR, NARROW_CHAR)
            Line = Line.replace(UNICODE_NON_BREAKING_CHAR, NON_BREAKING_CHAR)
            Line = Line.replace(u'\\r\\n', CR + LF)
            Line = Line.replace(u'\\n', CR + LF)
            Line = Line.replace(u'\\r', CR)
            Line = Line.replace(u'\\t', u' ')
            Line = Line.replace(u'\t', u' ')
            Line = Line.replace(u'\\"', u'"')
            Line = Line.replace(u"\\'", u"'")
            Line = Line.replace(BACK_SLASH_PLACEHOLDER, u'\\')
            StartPos = Line.find(u'\\x')
            while StartPos != -1:
                EndPos = Line.find(u'\\', StartPos + 1, StartPos + 7)
                if EndPos != -1 and EndPos - StartPos == 6:
                    if g4HexChar.match(Line[StartPos + 2:EndPos], re.UNICODE):
                        EndStr = Line[EndPos:]
                        UniStr = Line[StartPos + 2:EndPos]
                        if EndStr.startswith(u'\\x') and len(EndStr) >= 7:
                            if EndStr[6] == u'\\' and g4HexChar.match(EndStr[2:6], re.UNICODE):
                                Line = Line[0:StartPos] + UniStr + EndStr
                        else:
                            Line = Line[0:StartPos] + UniStr + EndStr[1:]
                StartPos = Line.find(u'\\x', StartPos + 1)
            IncList = gIncludePattern.findall(Line)
            if len(IncList) == 1:
                for Dir in [File.Dir] + self.IncludePathList:
                    IncFile = PathClass(str(IncList[0]), Dir)
                    if os.path.isfile(IncFile.Path):
                        Lines.extend(self.PreProcess(IncFile))
                        break
                else:
                    EdkLogger.error('Unicode File Parser', FILE_NOT_FOUND, Message='Cannot find include file', ExtraData=str(IncList[0]))
                continue
            Lines.append(Line)
        return Lines

    def LoadUniFile(self, File=None):
        if False:
            while True:
                i = 10
        if File is None:
            EdkLogger.error('Unicode File Parser', PARSER_ERROR, 'No unicode file is given')
        self.File = File
        Lines = self.PreProcess(File)
        for IndexI in range(len(Lines)):
            Line = Lines[IndexI]
            if IndexI + 1 < len(Lines):
                SecondLine = Lines[IndexI + 1]
            if IndexI + 2 < len(Lines):
                ThirdLine = Lines[IndexI + 2]
            if Line.find(u'#langdef ') >= 0:
                self.GetLangDef(File, Line)
                continue
            Name = ''
            Language = ''
            Value = ''
            if Line.find(u'#string ') >= 0 and Line.find(u'#language ') < 0 and (SecondLine.find(u'#string ') < 0) and (SecondLine.find(u'#language ') >= 0) and (ThirdLine.find(u'#string ') < 0) and (ThirdLine.find(u'#language ') < 0):
                Name = Line[Line.find(u'#string ') + len(u'#string '):].strip(' ')
                Language = SecondLine[SecondLine.find(u'#language ') + len(u'#language '):].strip(' ')
                for IndexJ in range(IndexI + 2, len(Lines)):
                    if Lines[IndexJ].find(u'#string ') < 0 and Lines[IndexJ].find(u'#language ') < 0:
                        Value = Value + Lines[IndexJ]
                    else:
                        IndexI = IndexJ
                        break
                Language = GetLanguageCode(Language, self.IsCompatibleMode, self.File)
                if not self.IsCompatibleMode and Name != '':
                    MatchString = gIdentifierPattern.match(Name)
                    if MatchString is None:
                        EdkLogger.error('Unicode File Parser', FORMAT_INVALID, 'The string token name %s defined in UNI file %s contains the invalid character.' % (Name, self.File))
                self.AddStringToList(Name, Language, Value)
                continue
            if Line.find(u'#string ') >= 0 and Line.find(u'#language ') >= 0:
                StringItem = Line
                for IndexJ in range(IndexI + 1, len(Lines)):
                    if Lines[IndexJ].find(u'#string ') >= 0 and Lines[IndexJ].find(u'#language ') >= 0:
                        IndexI = IndexJ
                        break
                    elif Lines[IndexJ].find(u'#string ') < 0 and Lines[IndexJ].find(u'#language ') >= 0:
                        StringItem = StringItem + Lines[IndexJ]
                    elif Lines[IndexJ].count(u'"') >= 2:
                        StringItem = StringItem[:StringItem.rfind(u'"')] + Lines[IndexJ][Lines[IndexJ].find(u'"') + len(u'"'):]
                self.GetStringObject(StringItem)
                continue

    def LoadUniFiles(self, FileList):
        if False:
            i = 10
            return i + 15
        if len(FileList) > 0:
            for File in FileList:
                self.LoadUniFile(File)

    def AddStringToList(self, Name, Language, Value, Token=None, Referenced=False, UseOtherLangDef='', Index=-1):
        if False:
            for i in range(10):
                print('nop')
        for LangNameItem in self.LanguageDef:
            if Language == LangNameItem[0]:
                break
        else:
            EdkLogger.error('Unicode File Parser', FORMAT_NOT_SUPPORTED, "The language '%s' for %s is not defined in Unicode file %s." % (Language, Name, self.File))
        if Language not in self.OrderedStringList:
            self.OrderedStringList[Language] = []
            self.OrderedStringDict[Language] = {}
        IsAdded = True
        if Name in self.OrderedStringDict[Language]:
            IsAdded = False
            if Value is not None:
                ItemIndexInList = self.OrderedStringDict[Language][Name]
                Item = self.OrderedStringList[Language][ItemIndexInList]
                Item.UpdateValue(Value)
                Item.UseOtherLangDef = ''
        if IsAdded:
            Token = len(self.OrderedStringList[Language])
            if Index == -1:
                self.OrderedStringList[Language].append(StringDefClassObject(Name, Value, Referenced, Token, UseOtherLangDef))
                self.OrderedStringDict[Language][Name] = Token
                for LangName in self.LanguageDef:
                    if LangName[0] != Language:
                        if UseOtherLangDef != '':
                            OtherLangDef = UseOtherLangDef
                        else:
                            OtherLangDef = Language
                        self.OrderedStringList[LangName[0]].append(StringDefClassObject(Name, '', Referenced, Token, OtherLangDef))
                        self.OrderedStringDict[LangName[0]][Name] = len(self.OrderedStringList[LangName[0]]) - 1
            else:
                self.OrderedStringList[Language].insert(Index, StringDefClassObject(Name, Value, Referenced, Token, UseOtherLangDef))
                self.OrderedStringDict[Language][Name] = Index

    def SetStringReferenced(self, Name):
        if False:
            print('Hello World!')
        Lang = self.LanguageDef[0][0]
        if Name in self.OrderedStringDict[Lang]:
            ItemIndexInList = self.OrderedStringDict[Lang][Name]
            Item = self.OrderedStringList[Lang][ItemIndexInList]
            Item.Referenced = True

    def FindStringValue(self, Name, Lang):
        if False:
            i = 10
            return i + 15
        if Name in self.OrderedStringDict[Lang]:
            ItemIndexInList = self.OrderedStringDict[Lang][Name]
            return self.OrderedStringList[Lang][ItemIndexInList]
        return None

    def FindByToken(self, Token, Lang):
        if False:
            for i in range(10):
                print('nop')
        for Item in self.OrderedStringList[Lang]:
            if Item.Token == Token:
                return Item
        return None

    def ReToken(self):
        if False:
            while True:
                i = 10
        FirstLangName = self.LanguageDef[0][0]
        for LangNameItem in self.LanguageDef:
            self.OrderedStringListByToken[LangNameItem[0]] = {}
        RefToken = 0
        for Index in range(0, len(self.OrderedStringList[FirstLangName])):
            FirstLangItem = self.OrderedStringList[FirstLangName][Index]
            if FirstLangItem.Referenced == True:
                for LangNameItem in self.LanguageDef:
                    LangName = LangNameItem[0]
                    OtherLangItem = self.OrderedStringList[LangName][Index]
                    OtherLangItem.Referenced = True
                    OtherLangItem.Token = RefToken
                    self.OrderedStringListByToken[LangName][OtherLangItem.Token] = OtherLangItem
                RefToken = RefToken + 1
        UnRefToken = 0
        for Index in range(0, len(self.OrderedStringList[FirstLangName])):
            FirstLangItem = self.OrderedStringList[FirstLangName][Index]
            if FirstLangItem.Referenced == False:
                for LangNameItem in self.LanguageDef:
                    LangName = LangNameItem[0]
                    OtherLangItem = self.OrderedStringList[LangName][Index]
                    OtherLangItem.Token = RefToken + UnRefToken
                    self.OrderedStringListByToken[LangName][OtherLangItem.Token] = OtherLangItem
                UnRefToken = UnRefToken + 1

    def ShowMe(self):
        if False:
            i = 10
            return i + 15
        print(self.LanguageDef)
        for Item in self.OrderedStringList:
            print(Item)
            for Member in self.OrderedStringList[Item]:
                print(str(Member))
if __name__ == '__main__':
    EdkLogger.Initialize()
    EdkLogger.SetLevel(EdkLogger.DEBUG_0)
    a = UniFileClassObject([PathClass('C:\\Edk\\Strings.uni'), PathClass('C:\\Edk\\Strings2.uni')])
    a.ReToken()
    a.ShowMe()