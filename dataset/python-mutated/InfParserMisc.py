"""
InfParserMisc
"""
import re
from Library import DataType as DT
from Library.StringUtils import gMACRO_PATTERN
from Library.StringUtils import ReplaceMacro
from Object.Parser.InfMisc import ErrorInInf
from Logger.StringTable import ERR_MARCO_DEFINITION_MISS_ERROR
gINF_SECTION_DEF = {DT.TAB_UNKNOWN.upper(): DT.MODEL_UNKNOWN, DT.TAB_HEADER.upper(): DT.MODEL_META_DATA_FILE_HEADER, DT.TAB_INF_DEFINES.upper(): DT.MODEL_META_DATA_DEFINE, DT.TAB_BUILD_OPTIONS.upper(): DT.MODEL_META_DATA_BUILD_OPTION, DT.TAB_LIBRARY_CLASSES.upper(): DT.MODEL_EFI_LIBRARY_CLASS, DT.TAB_PACKAGES.upper(): DT.MODEL_META_DATA_PACKAGE, DT.TAB_INF_FIXED_PCD.upper(): DT.MODEL_PCD_FIXED_AT_BUILD, DT.TAB_INF_PATCH_PCD.upper(): DT.MODEL_PCD_PATCHABLE_IN_MODULE, DT.TAB_INF_FEATURE_PCD.upper(): DT.MODEL_PCD_FEATURE_FLAG, DT.TAB_INF_PCD_EX.upper(): DT.MODEL_PCD_DYNAMIC_EX, DT.TAB_INF_PCD.upper(): DT.MODEL_PCD_DYNAMIC, DT.TAB_SOURCES.upper(): DT.MODEL_EFI_SOURCE_FILE, DT.TAB_GUIDS.upper(): DT.MODEL_EFI_GUID, DT.TAB_PROTOCOLS.upper(): DT.MODEL_EFI_PROTOCOL, DT.TAB_PPIS.upper(): DT.MODEL_EFI_PPI, DT.TAB_DEPEX.upper(): DT.MODEL_EFI_DEPEX, DT.TAB_BINARIES.upper(): DT.MODEL_EFI_BINARY_FILE, DT.TAB_USER_EXTENSIONS.upper(): DT.MODEL_META_DATA_USER_EXTENSION}

def InfExpandMacro(Content, LineInfo, GlobalMacros=None, SectionMacros=None, Flag=False):
    if False:
        return 10
    if GlobalMacros is None:
        GlobalMacros = {}
    if SectionMacros is None:
        SectionMacros = {}
    FileName = LineInfo[0]
    LineContent = LineInfo[1]
    LineNo = LineInfo[2]
    if LineContent.strip().startswith('#'):
        return Content
    NewLineInfo = (FileName, LineNo, LineContent)
    Content = ReplaceMacro(Content, SectionMacros, False, (LineContent, LineNo), FileName, Flag)
    Content = ReplaceMacro(Content, GlobalMacros, False, (LineContent, LineNo), FileName, Flag)
    MacroUsed = gMACRO_PATTERN.findall(Content)
    if len(MacroUsed) == 0:
        return Content
    else:
        for Macro in MacroUsed:
            gQuotedMacro = re.compile('.*".*\\$\\(%s\\).*".*' % Macro)
            if not gQuotedMacro.match(Content):
                ErrorInInf(ERR_MARCO_DEFINITION_MISS_ERROR, LineInfo=NewLineInfo)
    return Content

def IsBinaryInf(FileLineList):
    if False:
        for i in range(10):
            print('nop')
    if not FileLineList:
        return False
    ReIsSourcesSection = re.compile('^\\s*\\[Sources.*\\]\\s.*$', re.IGNORECASE)
    ReIsBinarySection = re.compile('^\\s*\\[Binaries.*\\]\\s.*$', re.IGNORECASE)
    BinarySectionFoundFlag = False
    for Line in FileLineList:
        if ReIsSourcesSection.match(Line):
            return False
        if ReIsBinarySection.match(Line):
            BinarySectionFoundFlag = True
    if BinarySectionFoundFlag:
        return True
    return False

def IsLibInstanceInfo(String):
    if False:
        return 10
    ReIsLibInstance = re.compile('^\\s*##\\s*@LIB_INSTANCES\\s*$')
    if ReIsLibInstance.match(String):
        return True
    else:
        return False

def IsAsBuildOptionInfo(String):
    if False:
        while True:
            i = 10
    ReIsAsBuildInstance = re.compile('^\\s*##\\s*@AsBuilt\\s*$')
    if ReIsAsBuildInstance.match(String):
        return True
    else:
        return False

class InfParserSectionRoot(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.FileLocalMacros = {}
        self.SectionHeaderContent = []
        self.LastSectionHeaderContent = []
        self.FullPath = ''
        self.InfDefSection = None
        self.InfBuildOptionSection = None
        self.InfLibraryClassSection = None
        self.InfPackageSection = None
        self.InfPcdSection = None
        self.InfSourcesSection = None
        self.InfUserExtensionSection = None
        self.InfProtocolSection = None
        self.InfPpiSection = None
        self.InfGuidSection = None
        self.InfDepexSection = None
        self.InfPeiDepexSection = None
        self.InfDxeDepexSection = None
        self.InfSmmDepexSection = None
        self.InfBinariesSection = None
        self.InfHeader = None
        self.InfSpecialCommentSection = None