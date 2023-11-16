from __future__ import print_function
import Common.LongFilePathOs as os
import Common.EdkLogger as EdkLogger
from Common.DataType import *
from Common.StringUtils import *
from Common.LongFilePathSupport import OpenLongFilePath as open
_ConfigFileToInternalTranslation = {'ModifierList': 'ModifierSet', 'AutoCorrect': 'AutoCorrect', 'BinaryExtList': 'BinaryExtList', 'CFunctionLayoutCheckAll': 'CFunctionLayoutCheckAll', 'CFunctionLayoutCheckDataDeclaration': 'CFunctionLayoutCheckDataDeclaration', 'CFunctionLayoutCheckFunctionBody': 'CFunctionLayoutCheckFunctionBody', 'CFunctionLayoutCheckFunctionName': 'CFunctionLayoutCheckFunctionName', 'CFunctionLayoutCheckFunctionPrototype': 'CFunctionLayoutCheckFunctionPrototype', 'CFunctionLayoutCheckNoInitOfVariable': 'CFunctionLayoutCheckNoInitOfVariable', 'CFunctionLayoutCheckNoStatic': 'CFunctionLayoutCheckNoStatic', 'CFunctionLayoutCheckOptionalFunctionalModifier': 'CFunctionLayoutCheckOptionalFunctionalModifier', 'CFunctionLayoutCheckReturnType': 'CFunctionLayoutCheckReturnType', 'CheckAll': 'CheckAll', 'Copyright': 'Copyright', 'DeclarationDataTypeCheckAll': 'DeclarationDataTypeCheckAll', 'DeclarationDataTypeCheckEFIAPIModifier': 'DeclarationDataTypeCheckEFIAPIModifier', 'DeclarationDataTypeCheckEnumeratedType': 'DeclarationDataTypeCheckEnumeratedType', 'DeclarationDataTypeCheckInOutModifier': 'DeclarationDataTypeCheckInOutModifier', 'DeclarationDataTypeCheckNoUseCType': 'DeclarationDataTypeCheckNoUseCType', 'DeclarationDataTypeCheckSameStructure': 'DeclarationDataTypeCheckSameStructure', 'DeclarationDataTypeCheckStructureDeclaration': 'DeclarationDataTypeCheckStructureDeclaration', 'DeclarationDataTypeCheckUnionType': 'DeclarationDataTypeCheckUnionType', 'DoxygenCheckAll': 'DoxygenCheckAll', 'DoxygenCheckCommand': 'DoxygenCheckCommand', 'DoxygenCheckCommentDescription': 'DoxygenCheckCommentDescription', 'DoxygenCheckCommentFormat': 'DoxygenCheckCommentFormat', 'DoxygenCheckFileHeader': 'DoxygenCheckFileHeader', 'DoxygenCheckFunctionHeader': 'DoxygenCheckFunctionHeader', 'GeneralCheckAll': 'GeneralCheckAll', 'GeneralCheckCarriageReturn': 'GeneralCheckCarriageReturn', 'GeneralCheckFileExistence': 'GeneralCheckFileExistence', 'GeneralCheckIndentation': 'GeneralCheckIndentation', 'GeneralCheckIndentationWidth': 'GeneralCheckIndentationWidth', 'GeneralCheckLine': 'GeneralCheckLine', 'GeneralCheckLineEnding': 'GeneralCheckLineEnding', 'GeneralCheckLineWidth': 'GeneralCheckLineWidth', 'GeneralCheckNoProgma': 'GeneralCheckNoProgma', 'GeneralCheckNoTab': 'GeneralCheckNoTab', 'GeneralCheckNo_Asm': 'GeneralCheckNo_Asm', 'GeneralCheckNonAcsii': 'GeneralCheckNonAcsii', 'GeneralCheckTabWidth': 'GeneralCheckTabWidth', 'GeneralCheckTrailingWhiteSpaceLine': 'GeneralCheckTrailingWhiteSpaceLine', 'GeneralCheckUni': 'GeneralCheckUni', 'HeaderCheckAll': 'HeaderCheckAll', 'HeaderCheckCFileCommentLicenseFormat': 'HeaderCheckCFileCommentLicenseFormat', 'HeaderCheckCFileCommentReferenceFormat': 'HeaderCheckCFileCommentReferenceFormat', 'HeaderCheckCFileCommentStartSpacesNum': 'HeaderCheckCFileCommentStartSpacesNum', 'HeaderCheckFile': 'HeaderCheckFile', 'HeaderCheckFileCommentEnd': 'HeaderCheckFileCommentEnd', 'HeaderCheckFunction': 'HeaderCheckFunction', 'IncludeFileCheckAll': 'IncludeFileCheckAll', 'IncludeFileCheckData': 'IncludeFileCheckData', 'IncludeFileCheckIfndefStatement': 'IncludeFileCheckIfndefStatement', 'IncludeFileCheckSameName': 'IncludeFileCheckSameName', 'MetaDataFileCheckAll': 'MetaDataFileCheckAll', 'MetaDataFileCheckBinaryInfInFdf': 'MetaDataFileCheckBinaryInfInFdf', 'MetaDataFileCheckGenerateFileList': 'MetaDataFileCheckGenerateFileList', 'MetaDataFileCheckGuidDuplicate': 'MetaDataFileCheckGuidDuplicate', 'MetaDataFileCheckLibraryDefinedInDec': 'MetaDataFileCheckLibraryDefinedInDec', 'MetaDataFileCheckLibraryInstance': 'MetaDataFileCheckLibraryInstance', 'MetaDataFileCheckLibraryInstanceDependent': 'MetaDataFileCheckLibraryInstanceDependent', 'MetaDataFileCheckLibraryInstanceOrder': 'MetaDataFileCheckLibraryInstanceOrder', 'MetaDataFileCheckLibraryNoUse': 'MetaDataFileCheckLibraryNoUse', 'MetaDataFileCheckModuleFileGuidDuplication': 'MetaDataFileCheckModuleFileGuidDuplication', 'MetaDataFileCheckModuleFileGuidFormat': 'MetaDataFileCheckModuleFileGuidFormat', 'MetaDataFileCheckModuleFileNoUse': 'MetaDataFileCheckModuleFileNoUse', 'MetaDataFileCheckModuleFilePcdFormat': 'MetaDataFileCheckModuleFilePcdFormat', 'MetaDataFileCheckModuleFilePpiFormat': 'MetaDataFileCheckModuleFilePpiFormat', 'MetaDataFileCheckModuleFileProtocolFormat': 'MetaDataFileCheckModuleFileProtocolFormat', 'MetaDataFileCheckPathName': 'MetaDataFileCheckPathName', 'MetaDataFileCheckPathOfGenerateFileList': 'MetaDataFileCheckPathOfGenerateFileList', 'MetaDataFileCheckPcdDuplicate': 'MetaDataFileCheckPcdDuplicate', 'MetaDataFileCheckPcdFlash': 'MetaDataFileCheckPcdFlash', 'MetaDataFileCheckPcdNoUse': 'MetaDataFileCheckPcdNoUse', 'MetaDataFileCheckPcdType': 'MetaDataFileCheckPcdType', 'NamingConventionCheckAll': 'NamingConventionCheckAll', 'NamingConventionCheckDefineStatement': 'NamingConventionCheckDefineStatement', 'NamingConventionCheckFunctionName': 'NamingConventionCheckFunctionName', 'NamingConventionCheckIfndefStatement': 'NamingConventionCheckIfndefStatement', 'NamingConventionCheckPathName': 'NamingConventionCheckPathName', 'NamingConventionCheckSingleCharacterVariable': 'NamingConventionCheckSingleCharacterVariable', 'NamingConventionCheckTypedefStatement': 'NamingConventionCheckTypedefStatement', 'NamingConventionCheckVariableName': 'NamingConventionCheckVariableName', 'PredicateExpressionCheckAll': 'PredicateExpressionCheckAll', 'PredicateExpressionCheckBooleanValue': 'PredicateExpressionCheckBooleanValue', 'PredicateExpressionCheckComparisonNullType': 'PredicateExpressionCheckComparisonNullType', 'PredicateExpressionCheckNonBooleanOperator': 'PredicateExpressionCheckNonBooleanOperator', 'ScanOnlyDirList': 'ScanOnlyDirList', 'SkipDirList': 'SkipDirList', 'SkipFileList': 'SkipFileList', 'SmmCommParaCheckAll': 'SmmCommParaCheckAll', 'SmmCommParaCheckBufferType': 'SmmCommParaCheckBufferType', 'SpaceCheckAll': 'SpaceCheckAll', 'SpellingCheckAll': 'SpellingCheckAll', 'TokenReleaceList': 'TokenReleaceList', 'UniCheckAll': 'UniCheckAll', 'UniCheckHelpInfo': 'UniCheckHelpInfo', 'UniCheckPCDInfo': 'UniCheckPCDInfo', 'Version': 'Version'}

class Configuration(object):

    def __init__(self, Filename):
        if False:
            return 10
        self.Filename = Filename
        self.Version = 0.1
        self.CheckAll = 0
        self.AutoCorrect = 0
        self.ModifierSet = MODIFIER_SET
        self.GeneralCheckAll = 0
        self.GeneralCheckNoTab = 1
        self.GeneralCheckTabWidth = 2
        self.GeneralCheckIndentation = 1
        self.GeneralCheckIndentationWidth = 2
        self.GeneralCheckLine = 1
        self.GeneralCheckLineWidth = 120
        self.GeneralCheckNo_Asm = 1
        self.GeneralCheckNoProgma = 1
        self.GeneralCheckCarriageReturn = 1
        self.GeneralCheckFileExistence = 1
        self.GeneralCheckNonAcsii = 1
        self.GeneralCheckUni = 1
        self.GeneralCheckLineEnding = 1
        self.GeneralCheckTrailingWhiteSpaceLine = 1
        self.CFunctionLayoutCheckNoDeprecated = 1
        self.SpaceCheckAll = 1
        self.PredicateExpressionCheckAll = 0
        self.PredicateExpressionCheckBooleanValue = 1
        self.PredicateExpressionCheckNonBooleanOperator = 1
        self.PredicateExpressionCheckComparisonNullType = 1
        self.HeaderCheckAll = 0
        self.HeaderCheckFile = 1
        self.HeaderCheckFunction = 1
        self.HeaderCheckFileCommentEnd = 1
        self.HeaderCheckCFileCommentStartSpacesNum = 1
        self.HeaderCheckCFileCommentReferenceFormat = 1
        self.HeaderCheckCFileCommentLicenseFormat = 1
        self.CFunctionLayoutCheckAll = 0
        self.CFunctionLayoutCheckReturnType = 1
        self.CFunctionLayoutCheckOptionalFunctionalModifier = 1
        self.CFunctionLayoutCheckFunctionName = 1
        self.CFunctionLayoutCheckFunctionPrototype = 1
        self.CFunctionLayoutCheckFunctionBody = 1
        self.CFunctionLayoutCheckDataDeclaration = 1
        self.CFunctionLayoutCheckNoInitOfVariable = 1
        self.CFunctionLayoutCheckNoStatic = 1
        self.IncludeFileCheckAll = 0
        self.IncludeFileCheckSameName = 1
        self.IncludeFileCheckIfndefStatement = 1
        self.IncludeFileCheckData = 1
        self.DeclarationDataTypeCheckAll = 0
        self.DeclarationDataTypeCheckNoUseCType = 1
        self.DeclarationDataTypeCheckInOutModifier = 1
        self.DeclarationDataTypeCheckEFIAPIModifier = 1
        self.DeclarationDataTypeCheckEnumeratedType = 1
        self.DeclarationDataTypeCheckStructureDeclaration = 1
        self.DeclarationDataTypeCheckSameStructure = 1
        self.DeclarationDataTypeCheckUnionType = 1
        self.NamingConventionCheckAll = 0
        self.NamingConventionCheckDefineStatement = 1
        self.NamingConventionCheckTypedefStatement = 1
        self.NamingConventionCheckIfndefStatement = 1
        self.NamingConventionCheckPathName = 1
        self.NamingConventionCheckVariableName = 1
        self.NamingConventionCheckFunctionName = 1
        self.NamingConventionCheckSingleCharacterVariable = 1
        self.DoxygenCheckAll = 0
        self.DoxygenCheckFileHeader = 1
        self.DoxygenCheckFunctionHeader = 1
        self.DoxygenCheckCommentDescription = 1
        self.DoxygenCheckCommentFormat = 1
        self.DoxygenCheckCommand = 1
        self.MetaDataFileCheckAll = 0
        self.MetaDataFileCheckPathName = 1
        self.MetaDataFileCheckGenerateFileList = 1
        self.MetaDataFileCheckPathOfGenerateFileList = 'File.log'
        self.MetaDataFileCheckLibraryInstance = 1
        self.MetaDataFileCheckLibraryInstanceDependent = 1
        self.MetaDataFileCheckLibraryInstanceOrder = 1
        self.MetaDataFileCheckLibraryNoUse = 1
        self.MetaDataFileCheckLibraryDefinedInDec = 1
        self.MetaDataFileCheckBinaryInfInFdf = 1
        self.MetaDataFileCheckPcdDuplicate = 1
        self.MetaDataFileCheckPcdFlash = 1
        self.MetaDataFileCheckPcdNoUse = 1
        self.MetaDataFileCheckGuidDuplicate = 1
        self.MetaDataFileCheckModuleFileNoUse = 1
        self.MetaDataFileCheckPcdType = 1
        self.MetaDataFileCheckModuleFileGuidDuplication = 1
        self.MetaDataFileCheckModuleFileGuidFormat = 1
        self.MetaDataFileCheckModuleFileProtocolFormat = 1
        self.MetaDataFileCheckModuleFilePpiFormat = 1
        self.MetaDataFileCheckModuleFilePcdFormat = 1
        self.UniCheckAll = 0
        self.UniCheckHelpInfo = 1
        self.UniCheckPCDInfo = 1
        self.SmmCommParaCheckAll = 0
        self.SmmCommParaCheckBufferType = -1
        self.SpellingCheckAll = 0
        self.SkipDirList = []
        self.SkipFileList = []
        self.BinaryExtList = []
        self.ScanOnlyDirList = []
        self.Copyright = []
        self.TokenReleaceList = []
        self.ParseConfig()

    def ParseConfig(self):
        if False:
            return 10
        Filepath = os.path.normpath(self.Filename)
        if not os.path.isfile(Filepath):
            ErrorMsg = "Can't find configuration file '%s'" % Filepath
            EdkLogger.error('Ecc', EdkLogger.ECC_ERROR, ErrorMsg, File=Filepath)
        LineNo = 0
        for Line in open(Filepath, 'r'):
            LineNo = LineNo + 1
            Line = CleanString(Line)
            if Line != '':
                List = GetSplitValueList(Line, TAB_EQUAL_SPLIT)
                if List[0] not in _ConfigFileToInternalTranslation:
                    ErrorMsg = "Invalid configuration option '%s' was found" % List[0]
                    EdkLogger.error('Ecc', EdkLogger.ECC_ERROR, ErrorMsg, File=Filepath, Line=LineNo)
                assert _ConfigFileToInternalTranslation[List[0]] in self.__dict__
                if List[0] == 'ModifierList':
                    List[1] = GetSplitValueList(List[1], TAB_COMMA_SPLIT)
                if List[0] == 'MetaDataFileCheckPathOfGenerateFileList' and List[1] == '':
                    continue
                if List[0] == 'SkipDirList':
                    List[1] = GetSplitValueList(List[1], TAB_COMMA_SPLIT)
                if List[0] == 'SkipFileList':
                    List[1] = GetSplitValueList(List[1], TAB_COMMA_SPLIT)
                if List[0] == 'BinaryExtList':
                    List[1] = GetSplitValueList(List[1], TAB_COMMA_SPLIT)
                if List[0] == 'Copyright':
                    List[1] = GetSplitValueList(List[1], TAB_COMMA_SPLIT)
                if List[0] == 'TokenReleaceList':
                    List[1] = GetSplitValueList(List[1], TAB_COMMA_SPLIT)
                self.__dict__[_ConfigFileToInternalTranslation[List[0]]] = List[1]

    def ShowMe(self):
        if False:
            i = 10
            return i + 15
        print(self.Filename)
        for Key in self.__dict__.keys():
            print(Key, '=', self.__dict__[Key])
if __name__ == '__main__':
    myconfig = Configuration('BaseTools\\Source\\Python\\Ecc\\config.ini')
    for each in myconfig.__dict__:
        if each == 'Filename':
            continue
        assert each in _ConfigFileToInternalTranslation.values()
    for each in _ConfigFileToInternalTranslation.values():
        assert each in myconfig.__dict__