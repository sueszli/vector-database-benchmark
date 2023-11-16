"""
XmlParser
"""
import re
from Library.Xml.XmlRoutines import XmlNode
from Library.Xml.XmlRoutines import CreateXmlElement
from Library.Xml.XmlRoutines import XmlList
from Library.Xml.XmlRoutines import XmlParseFile
from Core.DistributionPackageClass import DistributionPackageClass
from Object.POM.ModuleObject import DepexObject
from Library.ParserValidate import IsValidInfMoudleType
from Library.ParserValidate import IsValidInstallPath
from Library.Misc import IsEqualList
from Library.Misc import Sdict
from Logger.StringTable import ERR_XML_INVALID_VARIABLENAME
from Logger.StringTable import ERR_XML_INVALID_LIB_SUPMODLIST
from Logger.StringTable import ERR_XML_INVALID_EXTERN_SUPARCHLIST
from Logger.StringTable import ERR_XML_INVALID_EXTERN_SUPMODLIST
from Logger.StringTable import ERR_XML_INVALID_EXTERN_SUPMODLIST_NOT_LIB
from Logger.StringTable import ERR_FILE_NAME_INVALIDE
from Logger.ToolError import PARSER_ERROR
from Logger.ToolError import FORMAT_INVALID
from Xml.CommonXml import DistributionPackageHeaderXml
from Xml.CommonXml import MiscellaneousFileXml
from Xml.CommonXml import UserExtensionsXml
from Xml.XmlParserMisc import ConvertVariableName
from Xml.XmlParserMisc import IsRequiredItemListNull
from Xml.ModuleSurfaceAreaXml import ModuleSurfaceAreaXml
from Xml.PackageSurfaceAreaXml import PackageSurfaceAreaXml
import Logger.Log as Logger

class DistributionPackageXml(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.DistP = DistributionPackageClass()
        self.Pkg = ''

    def ValidateDistributionPackage(self):
        if False:
            print('Hello World!')
        XmlTreeLevel = ['DistributionPackage']
        if self.DistP:
            XmlTreeLevel = ['DistributionPackage', '']
            CheckDict = {'DistributionHeader': self.DistP.Header}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
            if self.DistP.Header:
                DpHeader = self.DistP.Header
                XmlTreeLevel = ['DistributionPackage', 'DistributionHeader']
                CheckDict = Sdict()
                if DpHeader.GetAbstract():
                    DPAbstract = DpHeader.GetAbstract()[0][1]
                else:
                    DPAbstract = ''
                if DpHeader.GetCopyright():
                    DPCopyright = DpHeader.GetCopyright()[0][1]
                else:
                    DPCopyright = ''
                if DpHeader.GetLicense():
                    DPLicense = DpHeader.GetLicense()[0][1]
                else:
                    DPLicense = ''
                CheckDict['Name'] = DpHeader.GetName()
                CheckDict['GUID'] = DpHeader.GetGuid()
                CheckDict['Version'] = DpHeader.GetVersion()
                CheckDict['Copyright'] = DPCopyright
                CheckDict['License'] = DPLicense
                CheckDict['Abstract'] = DPAbstract
                CheckDict['Vendor'] = DpHeader.GetVendor()
                CheckDict['Date'] = DpHeader.GetDate()
                CheckDict['XmlSpecification'] = DpHeader.GetXmlSpecification()
                IsRequiredItemListNull(CheckDict, XmlTreeLevel)
            else:
                XmlTreeLevel = ['DistributionPackage', 'DistributionHeader']
                CheckDict = CheckDict = {'DistributionHeader': ''}
                IsRequiredItemListNull(CheckDict, XmlTreeLevel)
            for Key in self.DistP.PackageSurfaceArea:
                ValidatePackageSurfaceArea(self.DistP.PackageSurfaceArea[Key])
            for Key in self.DistP.ModuleSurfaceArea:
                ValidateMS(self.DistP.ModuleSurfaceArea[Key], ['DistributionPackage', 'ModuleSurfaceArea'])
            if self.DistP.Tools:
                XmlTreeLevel = ['DistributionPackage', 'Tools', 'Header']
                CheckDict = {'Name': self.DistP.Tools.GetName()}
                IsRequiredItemListNull(CheckDict, XmlTreeLevel)
                if not self.DistP.Tools.GetFileList():
                    XmlTreeLevel = ['DistributionPackage', 'Tools']
                    CheckDict = {'FileName': None}
                    IsRequiredItemListNull(CheckDict, XmlTreeLevel)
                for Item in self.DistP.Tools.GetFileList():
                    XmlTreeLevel = ['DistributionPackage', 'Tools']
                    CheckDict = {'FileName': Item.GetURI()}
                    IsRequiredItemListNull(CheckDict, XmlTreeLevel)
            if self.DistP.MiscellaneousFiles:
                XmlTreeLevel = ['DistributionPackage', 'MiscellaneousFiles', 'Header']
                CheckDict = {'Name': self.DistP.MiscellaneousFiles.GetName()}
                IsRequiredItemListNull(CheckDict, XmlTreeLevel)
                if not self.DistP.MiscellaneousFiles.GetFileList():
                    XmlTreeLevel = ['DistributionPackage', 'MiscellaneousFiles']
                    CheckDict = {'FileName': None}
                    IsRequiredItemListNull(CheckDict, XmlTreeLevel)
                for Item in self.DistP.MiscellaneousFiles.GetFileList():
                    XmlTreeLevel = ['DistributionPackage', 'MiscellaneousFiles']
                    CheckDict = {'FileName': Item.GetURI()}
                    IsRequiredItemListNull(CheckDict, XmlTreeLevel)
            for Item in self.DistP.UserExtensions:
                XmlTreeLevel = ['DistributionPackage', 'UserExtensions']
                CheckDict = {'UserId': Item.GetUserID()}
                IsRequiredItemListNull(CheckDict, XmlTreeLevel)

    def FromXml(self, Filename=None):
        if False:
            for i in range(10):
                print('nop')
        if Filename is not None:
            self.DistP = DistributionPackageClass()
            self.Pkg = XmlParseFile(Filename)
            Tmp = DistributionPackageHeaderXml()
            DistributionPackageHeader = Tmp.FromXml(XmlNode(self.Pkg, '/DistributionPackage/DistributionHeader'), 'DistributionHeader')
            self.DistP.Header = DistributionPackageHeader
            for Item in XmlList(self.Pkg, '/DistributionPackage/PackageSurfaceArea'):
                Psa = PackageSurfaceAreaXml()
                Package = Psa.FromXml(Item, 'PackageSurfaceArea')
                self.DistP.PackageSurfaceArea[Package.GetGuid(), Package.GetVersion(), Package.GetPackagePath()] = Package
            for Item in XmlList(self.Pkg, '/DistributionPackage/ModuleSurfaceArea'):
                Msa = ModuleSurfaceAreaXml()
                Module = Msa.FromXml(Item, 'ModuleSurfaceArea', True)
                ModuleKey = (Module.GetGuid(), Module.GetVersion(), Module.GetName(), Module.GetModulePath())
                self.DistP.ModuleSurfaceArea[ModuleKey] = Module
            Tmp = MiscellaneousFileXml()
            self.DistP.Tools = Tmp.FromXml2(XmlNode(self.Pkg, '/DistributionPackage/Tools'), 'Tools')
            Tmp = MiscellaneousFileXml()
            self.DistP.MiscellaneousFiles = Tmp.FromXml2(XmlNode(self.Pkg, '/DistributionPackage/MiscellaneousFiles'), 'MiscellaneousFiles')
            for Item in XmlList(self.Pkg, '/DistributionPackage/UserExtensions'):
                Tmp = UserExtensionsXml()
                self.DistP.UserExtensions.append(Tmp.FromXml2(Item, 'UserExtensions'))
            self.ValidateDistributionPackage()
            return self.DistP

    def ToXml(self, DistP):
        if False:
            for i in range(10):
                print('nop')
        if self.DistP:
            pass
        if DistP is not None:
            Attrs = [['xmlns', 'http://www.uefi.org/2011/1.1'], ['xmlns:xsi', 'http:/www.w3.org/2001/XMLSchema-instance']]
            Root = CreateXmlElement('DistributionPackage', '', [], Attrs)
            Tmp = DistributionPackageHeaderXml()
            Root.appendChild(Tmp.ToXml(DistP.Header, 'DistributionHeader'))
            for Package in DistP.PackageSurfaceArea.values():
                Psa = PackageSurfaceAreaXml()
                DomPackage = Psa.ToXml(Package)
                Root.appendChild(DomPackage)
            for Module in DistP.ModuleSurfaceArea.values():
                Msa = ModuleSurfaceAreaXml()
                DomModule = Msa.ToXml(Module)
                Root.appendChild(DomModule)
            Tmp = MiscellaneousFileXml()
            ToolNode = Tmp.ToXml2(DistP.Tools, 'Tools')
            if ToolNode is not None:
                Root.appendChild(ToolNode)
            Tmp = MiscellaneousFileXml()
            MiscFileNode = Tmp.ToXml2(DistP.MiscellaneousFiles, 'MiscellaneousFiles')
            if MiscFileNode is not None:
                Root.appendChild(MiscFileNode)
            XmlContent = Root.toprettyxml(indent='  ')
            XmlContent = re.sub('[\\s\\r\\n]*<[^<>=]*/>', '', XmlContent)
            XmlContent = re.sub('[\\s\\r\\n]*<HelpText Lang="en-US"/>', '', XmlContent)
            XmlContent = re.sub('[\\s\\r\\n]*SupArchList[\\s\\r\\n]*=[\\s\\r\\n]*"[\\s\\r\\n]*COMMON[\\s\r\n]*"', '', XmlContent)
            XmlContent = re.sub('[\\s\\r\\n]*SupArchList[\\s\\r\\n]*=[\\s\\r\\n]*"[\\s\\r\\n]*common[\\s\r\n]*"', '', XmlContent)
            XmlContent = re.sub('[\\s\\r\\n]*<SupArchList>[\\s\\r\\n]*COMMON[\\s\\r\\n]*</SupArchList>[\\s\r\n]*', '', XmlContent)
            XmlContent = re.sub('[\\s\\r\\n]*<SupArchList>[\\s\\r\\n]*common[\\s\r\n]*</SupArchList>[\\s\r\n]*', '', XmlContent)
            XmlContent = re.sub('[\\s\\r\\n]*SupModList[\\s\\r\\n]*=[\\s\\r\\n]*"[\\s\\r\\n]*COMMON[\\s\r\n]*"', '', XmlContent)
            XmlContent = re.sub('[\\s\\r\\n]*SupModList[\\s\\r\\n]*=[\\s\\r\\n]*"[\\s\\r\\n]*common[\\s\r\n]*"', '', XmlContent)
            return XmlContent
        return ''

def ValidateMS(Module, TopXmlTreeLevel):
    if False:
        return 10
    ValidateMS1(Module, TopXmlTreeLevel)
    ValidateMS2(Module, TopXmlTreeLevel)
    ValidateMS3(Module, TopXmlTreeLevel)

def ValidateMS1(Module, TopXmlTreeLevel):
    if False:
        i = 10
        return i + 15
    XmlTreeLevel = TopXmlTreeLevel + ['Guids']
    for Item in Module.GetGuidList():
        if Item is None:
            CheckDict = {'GuidCName': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['Guids', 'GuidCName']
    for Item in Module.GetGuidList():
        CheckDict = {'CName': Item.GetCName(), 'GuidType': Item.GetGuidTypeList(), 'Usage': Item.GetUsage()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
        if Item.GetVariableName():
            Result = ConvertVariableName(Item.GetVariableName())
            if Result is None:
                Msg = '->'.join((Node for Node in XmlTreeLevel))
                ErrorMsg = ERR_XML_INVALID_VARIABLENAME % (Item.GetVariableName(), Item.GetCName(), Msg)
                Logger.Error('\nUPT', PARSER_ERROR, ErrorMsg, RaiseError=True)
            else:
                Item.SetVariableName(Result)
    XmlTreeLevel = TopXmlTreeLevel + ['Protocols']
    for Item in Module.GetProtocolList():
        if Item is None:
            CheckDict = {'Protocol': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['Protocols', 'Protocol']
    for Item in Module.GetProtocolList():
        CheckDict = {'CName': Item.GetCName(), 'Usage': Item.GetUsage()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['PPIs']
    for Item in Module.GetPpiList():
        if Item is None:
            CheckDict = {'Ppi': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['PPIs', 'Ppi']
    for Item in Module.GetPpiList():
        CheckDict = {'CName': Item.GetCName(), 'Usage': Item.GetUsage()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['PcdCoded']
    for Item in Module.GetPcdList():
        if Item is None:
            CheckDict = {'PcdEntry': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['PcdCoded', 'PcdEntry']
    for Item in Module.GetPcdList():
        CheckDict = {'TokenSpaceGuidCname': Item.GetTokenSpaceGuidCName(), 'CName': Item.GetCName(), 'PcdUsage': Item.GetValidUsage(), 'PcdItemType': Item.GetItemType()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['Externs']
    for Item in Module.GetExternList():
        if Item is None:
            CheckDict = {'Extern': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    for Item in Module.GetExternList():
        if len(Item.SupArchList) > 0:
            if not IsEqualList(Item.SupArchList, Module.SupArchList):
                Logger.Error('\nUPT', PARSER_ERROR, ERR_XML_INVALID_EXTERN_SUPARCHLIST % (str(Item.SupArchList), str(Module.SupArchList)), RaiseError=True)
    XmlTreeLevel = TopXmlTreeLevel + ['UserExtensions']
    for Item in Module.GetUserExtensionList():
        CheckDict = {'UserId': Item.GetUserID(), 'Identifier': Item.GetIdentifier()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['MiscellaneousFiles']
    for Item in Module.GetMiscFileList():
        if not Item.GetFileList():
            CheckDict = {'Filename': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
        for File in Item.GetFileList():
            CheckDict = {'Filename': File.GetURI()}

def ValidateMS2(Module, TopXmlTreeLevel):
    if False:
        for i in range(10):
            print('nop')
    XmlTreeLevel = TopXmlTreeLevel + ['Header']
    CheckDict = Sdict()
    CheckDict['Name'] = Module.GetName()
    CheckDict['BaseName'] = Module.GetBaseName()
    CheckDict['GUID'] = Module.GetGuid()
    CheckDict['Version'] = Module.GetVersion()
    IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['ModuleProperties']
    CheckDict = {'ModuleType': Module.GetModuleType(), 'Path': Module.GetModulePath()}
    IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    if not IsValidInstallPath(Module.GetModulePath()):
        Logger.Error('UPT', FORMAT_INVALID, ERR_FILE_NAME_INVALIDE % Module.GetModulePath())
    XmlTreeLevel = TopXmlTreeLevel + ['ModuleProperties'] + ['BootMode']
    for Item in Module.GetBootModeList():
        CheckDict = {'Usage': Item.GetUsage(), 'SupportedBootModes': Item.GetSupportedBootModes()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['ModuleProperties'] + ['Event']
    for Item in Module.GetEventList():
        CheckDict = {'Usage': Item.GetUsage(), 'EventType': Item.GetEventType()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['ModuleProperties'] + ['HOB']
    for Item in Module.GetHobList():
        CheckDict = {'Usage': Item.GetUsage(), 'HobType': Item.GetHobType()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    if Module.ModuleType == 'UEFI_RUNTIME_DRIVER':
        Module.ModuleType = 'DXE_RUNTIME_DRIVER'
        DxeObj = DepexObject()
        DxeObj.SetDepex('gEfiBdsArchProtocolGuid AND \ngEfiCpuArchProtocolGuid AND\n' + 'gEfiMetronomeArchProtocolGuid AND \ngEfiMonotonicCounterArchProtocolGuid AND\n' + 'gEfiRealTimeClockArchProtocolGuid AND \ngEfiResetArchProtocolGuid AND\n' + 'gEfiRuntimeArchProtocolGuid AND \ngEfiSecurityArchProtocolGuid AND\n' + 'gEfiTimerArchProtocolGuid AND \ngEfiVariableWriteArchProtocolGuid AND\n' + 'gEfiVariableArchProtocolGuid AND \ngEfiWatchdogTimerArchProtocolGuid')
        DxeObj.SetModuleType(['DXE_RUNTIME_DRIVER'])
        Module.PeiDepex = []
        Module.DxeDepex = []
        Module.SmmDepex = []
        Module.DxeDepex.append(DxeObj)
    XmlTreeLevel = TopXmlTreeLevel + ['LibraryClassDefinitions']
    for Item in Module.GetLibraryClassList():
        if Item is None:
            CheckDict = {'LibraryClass': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['LibraryClassDefinitions', 'LibraryClass']
    IsLibraryModule = False
    LibrarySupModList = []
    for Item in Module.GetLibraryClassList():
        CheckDict = {'Keyword': Item.GetLibraryClass(), 'Usage': Item.GetUsage()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
        if len(Item.SupModuleList) > 0:
            for SupModule in Item.SupModuleList:
                if not IsValidInfMoudleType(SupModule):
                    Logger.Error('\nUPT', PARSER_ERROR, ERR_XML_INVALID_LIB_SUPMODLIST % (Item.LibraryClass, str(SupModule)), RaiseError=True)
        if Item.Usage == 'PRODUCES' or Item.Usage == 'SOMETIMES_PRODUCES':
            IsLibraryModule = True
            LibrarySupModList = Item.SupModuleList
    if IsLibraryModule:
        for Item in Module.GetExternList():
            if Item.Constructor or Item.Destructor:
                if hasattr(Item, 'SupModList') and len(Item.SupModList) > 0 and (not IsEqualList(Item.SupModList, LibrarySupModList)):
                    Logger.Error('\nUPT', PARSER_ERROR, ERR_XML_INVALID_EXTERN_SUPMODLIST % (str(Item.SupModList), str(LibrarySupModList)), RaiseError=True)
    if not IsLibraryModule:
        for Item in Module.GetExternList():
            if hasattr(Item, 'SupModList') and len(Item.SupModList) > 0 and (not IsEqualList(Item.SupModList, [Module.ModuleType])):
                Logger.Error('\nUPT', PARSER_ERROR, ERR_XML_INVALID_EXTERN_SUPMODLIST_NOT_LIB % (str(Module.ModuleType), str(Item.SupModList)), RaiseError=True)
    XmlTreeLevel = TopXmlTreeLevel + ['SourceFiles']
    for Item in Module.GetSourceFileList():
        if Item is None:
            CheckDict = {'Filename': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['SourceFiles']
    for Item in Module.GetSourceFileList():
        CheckDict = {'Filename': Item.GetSourceFile()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    for ItemCount in range(len(Module.GetBinaryFileList())):
        Item = Module.GetBinaryFileList()[ItemCount]
        if Item and len(Item.FileNamList) > 0 and (Item.FileNamList[0].FileType == 'FREEFORM'):
            Item.FileNamList[0].FileType = 'SUBTYPE_GUID'
            Module.GetBinaryFileList()[ItemCount] = Item

def ValidateMS3(Module, TopXmlTreeLevel):
    if False:
        i = 10
        return i + 15
    XmlTreeLevel = TopXmlTreeLevel + ['PackageDependencies']
    for Item in Module.GetPackageDependencyList():
        if Item is None:
            CheckDict = {'Package': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['PackageDependencies', 'Package']
    for Item in Module.GetPackageDependencyList():
        CheckDict = {'GUID': Item.GetGuid()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    for Item in Module.GetBinaryFileList():
        if Item is None:
            XmlTreeLevel = TopXmlTreeLevel + ['BinaryFiles']
            CheckDict = {'BinaryFile': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
        if not Item.GetFileNameList():
            XmlTreeLevel = TopXmlTreeLevel + ['BinaryFiles', 'BinaryFile']
            CheckDict = {'Filename': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
        XmlTreeLevel = TopXmlTreeLevel + ['BinaryFiles', 'BinaryFile']
        for File in Item.GetFileNameList():
            CheckDict = {'Filename': File.GetFilename(), 'FileType': File.GetFileType()}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
        for AsBuilt in Item.GetAsBuiltList():
            if len(AsBuilt.LibraryInstancesList) == 1 and (not AsBuilt.LibraryInstancesList[0]):
                CheckDict = {'GUID': ''}
                XmlTreeLevel = TopXmlTreeLevel + ['BinaryFiles', 'BinaryFile', 'AsBuilt', 'LibraryInstances']
                IsRequiredItemListNull(CheckDict, XmlTreeLevel)
            for LibItem in AsBuilt.LibraryInstancesList:
                CheckDict = {'Guid': LibItem.Guid, 'Version': LibItem.Version}
                XmlTreeLevel = TopXmlTreeLevel + ['BinaryFiles', 'BinaryFile', 'AsBuilt', 'LibraryInstances']
                IsRequiredItemListNull(CheckDict, XmlTreeLevel)
            for PatchPcdItem in AsBuilt.PatchPcdList:
                CheckDict = {'TokenSpaceGuidValue': PatchPcdItem.TokenSpaceGuidValue, 'PcdCName': PatchPcdItem.PcdCName, 'Token': PatchPcdItem.Token, 'DatumType': PatchPcdItem.DatumType, 'Value': PatchPcdItem.DefaultValue, 'Offset': PatchPcdItem.Offset}
                XmlTreeLevel = TopXmlTreeLevel + ['BinaryFiles', 'BinaryFile', 'AsBuilt', 'PatchPcdValue']
                IsRequiredItemListNull(CheckDict, XmlTreeLevel)
                for PcdErrorItem in PatchPcdItem.PcdErrorsList:
                    CheckDict = {'ErrorNumber': PcdErrorItem.ErrorNumber}
                    XmlTreeLevel = TopXmlTreeLevel + ['BinaryFiles', 'BinaryFile', 'AsBuilt', 'PatchPcdValue', 'PcdError']
                    IsRequiredItemListNull(CheckDict, XmlTreeLevel)
            for PcdExItem in AsBuilt.PcdExValueList:
                CheckDict = {'TokenSpaceGuidValue': PcdExItem.TokenSpaceGuidValue, 'Token': PcdExItem.Token, 'DatumType': PcdExItem.DatumType}
                XmlTreeLevel = TopXmlTreeLevel + ['BinaryFiles', 'BinaryFile', 'AsBuilt', 'PcdExValue']
                IsRequiredItemListNull(CheckDict, XmlTreeLevel)
                for PcdErrorItem in PcdExItem.PcdErrorsList:
                    CheckDict = {'ErrorNumber': PcdErrorItem.ErrorNumber}
                    XmlTreeLevel = TopXmlTreeLevel + ['BinaryFiles', 'BinaryFile', 'AsBuilt', 'PcdExValue', 'PcdError']
                    IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['SmmDepex']
    for Item in Module.GetSmmDepex():
        CheckDict = {'Expression': Item.GetDepex()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['PeiDepex']
    for Item in Module.GetPeiDepex():
        CheckDict = {'Expression': Item.GetDepex()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['DxeDepex']
    for Item in Module.GetDxeDepex():
        CheckDict = {'Expression': Item.GetDepex()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = TopXmlTreeLevel + ['UserExtensions']
    for Item in Module.GetUserExtensionList():
        CheckDict = {'UserId': Item.GetUserID(), 'Identifier': Item.GetIdentifier()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)

def ValidatePS1(Package):
    if False:
        print('Hello World!')
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'Header']
    CheckDict = Sdict()
    CheckDict['Name'] = Package.GetName()
    CheckDict['BaseName'] = Package.GetBaseName()
    CheckDict['GUID'] = Package.GetGuid()
    CheckDict['Version'] = Package.GetVersion()
    CheckDict['PackagePath'] = Package.GetPackagePath()
    IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    if not IsValidInstallPath(Package.GetPackagePath()):
        Logger.Error('UPT', FORMAT_INVALID, ERR_FILE_NAME_INVALIDE % Package.GetPackagePath())
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'ClonedFrom']
    for Item in Package.GetClonedFromList():
        if Item is None:
            CheckDict = Sdict()
            CheckDict['GUID'] = ''
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
        CheckDict = Sdict()
        CheckDict['GUID'] = Item.GetPackageGuid()
        CheckDict['Version'] = Item.GetPackageVersion()
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'LibraryClassDeclarations']
    for Item in Package.GetLibraryClassList():
        if Item is None:
            CheckDict = {'LibraryClass': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'LibraryClassDeclarations', 'LibraryClass']
    for Item in Package.GetLibraryClassList():
        CheckDict = {'Keyword': Item.GetLibraryClass(), 'HeaderFile': Item.GetIncludeHeader()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'IndustryStandardIncludes']
    for Item in Package.GetStandardIncludeFileList():
        if Item is None:
            CheckDict = {'IndustryStandardHeader': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'IndustryStandardIncludes', 'IndustryStandardHeader']
    for Item in Package.GetStandardIncludeFileList():
        CheckDict = {'HeaderFile': Item.GetFilePath()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'PackageIncludes']
    for Item in Package.GetPackageIncludeFileList():
        if Item is None:
            CheckDict = {'PackageHeader': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'PackageIncludes', 'PackageHeader']
    for Item in Package.GetPackageIncludeFileList():
        CheckDict = {'HeaderFile': Item.GetFilePath()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)

def ValidatePS2(Package):
    if False:
        return 10
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'Modules', 'ModuleSurfaceArea']
    for Item in Package.GetModuleDict().values():
        ValidateMS(Item, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'GuidDeclarations']
    for Item in Package.GetGuidList():
        if Item is None:
            CheckDict = {'Entry': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'GuidDeclarations', 'Entry']
    for Item in Package.GetGuidList():
        CheckDict = {'CName': Item.GetCName(), 'GuidValue': Item.GetGuid()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'ProtocolDeclarations']
    for Item in Package.GetProtocolList():
        if Item is None:
            CheckDict = {'Entry': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'ProtocolDeclarations', 'Entry']
    for Item in Package.GetProtocolList():
        CheckDict = {'CName': Item.GetCName(), 'GuidValue': Item.GetGuid()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'PpiDeclarations']
    for Item in Package.GetPpiList():
        if Item is None:
            CheckDict = {'Entry': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'PpiDeclarations', 'Entry']
    for Item in Package.GetPpiList():
        CheckDict = {'CName': Item.GetCName(), 'GuidValue': Item.GetGuid()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'PcdDeclarations']
    for Item in Package.GetPcdList():
        if Item is None:
            CheckDict = {'PcdEntry': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'PcdDeclarations', 'PcdEntry']
    for Item in Package.GetPcdList():
        CheckDict = {'TokenSpaceGuidCname': Item.GetTokenSpaceGuidCName(), 'Token': Item.GetToken(), 'CName': Item.GetCName(), 'DatumType': Item.GetDatumType(), 'ValidUsage': Item.GetValidUsage(), 'DefaultValue': Item.GetDefaultValue()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'UserExtensions']
    for Item in Package.GetUserExtensionList():
        CheckDict = {'UserId': Item.GetUserID(), 'Identifier': Item.GetIdentifier()}
        IsRequiredItemListNull(CheckDict, XmlTreeLevel)
    XmlTreeLevel = ['DistributionPackage', 'PackageSurfaceArea', 'MiscellaneousFiles']
    for Item in Package.GetMiscFileList():
        if not Item.GetFileList():
            CheckDict = {'Filename': ''}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)
        for File in Item.GetFileList():
            CheckDict = {'Filename': File.GetURI()}
            IsRequiredItemListNull(CheckDict, XmlTreeLevel)

def ValidatePackageSurfaceArea(Package):
    if False:
        while True:
            i = 10
    ValidatePS1(Package)
    ValidatePS2(Package)