"""
PackageSurfaceAreaXml
"""
from xml.dom import minidom
from Library.StringUtils import GetStringOfList
from Library.Xml.XmlRoutines import XmlElement
from Library.Xml.XmlRoutines import XmlNode
from Library.Xml.XmlRoutines import XmlList
from Library.Xml.XmlRoutines import CreateXmlElement
from Object.POM.CommonObject import IncludeObject
from Object.POM.CommonObject import TextObject
from Object.POM.PackageObject import PackageObject
from Xml.CommonXml import ClonedFromXml
from Xml.CommonXml import PackageHeaderXml
from Xml.CommonXml import HelpTextXml
from Xml.CommonXml import CommonDefinesXml
from Xml.CommonXml import LibraryClassXml
from Xml.CommonXml import UserExtensionsXml
from Xml.CommonXml import MiscellaneousFileXml
from Xml.GuidProtocolPpiXml import GuidXml
from Xml.GuidProtocolPpiXml import ProtocolXml
from Xml.GuidProtocolPpiXml import PpiXml
from Xml.ModuleSurfaceAreaXml import ModuleSurfaceAreaXml
from Xml.PcdXml import PcdEntryXml

class IndustryStandardHeaderXml(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.HeaderFile = ''
        self.HelpText = []

    def FromXml(self, Item, Key):
        if False:
            i = 10
            return i + 15
        self.HeaderFile = XmlElement(Item, '%s/HeaderFile' % Key)
        for HelpTextItem in XmlList(Item, '%s/HelpText' % Key):
            HelpTextObj = HelpTextXml()
            HelpTextObj.FromXml(HelpTextItem, '%s/HelpText' % Key)
            self.HelpText.append(HelpTextObj)
        Include = IncludeObject()
        Include.SetFilePath(self.HeaderFile)
        HelpTxt = TextObject()
        HelpTxt.SetString(self.HelpText)
        Include.SetHelpText(HelpTxt)
        return Include

    def ToXml(self, IndustryStandardHeader, Key):
        if False:
            print('Hello World!')
        if self.HeaderFile:
            pass
        AttributeList = []
        NodeList = [['HeaderFile', IndustryStandardHeader.GetFilePath()]]
        Root = CreateXmlElement('%s' % Key, '', NodeList, AttributeList)
        return Root

    def __str__(self):
        if False:
            i = 10
            return i + 15
        Str = 'HeaderFile = %s' % self.HeaderFile
        for Item in self.HelpText:
            Str = Str + '\n\t' + str(Item)
        return Str

class PackageIncludeHeaderXml(object):

    def __init__(self):
        if False:
            return 10
        self.HeaderFile = ''
        self.CommonDefines = CommonDefinesXml()
        self.HelpText = []

    def FromXml(self, Item, Key):
        if False:
            print('Hello World!')
        self.HeaderFile = XmlElement(Item, '%s/HeaderFile' % Key)
        self.CommonDefines.FromXml(XmlNode(Item, '%s/HeaderFile' % Key), 'HeaderFile')
        for HelpTextItem in XmlList(Item, '%s/HelpText' % Key):
            HelpTextObj = HelpTextXml()
            HelpTextObj.FromXml(HelpTextItem, '%s/HelpText' % Key)
            self.HelpText.append(HelpTextObj)
        Include = IncludeObject()
        Include.SetFilePath(self.HeaderFile)
        Include.SetSupArchList(self.CommonDefines.SupArchList)
        HelpTxt = TextObject()
        HelpTxt.SetString(self.HelpText)
        Include.SetHelpText(HelpTxt)
        return Include

    def ToXml(self, PackageIncludeHeader, Key):
        if False:
            while True:
                i = 10
        if self.HeaderFile:
            pass
        AttributeList = [['SupArchList', GetStringOfList(PackageIncludeHeader.GetSupArchList())], ['SupModList', GetStringOfList(PackageIncludeHeader.GetSupModuleList())]]
        HeaderFileNode = CreateXmlElement('HeaderFile', PackageIncludeHeader.FilePath, [], AttributeList)
        NodeList = [HeaderFileNode]
        for Item in PackageIncludeHeader.GetHelpTextList():
            Tmp = HelpTextXml()
            NodeList.append(Tmp.ToXml(Item))
        Root = CreateXmlElement('%s' % Key, '', NodeList, [])
        return Root

    def __str__(self):
        if False:
            return 10
        Str = 'HeaderFile = %s\n\t%s' % (self.HeaderFile, self.CommonDefines)
        for Item in self.HelpText:
            Str = Str + '\n\t' + str(Item)
        return Str

class PcdCheckXml(object):

    def __init__(self):
        if False:
            return 10
        self.PcdCheck = ''

    def FromXml(self, Item, Key):
        if False:
            while True:
                i = 10
        if Key:
            pass
        self.PcdCheck = XmlElement(Item, 'PcdCheck')
        return self.PcdCheck

    def ToXml(self, PcdCheck, Key):
        if False:
            i = 10
            return i + 15
        if self.PcdCheck:
            pass
        Root = CreateXmlElement('%s' % Key, PcdCheck, [], [])
        return Root

    def __str__(self):
        if False:
            print('Hello World!')
        return 'PcdCheck = %s' % self.PcdCheck

class PackageSurfaceAreaXml(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.Package = None

    def FromXml(self, Item, Key):
        if False:
            print('Hello World!')
        if Key:
            pass
        Package = PackageObject()
        Tmp = PackageHeaderXml()
        Tmp.FromXml(XmlNode(Item, '/PackageSurfaceArea/Header'), 'Header', Package)
        Tmp = ClonedFromXml()
        if XmlNode(Item, '/PackageSurfaceArea/ClonedFrom'):
            ClonedFrom = Tmp.FromXml(XmlNode(Item, '/PackageSurfaceArea/ClonedFrom'), 'ClonedFrom')
            Package.SetClonedFromList([ClonedFrom])
        for SubItem in XmlList(Item, '/PackageSurfaceArea/LibraryClassDeclarations/LibraryClass'):
            Tmp = LibraryClassXml()
            LibraryClass = Tmp.FromXml(SubItem, 'LibraryClass')
            Package.SetLibraryClassList(Package.GetLibraryClassList() + [LibraryClass])
        if XmlList(Item, '/PackageSurfaceArea/LibraryClassDeclarations') and (not XmlList(Item, '/PackageSurfaceArea/LibraryClassDeclarations/LibraryClass')):
            Package.SetLibraryClassList([None])
        for SubItem in XmlList(Item, '/PackageSurfaceArea/IndustryStandardIncludes/IndustryStandardHeader'):
            Tmp = IndustryStandardHeaderXml()
            Include = Tmp.FromXml(SubItem, 'IndustryStandardHeader')
            Package.SetStandardIncludeFileList(Package.GetStandardIncludeFileList() + [Include])
        if XmlList(Item, '/PackageSurfaceArea/IndustryStandardIncludes') and (not XmlList(Item, '/PackageSurfaceArea/IndustryStandardIncludes/IndustryStandardHeader')):
            Package.SetStandardIncludeFileList([None])
        for SubItem in XmlList(Item, '/PackageSurfaceArea/PackageIncludes/PackageHeader'):
            Tmp = PackageIncludeHeaderXml()
            Include = Tmp.FromXml(SubItem, 'PackageHeader')
            Package.SetPackageIncludeFileList(Package.GetPackageIncludeFileList() + [Include])
        if XmlList(Item, '/PackageSurfaceArea/PackageIncludes') and (not XmlList(Item, '/PackageSurfaceArea/PackageIncludes/PackageHeader')):
            Package.SetPackageIncludeFileList([None])
        for SubItem in XmlList(Item, '/PackageSurfaceArea/GuidDeclarations/Entry'):
            Tmp = GuidXml('Package')
            GuidProtocolPpi = Tmp.FromXml(SubItem, 'Entry')
            Package.SetGuidList(Package.GetGuidList() + [GuidProtocolPpi])
        if XmlList(Item, '/PackageSurfaceArea/GuidDeclarations') and (not XmlList(Item, '/PackageSurfaceArea/GuidDeclarations/Entry')):
            Package.SetGuidList([None])
        for SubItem in XmlList(Item, '/PackageSurfaceArea/ProtocolDeclarations/Entry'):
            Tmp = ProtocolXml('Package')
            GuidProtocolPpi = Tmp.FromXml(SubItem, 'Entry')
            Package.SetProtocolList(Package.GetProtocolList() + [GuidProtocolPpi])
        if XmlList(Item, '/PackageSurfaceArea/ProtocolDeclarations') and (not XmlList(Item, '/PackageSurfaceArea/ProtocolDeclarations/Entry')):
            Package.SetProtocolList([None])
        for SubItem in XmlList(Item, '/PackageSurfaceArea/PpiDeclarations/Entry'):
            Tmp = PpiXml('Package')
            GuidProtocolPpi = Tmp.FromXml(SubItem, 'Entry')
            Package.SetPpiList(Package.GetPpiList() + [GuidProtocolPpi])
        if XmlList(Item, '/PackageSurfaceArea/PpiDeclarations') and (not XmlList(Item, '/PackageSurfaceArea/PpiDeclarations/Entry')):
            Package.SetPpiList([None])
        for SubItem in XmlList(Item, '/PackageSurfaceArea/PcdDeclarations/PcdEntry'):
            Tmp = PcdEntryXml()
            PcdEntry = Tmp.FromXml2(SubItem, 'PcdEntry')
            Package.SetPcdList(Package.GetPcdList() + [PcdEntry])
            for PcdErrorObj in PcdEntry.GetPcdErrorsList():
                PcdErrorMessageList = PcdErrorObj.GetErrorMessageList()
                if PcdErrorMessageList:
                    Package.PcdErrorCommentDict[PcdEntry.GetTokenSpaceGuidCName(), PcdErrorObj.GetErrorNumber()] = PcdErrorMessageList
        if XmlList(Item, '/PackageSurfaceArea/PcdDeclarations') and (not XmlList(Item, '/PackageSurfaceArea/PcdDeclarations/PcdEntry')):
            Package.SetPcdList([None])
        for SubItem in XmlList(Item, '/PackageSurfaceArea/PcdRelationshipChecks/PcdCheck'):
            Tmp = PcdCheckXml()
            PcdCheck = Tmp.FromXml(SubItem, 'PcdCheck')
            Package.PcdChecks.append(PcdCheck)
        for SubItem in XmlList(Item, '/PackageSurfaceArea/Modules/ModuleSurfaceArea'):
            Tmp = ModuleSurfaceAreaXml()
            Module = Tmp.FromXml(SubItem, 'ModuleSurfaceArea')
            ModuleDictKey = (Module.GetGuid(), Module.GetVersion(), Module.GetName(), Module.GetModulePath())
            Package.ModuleDict[ModuleDictKey] = Module
        Tmp = MiscellaneousFileXml()
        MiscFileList = Tmp.FromXml(XmlNode(Item, '/PackageSurfaceArea/MiscellaneousFiles'), 'MiscellaneousFiles')
        if MiscFileList:
            Package.SetMiscFileList([MiscFileList])
        else:
            Package.SetMiscFileList([])
        for Item in XmlList(Item, '/PackageSurfaceArea/UserExtensions'):
            Tmp = UserExtensionsXml()
            UserExtension = Tmp.FromXml(Item, 'UserExtensions')
            Package.UserExtensionList.append(UserExtension)
        self.Package = Package
        return self.Package

    def ToXml(self, Package):
        if False:
            while True:
                i = 10
        if self.Package:
            pass
        DomPackage = minidom.Document().createElement('PackageSurfaceArea')
        Tmp = PackageHeaderXml()
        DomPackage.appendChild(Tmp.ToXml(Package, 'Header'))
        Tmp = ClonedFromXml()
        if Package.GetClonedFromList() != []:
            DomPackage.appendChild(Tmp.ToXml(Package.GetClonedFromList[0], 'ClonedFrom'))
        LibraryClassNode = CreateXmlElement('LibraryClassDeclarations', '', [], [])
        for LibraryClass in Package.GetLibraryClassList():
            Tmp = LibraryClassXml()
            LibraryClassNode.appendChild(Tmp.ToXml(LibraryClass, 'LibraryClass'))
        DomPackage.appendChild(LibraryClassNode)
        IndustryStandardHeaderNode = CreateXmlElement('IndustryStandardIncludes', '', [], [])
        for Include in Package.GetStandardIncludeFileList():
            Tmp = IndustryStandardHeaderXml()
            IndustryStandardHeaderNode.appendChild(Tmp.ToXml(Include, 'IndustryStandardHeader'))
        DomPackage.appendChild(IndustryStandardHeaderNode)
        PackageIncludeHeaderNode = CreateXmlElement('PackageIncludes', '', [], [])
        for Include in Package.GetPackageIncludeFileList():
            Tmp = PackageIncludeHeaderXml()
            PackageIncludeHeaderNode.appendChild(Tmp.ToXml(Include, 'PackageHeader'))
        DomPackage.appendChild(PackageIncludeHeaderNode)
        ModuleNode = CreateXmlElement('Modules', '', [], [])
        for Module in Package.GetModuleDict().values():
            Tmp = ModuleSurfaceAreaXml()
            ModuleNode.appendChild(Tmp.ToXml(Module))
        DomPackage.appendChild(ModuleNode)
        GuidProtocolPpiNode = CreateXmlElement('GuidDeclarations', '', [], [])
        for GuidProtocolPpi in Package.GetGuidList():
            Tmp = GuidXml('Package')
            GuidProtocolPpiNode.appendChild(Tmp.ToXml(GuidProtocolPpi, 'Entry'))
        DomPackage.appendChild(GuidProtocolPpiNode)
        GuidProtocolPpiNode = CreateXmlElement('ProtocolDeclarations', '', [], [])
        for GuidProtocolPpi in Package.GetProtocolList():
            Tmp = ProtocolXml('Package')
            GuidProtocolPpiNode.appendChild(Tmp.ToXml(GuidProtocolPpi, 'Entry'))
        DomPackage.appendChild(GuidProtocolPpiNode)
        GuidProtocolPpiNode = CreateXmlElement('PpiDeclarations', '', [], [])
        for GuidProtocolPpi in Package.GetPpiList():
            Tmp = PpiXml('Package')
            GuidProtocolPpiNode.appendChild(Tmp.ToXml(GuidProtocolPpi, 'Entry'))
        DomPackage.appendChild(GuidProtocolPpiNode)
        PcdEntryNode = CreateXmlElement('PcdDeclarations', '', [], [])
        for PcdEntry in Package.GetPcdList():
            Tmp = PcdEntryXml()
            PcdEntryNode.appendChild(Tmp.ToXml2(PcdEntry, 'PcdEntry'))
        DomPackage.appendChild(PcdEntryNode)
        Tmp = MiscellaneousFileXml()
        if Package.GetMiscFileList():
            DomPackage.appendChild(Tmp.ToXml(Package.GetMiscFileList()[0], 'MiscellaneousFiles'))
        if Package.GetUserExtensionList():
            for UserExtension in Package.GetUserExtensionList():
                Tmp = UserExtensionsXml()
                DomPackage.appendChild(Tmp.ToXml(UserExtension, 'UserExtensions'))
        return DomPackage