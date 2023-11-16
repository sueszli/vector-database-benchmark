class FDClassObject:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.FdUiName = ''
        self.CreateFileName = None
        self.BaseAddress = None
        self.BaseAddressPcd = None
        self.Size = None
        self.SizePcd = None
        self.ErasePolarity = None
        self.BlockSizeList = []
        self.DefineVarDict = {}
        self.SetVarDict = {}
        self.RegionList = []

class FfsClassObject:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.NameGuid = None
        self.Fixed = False
        self.CheckSum = False
        self.Alignment = None
        self.SectionList = []

class FileStatementClassObject(FfsClassObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        FfsClassObject.__init__(self)
        self.FvFileType = None
        self.FileName = None
        self.KeyStringList = []
        self.FvName = None
        self.FdName = None
        self.DefineVarDict = {}
        self.KeepReloc = None

class FfsInfStatementClassObject(FfsClassObject):

    def __init__(self):
        if False:
            print('Hello World!')
        FfsClassObject.__init__(self)
        self.Rule = None
        self.Version = None
        self.Ui = None
        self.InfFileName = None
        self.BuildNum = ''
        self.KeyStringList = []
        self.KeepReloc = None
        self.UseArch = None

class SectionClassObject:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.Alignment = None

class DepexSectionClassObject(SectionClassObject):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.DepexType = None
        self.Expression = None
        self.ExpressionProcessed = False

class CompressSectionClassObject(SectionClassObject):

    def __init__(self):
        if False:
            return 10
        SectionClassObject.__init__(self)
        self.CompType = None
        self.SectionList = []

class DataSectionClassObject(SectionClassObject):

    def __init__(self):
        if False:
            return 10
        SectionClassObject.__init__(self)
        self.SecType = None
        self.SectFileName = None
        self.SectionList = []
        self.KeepReloc = True

class EfiSectionClassObject(SectionClassObject):

    def __init__(self):
        if False:
            print('Hello World!')
        SectionClassObject.__init__(self)
        self.SectionType = None
        self.Optional = False
        self.FileType = None
        self.StringData = None
        self.FileName = None
        self.FileExtension = None
        self.BuildNum = None
        self.KeepReloc = None

class FvImageSectionClassObject(SectionClassObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        SectionClassObject.__init__(self)
        self.Fv = None
        self.FvName = None
        self.FvFileType = None
        self.FvFileName = None
        self.FvFileExtension = None
        self.FvAddr = None

class GuidSectionClassObject(SectionClassObject):

    def __init__(self):
        if False:
            return 10
        SectionClassObject.__init__(self)
        self.NameGuid = None
        self.SectionList = []
        self.SectionType = None
        self.ProcessRequired = False
        self.AuthStatusValid = False
        self.ExtraHeaderSize = -1
        self.FvAddr = []
        self.FvParentAddr = None
        self.IncludeFvSection = False

class SubTypeGuidSectionClassObject(SectionClassObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        SectionClassObject.__init__(self)
        self.SubTypeGuid = None

class UiSectionClassObject(SectionClassObject):

    def __init__(self):
        if False:
            return 10
        SectionClassObject.__init__(self)
        self.StringData = None
        self.FileName = None

class VerSectionClassObject(SectionClassObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        SectionClassObject.__init__(self)
        self.BuildNum = None
        self.StringData = None
        self.FileName = None

class RuleClassObject:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.Arch = None
        self.ModuleType = None
        self.TemplateName = None
        self.NameGuid = None
        self.Fixed = False
        self.Alignment = None
        self.SectAlignment = None
        self.CheckSum = False
        self.FvFileType = None
        self.KeyStringList = []
        self.KeepReloc = None

class RuleComplexFileClassObject(RuleClassObject):

    def __init__(self):
        if False:
            print('Hello World!')
        RuleClassObject.__init__(self)
        self.SectionList = []

class RuleSimpleFileClassObject(RuleClassObject):

    def __init__(self):
        if False:
            return 10
        RuleClassObject.__init__(self)
        self.FileName = None
        self.SectionType = ''
        self.FileExtension = None

class RuleFileExtensionClassObject(RuleClassObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        RuleClassObject.__init__(self)
        self.FileExtension = None

class CapsuleClassObject:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.SpecName = None
        self.UiCapsuleName = None
        self.CreateFile = None
        self.GroupIdNumber = None
        self.DefineVarDict = {}
        self.SetVarDict = {}
        self.TokensDict = {}
        self.CapsuleDataList = []
        self.FmpPayloadList = []

class OptionRomClassObject:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.DriverName = None
        self.FfsList = []