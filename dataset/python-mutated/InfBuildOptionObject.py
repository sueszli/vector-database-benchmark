"""
InfBuildOptionObject
"""
from Library import GlobalData
from Object.Parser.InfCommonObject import InfSectionCommonDef

class InfBuildOptionItem:

    def __init__(self):
        if False:
            print('Hello World!')
        self.Content = ''
        self.SupArchList = []
        self.AsBuildList = []

    def SetContent(self, Content):
        if False:
            while True:
                i = 10
        self.Content = Content

    def GetContent(self):
        if False:
            i = 10
            return i + 15
        return self.Content

    def SetSupArchList(self, SupArchList):
        if False:
            print('Hello World!')
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            for i in range(10):
                print('nop')
        return self.SupArchList

    def SetAsBuildList(self, AsBuildList):
        if False:
            print('Hello World!')
        self.AsBuildList = AsBuildList

    def GetAsBuildList(self):
        if False:
            for i in range(10):
                print('nop')
        return self.AsBuildList

class InfBuildOptionsObject(InfSectionCommonDef):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.BuildOptions = []
        InfSectionCommonDef.__init__(self)

    def SetBuildOptions(self, BuildOptCont, ArchList=None, SectionContent=''):
        if False:
            return 10
        if not GlobalData.gIS_BINARY_INF:
            if SectionContent.strip() != '':
                InfBuildOptionItemObj = InfBuildOptionItem()
                InfBuildOptionItemObj.SetContent(SectionContent)
                InfBuildOptionItemObj.SetSupArchList(ArchList)
                self.BuildOptions.append(InfBuildOptionItemObj)
        elif len(BuildOptCont) >= 1:
            InfBuildOptionItemObj = InfBuildOptionItem()
            InfBuildOptionItemObj.SetAsBuildList(BuildOptCont)
            InfBuildOptionItemObj.SetSupArchList(ArchList)
            self.BuildOptions.append(InfBuildOptionItemObj)
        return True

    def GetBuildOptions(self):
        if False:
            while True:
                i = 10
        return self.BuildOptions