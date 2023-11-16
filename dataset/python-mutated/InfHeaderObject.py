"""
InfHeaderObject
"""

class InfHeaderObject:

    def __init__(self):
        if False:
            return 10
        self.FileName = ''
        self.Abstract = ''
        self.Description = ''
        self.Copyright = ''
        self.License = ''

    def SetFileName(self, FileName):
        if False:
            return 10
        if not (FileName == '' or FileName is None):
            self.FileName = FileName
            return True
        else:
            return False

    def GetFileName(self):
        if False:
            while True:
                i = 10
        return self.FileName

    def SetAbstract(self, Abstract):
        if False:
            i = 10
            return i + 15
        if not (Abstract == '' or Abstract is None):
            self.Abstract = Abstract
            return True
        else:
            return False

    def GetAbstract(self):
        if False:
            return 10
        return self.Abstract

    def SetDescription(self, Description):
        if False:
            while True:
                i = 10
        if not (Description == '' or Description is None):
            self.Description = Description
            return True
        else:
            return False

    def GetDescription(self):
        if False:
            return 10
        return self.Description

    def SetCopyright(self, Copyright):
        if False:
            print('Hello World!')
        if not (Copyright == '' or Copyright is None):
            self.Copyright = Copyright
            return True
        else:
            return False

    def GetCopyright(self):
        if False:
            while True:
                i = 10
        return self.Copyright

    def SetLicense(self, License):
        if False:
            print('Hello World!')
        if not (License == '' or License is None):
            self.License = License
            return True
        else:
            return False

    def GetLicense(self):
        if False:
            print('Hello World!')
        return self.License