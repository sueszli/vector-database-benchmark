"""
InfCommonObject
"""

class InfLineCommentObject:

    def __init__(self):
        if False:
            print('Hello World!')
        self.HeaderComments = ''
        self.TailComments = ''

    def SetHeaderComments(self, HeaderComments):
        if False:
            return 10
        self.HeaderComments = HeaderComments

    def GetHeaderComments(self):
        if False:
            return 10
        return self.HeaderComments

    def SetTailComments(self, TailComments):
        if False:
            print('Hello World!')
        self.TailComments = TailComments

    def GetTailComments(self):
        if False:
            while True:
                i = 10
        return self.TailComments

class CurrentLine:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.LineNo = ''
        self.LineString = ''
        self.FileName = ''

    def SetLineNo(self, LineNo):
        if False:
            while True:
                i = 10
        self.LineNo = LineNo

    def GetLineNo(self):
        if False:
            return 10
        return self.LineNo

    def SetLineString(self, LineString):
        if False:
            for i in range(10):
                print('nop')
        self.LineString = LineString

    def GetLineString(self):
        if False:
            return 10
        return self.LineString

    def SetFileName(self, FileName):
        if False:
            while True:
                i = 10
        self.FileName = FileName

    def GetFileName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.FileName

class InfSectionCommonDef:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.HeaderComments = ''
        self.TailComments = ''
        self.SupArchList = []
        self.AllContent = {}

    def SetHeaderComments(self, HeaderComments):
        if False:
            for i in range(10):
                print('nop')
        self.HeaderComments = HeaderComments

    def GetHeaderComments(self):
        if False:
            i = 10
            return i + 15
        return self.HeaderComments

    def SetTailComments(self, TailComments):
        if False:
            while True:
                i = 10
        self.TailComments = TailComments

    def GetTailComments(self):
        if False:
            for i in range(10):
                print('nop')
        return self.TailComments

    def SetSupArchList(self, Arch):
        if False:
            i = 10
            return i + 15
        if Arch not in self.SupArchList:
            self.SupArchList.append(Arch)

    def GetSupArchList(self):
        if False:
            for i in range(10):
                print('nop')
        return self.SupArchList

    def SetAllContent(self, Content):
        if False:
            return 10
        self.AllContent = Content

    def GetAllContent(self):
        if False:
            print('Hello World!')
        return self.AllContent