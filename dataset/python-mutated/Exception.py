from __future__ import print_function
from __future__ import absolute_import
from Ecc.Xml.XmlRoutines import *
import Common.LongFilePathOs as os

class ExceptionXml(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.KeyWord = ''
        self.ErrorID = ''
        self.FilePath = ''

    def FromXml(self, Item, Key):
        if False:
            for i in range(10):
                print('nop')
        self.KeyWord = XmlElement(Item, '%s/KeyWord' % Key)
        self.ErrorID = XmlElement(Item, '%s/ErrorID' % Key)
        self.FilePath = os.path.normpath(XmlElement(Item, '%s/FilePath' % Key))

    def __str__(self):
        if False:
            return 10
        return 'ErrorID = %s KeyWord = %s FilePath = %s' % (self.ErrorID, self.KeyWord, self.FilePath)

class ExceptionListXml(object):

    def __init__(self):
        if False:
            return 10
        self.List = []

    def FromXmlFile(self, FilePath):
        if False:
            return 10
        XmlContent = XmlParseFile(FilePath)
        for Item in XmlList(XmlContent, '/ExceptionList/Exception'):
            Exp = ExceptionXml()
            Exp.FromXml(Item, 'Exception')
            self.List.append(Exp)

    def ToList(self):
        if False:
            i = 10
            return i + 15
        RtnList = []
        for Item in self.List:
            RtnList.append((Item.ErrorID, Item.KeyWord))
        return RtnList

    def __str__(self):
        if False:
            print('Hello World!')
        RtnStr = ''
        if self.List:
            for Item in self.List:
                RtnStr = RtnStr + str(Item) + '\n'
        return RtnStr

class ExceptionCheck(object):

    def __init__(self, FilePath=None):
        if False:
            for i in range(10):
                print('nop')
        self.ExceptionList = []
        self.ExceptionListXml = ExceptionListXml()
        self.LoadExceptionListXml(FilePath)

    def LoadExceptionListXml(self, FilePath):
        if False:
            i = 10
            return i + 15
        if FilePath and os.path.isfile(FilePath):
            self.ExceptionListXml.FromXmlFile(FilePath)
            self.ExceptionList = self.ExceptionListXml.ToList()

    def IsException(self, ErrorID, KeyWord, FileID=-1):
        if False:
            print('Hello World!')
        if (str(ErrorID), KeyWord.replace('\r\n', '\n')) in self.ExceptionList:
            return True
        else:
            return False
if __name__ == '__main__':
    El = ExceptionCheck('C:\\Hess\\Project\\BuildTool\\src\\Ecc\\exception.xml')
    print(El.ExceptionList)