from plugins.EdkPlugins.basemodel import ini
import re, os
from plugins.EdkPlugins.basemodel.message import *

class DECFile(ini.BaseINIFile):

    def GetSectionInstance(self, parent, name, isCombined=False):
        if False:
            i = 10
            return i + 15
        return DECSection(parent, name, isCombined)

    def GetComponents(self):
        if False:
            return 10
        return self.GetSectionByName('Components')

    def GetPackageRootPath(self):
        if False:
            i = 10
            return i + 15
        return os.path.dirname(self.GetFilename()).strip()

    def GetBaseName(self):
        if False:
            print('Hello World!')
        return self.GetDefine('PACKAGE_NAME').strip()

    def GetVersion(self):
        if False:
            for i in range(10):
                print('nop')
        return self.GetDefine('PACKAGE_VERSION').strip()

    def GetSectionObjectsByName(self, name, arch=None):
        if False:
            for i in range(10):
                print('nop')
        arr = []
        sects = self.GetSectionByName(name)
        for sect in sects:
            if not sect.IsArchMatch(arch):
                continue
            for obj in sect.GetObjects():
                arr.append(obj)
        return arr

class DECSection(ini.BaseINISection):

    def GetSectionINIObject(self, parent):
        if False:
            return 10
        type = self.GetType()
        if type.lower().find('defines') != -1:
            return DECDefineSectionObject(self)
        if type.lower().find('includes') != -1:
            return DECIncludeObject(self)
        if type.lower().find('pcd') != -1:
            return DECPcdObject(self)
        if type.lower() == 'libraryclasses':
            return DECLibraryClassObject(self)
        if type.lower() == 'guids':
            return DECGuidObject(self)
        if type.lower() == 'ppis':
            return DECPpiObject(self)
        if type.lower() == 'protocols':
            return DECProtocolObject(self)
        return DECSectionObject(self)

    def GetType(self):
        if False:
            print('Hello World!')
        arr = self._name.split('.')
        return arr[0].strip()

    def GetArch(self):
        if False:
            for i in range(10):
                print('nop')
        arr = self._name.split('.')
        if len(arr) == 1:
            return 'common'
        return arr[1]

    def IsArchMatch(self, arch):
        if False:
            return 10
        if arch is None or self.GetArch() == 'common':
            return True
        if self.GetArch().lower() != arch.lower():
            return False
        return True

class DECSectionObject(ini.BaseINISectionObject):

    def GetArch(self):
        if False:
            return 10
        return self.GetParent().GetArch()

class DECDefineSectionObject(DECSectionObject):

    def __init__(self, parent):
        if False:
            print('Hello World!')
        DECSectionObject.__init__(self, parent)
        self._key = None
        self._value = None

    def Parse(self):
        if False:
            for i in range(10):
                print('nop')
        assert self._start == self._end, 'The object in define section must be in single line'
        line = self.GetLineByOffset(self._start).strip()
        line = line.split('#')[0]
        arr = line.split('=')
        if len(arr) != 2:
            ErrorMsg('Invalid define section object', self.GetFilename(), self.GetParent().GetName())
            return False
        self._key = arr[0].strip()
        self._value = arr[1].strip()
        return True

    def GetKey(self):
        if False:
            for i in range(10):
                print('nop')
        return self._key

    def GetValue(self):
        if False:
            print('Hello World!')
        return self._value

class DECGuidObject(DECSectionObject):
    _objs = {}

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        DECSectionObject.__init__(self, parent)
        self._name = None

    def Parse(self):
        if False:
            while True:
                i = 10
        line = self.GetLineByOffset(self._start).strip().split('#')[0]
        self._name = line.split('=')[0].strip()
        self._guid = line.split('=')[1].strip()
        objdict = DECGuidObject._objs
        if self._name not in objdict.keys():
            objdict[self._name] = [self]
        else:
            objdict[self._name].append(self)
        return True

    def GetName(self):
        if False:
            while True:
                i = 10
        return self._name

    def GetGuid(self):
        if False:
            while True:
                i = 10
        return self._guid

    def Destroy(self):
        if False:
            i = 10
            return i + 15
        objdict = DECGuidObject._objs
        objdict[self._name].remove(self)
        if len(objdict[self._name]) == 0:
            del objdict[self._name]

    @staticmethod
    def GetObjectDict():
        if False:
            print('Hello World!')
        return DECGuidObject._objs

class DECPpiObject(DECSectionObject):
    _objs = {}

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        DECSectionObject.__init__(self, parent)
        self._name = None

    def Parse(self):
        if False:
            while True:
                i = 10
        line = self.GetLineByOffset(self._start).strip().split('#')[0]
        self._name = line.split('=')[0].strip()
        self._guid = line.split('=')[1].strip()
        objdict = DECPpiObject._objs
        if self._name not in objdict.keys():
            objdict[self._name] = [self]
        else:
            objdict[self._name].append(self)
        return True

    def GetName(self):
        if False:
            while True:
                i = 10
        return self._name

    def GetGuid(self):
        if False:
            for i in range(10):
                print('nop')
        return self._guid

    def Destroy(self):
        if False:
            print('Hello World!')
        objdict = DECPpiObject._objs
        objdict[self._name].remove(self)
        if len(objdict[self._name]) == 0:
            del objdict[self._name]

    @staticmethod
    def GetObjectDict():
        if False:
            return 10
        return DECPpiObject._objs

class DECProtocolObject(DECSectionObject):
    _objs = {}

    def __init__(self, parent):
        if False:
            return 10
        DECSectionObject.__init__(self, parent)
        self._name = None

    def Parse(self):
        if False:
            print('Hello World!')
        line = self.GetLineByOffset(self._start).strip().split('#')[0]
        self._name = line.split('=')[0].strip()
        self._guid = line.split('=')[1].strip()
        objdict = DECProtocolObject._objs
        if self._name not in objdict.keys():
            objdict[self._name] = [self]
        else:
            objdict[self._name].append(self)
        return True

    def GetName(self):
        if False:
            while True:
                i = 10
        return self._name

    def GetGuid(self):
        if False:
            i = 10
            return i + 15
        return self._guid

    def Destroy(self):
        if False:
            while True:
                i = 10
        objdict = DECProtocolObject._objs
        objdict[self._name].remove(self)
        if len(objdict[self._name]) == 0:
            del objdict[self._name]

    @staticmethod
    def GetObjectDict():
        if False:
            for i in range(10):
                print('nop')
        return DECProtocolObject._objs

class DECLibraryClassObject(DECSectionObject):
    _objs = {}

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        DECSectionObject.__init__(self, parent)
        self.mClassName = None
        self.mHeaderFile = None

    def Parse(self):
        if False:
            i = 10
            return i + 15
        line = self.GetLineByOffset(self._start).strip().split('#')[0]
        (self.mClassName, self.mHeaderFile) = line.split('|')
        objdict = DECLibraryClassObject._objs
        if self.mClassName not in objdict.keys():
            objdict[self.mClassName] = [self]
        else:
            objdict[self.mClassName].append(self)
        return True

    def GetClassName(self):
        if False:
            i = 10
            return i + 15
        return self.mClassName

    def GetName(self):
        if False:
            while True:
                i = 10
        return self.mClassName

    def GetHeaderFile(self):
        if False:
            return 10
        return self.mHeaderFile

    def Destroy(self):
        if False:
            while True:
                i = 10
        objdict = DECLibraryClassObject._objs
        objdict[self.mClassName].remove(self)
        if len(objdict[self.mClassName]) == 0:
            del objdict[self.mClassName]

    @staticmethod
    def GetObjectDict():
        if False:
            return 10
        return DECLibraryClassObject._objs

class DECIncludeObject(DECSectionObject):

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        DECSectionObject.__init__(self, parent)

    def GetPath(self):
        if False:
            i = 10
            return i + 15
        return self.GetLineByOffset(self._start).split('#')[0].strip()

class DECPcdObject(DECSectionObject):
    _objs = {}

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        DECSectionObject.__init__(self, parent)
        self.mPcdName = None
        self.mPcdDefaultValue = None
        self.mPcdDataType = None
        self.mPcdToken = None

    def Parse(self):
        if False:
            for i in range(10):
                print('nop')
        line = self.GetLineByOffset(self._start).strip().split('#')[0]
        (self.mPcdName, self.mPcdDefaultValue, self.mPcdDataType, self.mPcdToken) = line.split('|')
        objdict = DECPcdObject._objs
        if self.mPcdName not in objdict.keys():
            objdict[self.mPcdName] = [self]
        else:
            objdict[self.mPcdName].append(self)
        return True

    def Destroy(self):
        if False:
            i = 10
            return i + 15
        objdict = DECPcdObject._objs
        objdict[self.mPcdName].remove(self)
        if len(objdict[self.mPcdName]) == 0:
            del objdict[self.mPcdName]

    def GetPcdType(self):
        if False:
            print('Hello World!')
        return self.GetParent().GetType()

    def GetPcdName(self):
        if False:
            return 10
        return self.mPcdName

    def GetPcdValue(self):
        if False:
            i = 10
            return i + 15
        return self.mPcdDefaultValue

    def GetPcdDataType(self):
        if False:
            return 10
        return self.mPcdDataType

    def GetPcdToken(self):
        if False:
            return 10
        return self.mPcdToken

    def GetName(self):
        if False:
            i = 10
            return i + 15
        return self.GetPcdName().split('.')[1]

    @staticmethod
    def GetObjectDict():
        if False:
            return 10
        return DECPcdObject._objs