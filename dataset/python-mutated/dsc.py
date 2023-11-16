from plugins.EdkPlugins.basemodel import ini
import re, os
from plugins.EdkPlugins.basemodel.message import *

class DSCFile(ini.BaseINIFile):

    def GetSectionInstance(self, parent, name, isCombined=False):
        if False:
            while True:
                i = 10
        return DSCSection(parent, name, isCombined)

    def GetComponents(self):
        if False:
            return 10
        return self.GetSectionObjectsByName('Components')

class DSCSection(ini.BaseINISection):

    def GetSectionINIObject(self, parent):
        if False:
            while True:
                i = 10
        type = self.GetType()
        if type.lower() == 'components':
            return DSCComponentObject(self)
        if type.lower() == 'libraryclasses':
            return DSCLibraryClassObject(self)
        if type.lower() == 'defines':
            return ini.BaseINISectionObject(self)
        if type.lower() == 'pcdsfeatureflag' or type.lower() == 'pcdsfixedatbuild' or type.lower() == 'pcdspatchableinmodule' or (type.lower() == 'pcdsdynamicdefault') or (type.lower() == 'pcdsdynamicex') or (type.lower() == 'pcdsdynamichii') or (type.lower() == 'pcdsdynamicvpd'):
            return DSCPcdObject(self)
        return DSCSectionObject(self)

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

    def GetModuleType(self):
        if False:
            i = 10
            return i + 15
        arr = self._name.split('.')
        if len(arr) < 3:
            return 'common'
        return arr[2]

class DSCSectionObject(ini.BaseINISectionObject):

    def GetArch(self):
        if False:
            while True:
                i = 10
        return self.GetParent().GetArch()

class DSCPcdObject(DSCSectionObject):

    def __init__(self, parent):
        if False:
            return 10
        ini.BaseINISectionObject.__init__(self, parent)
        self._name = None

    def Parse(self):
        if False:
            for i in range(10):
                print('nop')
        line = self.GetLineByOffset(self._start).strip().split('#')[0]
        self._name = line.split('|')[0]
        self._value = line.split('|')[1]
        return True

    def GetPcdName(self):
        if False:
            print('Hello World!')
        return self._name

    def GetPcdType(self):
        if False:
            while True:
                i = 10
        return self.GetParent().GetType()

    def GetPcdValue(self):
        if False:
            for i in range(10):
                print('nop')
        return self._value

class DSCLibraryClassObject(DSCSectionObject):

    def __init__(self, parent):
        if False:
            return 10
        ini.BaseINISectionObject.__init__(self, parent)

    def GetClass(self):
        if False:
            while True:
                i = 10
        line = self.GetLineByOffset(self._start)
        return line.split('#')[0].split('|')[0].strip()

    def GetInstance(self):
        if False:
            for i in range(10):
                print('nop')
        line = self.GetLineByOffset(self._start)
        return line.split('#')[0].split('|')[1].strip()

    def GetArch(self):
        if False:
            for i in range(10):
                print('nop')
        return self.GetParent().GetArch()

    def GetModuleType(self):
        if False:
            while True:
                i = 10
        return self.GetParent().GetModuleType()

class DSCComponentObject(DSCSectionObject):

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        ini.BaseINISectionObject.__init__(self, parent)
        self._OveridePcds = {}
        self._OverideLibraries = {}
        self._Filename = ''

    def __del__(self):
        if False:
            i = 10
            return i + 15
        self._OverideLibraries.clear()
        self._OverideLibraries.clear()
        ini.BaseINISectionObject.__del__(self)

    def AddOverideLib(self, libclass, libinstPath):
        if False:
            for i in range(10):
                print('nop')
        if libclass not in self._OverideLibraries.keys():
            self._OverideLibraries[libclass] = libinstPath

    def AddOveridePcd(self, name, type, value=None):
        if False:
            i = 10
            return i + 15
        if type not in self._OveridePcds.keys():
            self._OveridePcds[type] = []
        self._OveridePcds[type].append((name, value))

    def GetOverideLibs(self):
        if False:
            i = 10
            return i + 15
        return self._OverideLibraries

    def GetArch(self):
        if False:
            print('Hello World!')
        return self.GetParent().GetArch()

    def GetOveridePcds(self):
        if False:
            while True:
                i = 10
        return self._OveridePcds

    def GetFilename(self):
        if False:
            i = 10
            return i + 15
        return self.GetLineByOffset(self._start).split('#')[0].split('{')[0].strip()

    def SetFilename(self, fName):
        if False:
            i = 10
            return i + 15
        self._Filename = fName

    def Parse(self):
        if False:
            i = 10
            return i + 15
        if self._start < self._end:
            curr = self._start + 1
            end = self._end - 1
            OverideName = ''
            while curr <= end:
                line = self.GetLineByOffset(curr).strip()
                if len(line) > 0 and line[0] != '#':
                    line = line.split('#')[0].strip()
                    if line[0] == '<':
                        OverideName = line[1:len(line) - 1]
                    elif OverideName.lower() == 'libraryclasses':
                        arr = line.split('|')
                        self._OverideLibraries[arr[0].strip()] = arr[1].strip()
                    elif OverideName.lower() == 'pcds':
                        ErrorMsg('EDES does not support PCD overide', self.GetFileName(), self.GetParent().GetLinenumberByOffset(curr))
                curr = curr + 1
        return True

    def GenerateLines(self):
        if False:
            i = 10
            return i + 15
        lines = []
        hasLib = False
        hasPcd = False
        if len(self._OverideLibraries) != 0:
            hasLib = True
        if len(self._OveridePcds) != 0:
            hasPcd = True
        if hasLib or hasPcd:
            lines.append('  %s {\n' % self._Filename)
        else:
            lines.append('  %s \n' % self._Filename)
            return lines
        if hasLib:
            lines.append('    <LibraryClasses>\n')
            for libKey in self._OverideLibraries.keys():
                lines.append('      %s|%s\n' % (libKey, self._OverideLibraries[libKey]))
        if hasPcd:
            for key in self._OveridePcds.keys():
                lines.append('    <%s>\n' % key)
                for (name, value) in self._OveridePcds[key]:
                    if value is not None:
                        lines.append('      %s|%s\n' % (name, value))
                    else:
                        lines.append('      %s\n' % name)
        if hasLib or hasPcd:
            lines.append('  }\n')
        return lines