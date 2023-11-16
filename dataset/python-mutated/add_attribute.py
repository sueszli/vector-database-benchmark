from __future__ import annotations
import os.path
import sys
(className, attributeName, attributeType) = sys.argv[1:4]
if len(sys.argv) > 4:
    attributeClassType = sys.argv[4]
else:
    attributeClassType = ''
types = {'string': ('str', None, 'self._makeStringAttribute(attributes["' + attributeName + '"])', 'str'), 'int': ('int', None, 'self._makeIntAttribute(attributes["' + attributeName + '"])', 'int'), 'bool': ('bool', None, 'self._makeBoolAttribute(attributes["' + attributeName + '"])', 'bool'), 'datetime': ('datetime', 'str', 'self._makeDatetimeAttribute(attributes["' + attributeName + '"])', 'datetime'), 'class': (':class:`' + attributeClassType + '`', None, 'self._makeClassAttribute(' + attributeClassType + ', attributes["' + attributeName + '"])', attributeClassType)}
(attributeDocType, attributeAssertType, attributeValue, attributeClassType) = types[attributeType]
if attributeType == 'class':
    attributeClassType = f"'{attributeClassType}'"
fileName = os.path.join('github', className + '.py')

def add_as_class_property(lines: list[str]) -> list[str]:
    if False:
        print('Hello World!')
    newLines = []
    i = 0
    added = False
    isCompletable = True
    isProperty = False
    while not added:
        line = lines[i].rstrip()
        i += 1
        if line.startswith('class '):
            if 'NonCompletableGithubObject' in line:
                isCompletable = False
        elif line == '    @property':
            isProperty = True
        elif line.startswith('    def '):
            attrName = line[8:-7]
            if (not attrName.startswith('__repr__') and (not attrName.startswith('_initAttributes'))) and (attrName == '_identity' or attrName > attributeName or (not isProperty)):
                if not isProperty:
                    newLines.append('    @property')
                newLines.append('    def ' + attributeName + '(self) -> ' + attributeClassType + ':')
                if isCompletable:
                    newLines.append('        self._completeIfNotSet(self._' + attributeName + ')')
                newLines.append('        return self._' + attributeName + '.value')
                newLines.append('')
                if isProperty:
                    newLines.append('    @property')
                added = True
            isProperty = False
        newLines.append(line)
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        newLines.append(line)
    return newLines

def add_to_initAttributes(lines: list[str]) -> list[str]:
    if False:
        return 10
    newLines = []
    added = False
    i = 0
    inInit = False
    while not added:
        line = lines[i].rstrip()
        i += 1
        if line.strip().startswith('def _initAttributes(self)'):
            inInit = True
        if inInit:
            if not line or line.endswith(' = github.GithubObject.NotSet') or line.endswith(' = NotSet'):
                if line:
                    attrName = line[14:-29]
                if not line or attrName > attributeName:
                    newLines.append(f'        self._{attributeName}: Attribute[{attributeClassType}] = NotSet')
                    added = True
        newLines.append(line)
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        newLines.append(line)
    return newLines

def add_to_useAttributes(lines: list[str]) -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    i = 0
    newLines = []
    added = False
    inUse = False
    while not added:
        try:
            line = lines[i].rstrip()
        except IndexError:
            line = ''
        i += 1
        if line.strip().startswith('def _useAttributes(self, attributes:'):
            inUse = True
        if inUse:
            if not line or line.endswith(' in attributes:  # pragma no branch'):
                if line:
                    attrName = line[12:-36]
                if not line or attrName > attributeName:
                    newLines.append('        if "' + attributeName + '" in attributes:  # pragma no branch')
                    if attributeAssertType:
                        newLines.append('            assert attributes["' + attributeName + '"] is None or isinstance(attributes["' + attributeName + '"], ' + attributeAssertType + '), attributes["' + attributeName + '"]')
                    newLines.append(f'            self._{attributeName} = {attributeValue}')
                    added = True
        newLines.append(line)
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        newLines.append(line)
    return newLines
with open(fileName) as f:
    source = f.readlines()
source = add_as_class_property(source)
source = add_to_initAttributes(source)
source = add_to_useAttributes(source)
with open(fileName, 'w', newline='\n') as f:
    f.write('\n'.join(source) + '\n')