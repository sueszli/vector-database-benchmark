import os
import re
import sys

def readDataFromFile(binfile, offset, len=1):
    if False:
        while True:
            i = 10
    fd = open(binfile, 'r+b')
    fsize = os.path.getsize(binfile)
    offval = offset & 4294967295
    if offval & 2147483648:
        offval = fsize - (4294967295 - offval + 1)
    fd.seek(offval)
    if sys.version_info[0] < 3:
        bytearray = [ord(b) for b in fd.read(len)]
    else:
        bytearray = [b for b in fd.read(len)]
    value = 0
    idx = len - 1
    while idx >= 0:
        value = value << 8 | bytearray[idx]
        idx = idx - 1
    fd.close()
    return value

def IsFspHeaderValid(binfile):
    if False:
        for i in range(10):
            print('nop')
    fd = open(binfile, 'rb')
    bindat = fd.read(512)
    fd.close()
    HeaderList = [b'FSPH', b'FSPP', b'FSPE']
    OffsetList = []
    for each in HeaderList:
        if each in bindat:
            idx = bindat.index(each)
        else:
            idx = 0
        OffsetList.append(idx)
    if not OffsetList[0] or not OffsetList[1]:
        return False
    if sys.version_info[0] < 3:
        Revision = ord(bindat[OffsetList[0] + 11])
    else:
        Revision = bindat[OffsetList[0] + 11]
    if Revision > 1 and (not OffsetList[2]):
        return False
    return True

def patchDataInFile(binfile, offset, value, len=1):
    if False:
        return 10
    fd = open(binfile, 'r+b')
    fsize = os.path.getsize(binfile)
    offval = offset & 4294967295
    if offval & 2147483648:
        offval = fsize - (4294967295 - offval + 1)
    bytearray = []
    idx = 0
    while idx < len:
        bytearray.append(value & 255)
        value = value >> 8
        idx = idx + 1
    fd.seek(offval)
    if sys.version_info[0] < 3:
        fd.write(''.join((chr(b) for b in bytearray)))
    else:
        fd.write(bytes(bytearray))
    fd.close()
    return len

class Symbols:

    def __init__(self):
        if False:
            return 10
        self.dictSymbolAddress = {}
        self.dictGuidNameXref = {}
        self.dictFfsOffset = {}
        self.dictVariable = {}
        self.dictModBase = {}
        self.fdFile = None
        self.string = ''
        self.fdBase = 4294967295
        self.fdSize = 0
        self.index = 0
        self.fvList = []
        self.parenthesisOpenSet = '([{<'
        self.parenthesisCloseSet = ')]}>'

    def getFdFile(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fdFile

    def getFdSize(self):
        if False:
            return 10
        return self.fdSize

    def parseFvInfFile(self, infFile):
        if False:
            print('Hello World!')
        fvInfo = {}
        fvFile = infFile[0:-4] + '.Fv'
        fvInfo['Name'] = os.path.splitext(os.path.basename(infFile))[0]
        fvInfo['Offset'] = self.getFvOffsetInFd(fvFile)
        fvInfo['Size'] = readDataFromFile(fvFile, 32, 4)
        fdIn = open(infFile, 'r')
        rptLines = fdIn.readlines()
        fdIn.close()
        fvInfo['Base'] = 0
        for rptLine in rptLines:
            match = re.match('^EFI_BASE_ADDRESS\\s*=\\s*(0x[a-fA-F0-9]+)', rptLine)
            if match:
                fvInfo['Base'] = int(match.group(1), 16)
                break
        self.fvList.append(dict(fvInfo))
        return 0

    def createDicts(self, fvDir, fvNames):
        if False:
            i = 10
            return i + 15
        if not os.path.isdir(fvDir):
            raise Exception("'%s' is not a valid directory!" % fvDir)
        fdFile = os.path.join(fvDir, fvNames + '.fd')
        if os.path.exists(fdFile):
            print("Tool identified Fd file as a input to patch '%s'" % fdFile)
            self.fdFile = fdFile
            self.fdSize = os.path.getsize(fdFile)
            return 0
        xrefFile = os.path.join(fvDir, 'Guid.xref')
        if not os.path.exists(xrefFile):
            raise Exception("Cannot open GUID Xref file '%s'!" % xrefFile)
        self.dictGuidNameXref = {}
        self.parseGuidXrefFile(xrefFile)
        fvList = fvNames.split(':')
        fdBase = fvList.pop()
        if len(fvList) == 0:
            fvList.append(fdBase)
        fdFile = os.path.join(fvDir, fdBase.strip() + '.fd')
        if not os.path.exists(fdFile):
            raise Exception("Cannot open FD file '%s'!" % fdFile)
        self.fdFile = fdFile
        self.fdSize = os.path.getsize(fdFile)
        infFile = os.path.join(fvDir, fvList[0].strip()) + '.inf'
        if not os.path.exists(infFile):
            raise Exception("Cannot open INF file '%s'!" % infFile)
        self.parseInfFile(infFile)
        self.dictVariable = {}
        self.dictVariable['FDSIZE'] = self.fdSize
        self.dictVariable['FDBASE'] = self.fdBase
        self.fvList = []
        self.dictSymbolAddress = {}
        self.dictFfsOffset = {}
        for file in fvList:
            fvFile = os.path.join(fvDir, file.strip()) + '.Fv'
            mapFile = fvFile + '.map'
            if not os.path.exists(mapFile):
                raise Exception("Cannot open MAP file '%s'!" % mapFile)
            infFile = fvFile[0:-3] + '.inf'
            self.parseFvInfFile(infFile)
            self.parseFvMapFile(mapFile)
            fvTxtFile = fvFile + '.txt'
            if not os.path.exists(fvTxtFile):
                raise Exception("Cannot open FV TXT file '%s'!" % fvTxtFile)
            self.parseFvTxtFile(fvTxtFile)
        for fv in self.fvList:
            self.dictVariable['_BASE_%s_' % fv['Name']] = fv['Base']
        ffsDir = os.path.join(fvDir, 'Ffs')
        if os.path.isdir(ffsDir):
            for item in os.listdir(ffsDir):
                if len(item) <= 36:
                    continue
                mapFile = os.path.join(ffsDir, item, '%s.map' % item[0:36])
                if not os.path.exists(mapFile):
                    continue
                self.parseModMapFile(item[36:], mapFile)
        return 0

    def getFvOffsetInFd(self, fvFile):
        if False:
            for i in range(10):
                print('nop')
        fvHandle = open(fvFile, 'r+b')
        fdHandle = open(self.fdFile, 'r+b')
        offset = fdHandle.read().find(fvHandle.read(112))
        fvHandle.close()
        fdHandle.close()
        if offset == -1:
            raise Exception('Could not locate FV file %s in FD!' % fvFile)
        return offset

    def parseInfFile(self, infFile):
        if False:
            i = 10
            return i + 15
        fvOffset = self.getFvOffsetInFd(infFile[0:-4] + '.Fv')
        fdIn = open(infFile, 'r')
        rptLine = fdIn.readline()
        self.fdBase = 4294967295
        while rptLine != '':
            match = re.match('^EFI_BASE_ADDRESS\\s*=\\s*(0x[a-fA-F0-9]+)', rptLine)
            if match is not None:
                self.fdBase = int(match.group(1), 16) - fvOffset
                break
            rptLine = fdIn.readline()
        fdIn.close()
        if self.fdBase == 4294967295:
            raise Exception('Could not find EFI_BASE_ADDRESS in INF file!' % infFile)
        return 0

    def parseFvTxtFile(self, fvTxtFile):
        if False:
            i = 10
            return i + 15
        fvName = os.path.basename(fvTxtFile)[0:-7].upper()
        fvOffset = self.getFvOffsetInFd(fvTxtFile[0:-4])
        fdIn = open(fvTxtFile, 'r')
        rptLine = fdIn.readline()
        while rptLine != '':
            match = re.match('(0x[a-fA-F0-9]+)\\s([0-9a-fA-F\\-]+)', rptLine)
            if match is not None:
                if match.group(2) in self.dictFfsOffset:
                    self.dictFfsOffset[fvName + ':' + match.group(2)] = '0x%08X' % (int(match.group(1), 16) + fvOffset)
                else:
                    self.dictFfsOffset[match.group(2)] = '0x%08X' % (int(match.group(1), 16) + fvOffset)
            rptLine = fdIn.readline()
        fdIn.close()
        return 0

    def parseFvMapFile(self, mapFile):
        if False:
            return 10
        fdIn = open(mapFile, 'r')
        rptLine = fdIn.readline()
        modName = ''
        foundModHdr = False
        while rptLine != '':
            if rptLine[0] != ' ':
                match = re.match('([_a-zA-Z0-9\\-]+)\\s\\(.+BaseAddress=(0x[0-9a-fA-F]+),\\s+EntryPoint=(0x[0-9a-fA-F]+),\\s*Type=\\w+\\)', rptLine)
                if match is None:
                    match = re.match('([_a-zA-Z0-9\\-]+)\\s\\(.+BaseAddress=(0x[0-9a-fA-F]+),\\s+EntryPoint=(0x[0-9a-fA-F]+)\\)', rptLine)
                if match is not None:
                    foundModHdr = True
                    modName = match.group(1)
                    if len(modName) == 36:
                        modName = self.dictGuidNameXref[modName.upper()]
                    self.dictModBase['%s:BASE' % modName] = int(match.group(2), 16)
                    self.dictModBase['%s:ENTRY' % modName] = int(match.group(3), 16)
                match = re.match('\\(GUID=([A-Z0-9\\-]+)\\s+\\.textbaseaddress=(0x[0-9a-fA-F]+)\\s+\\.databaseaddress=(0x[0-9a-fA-F]+)\\)', rptLine)
                if match is not None:
                    if foundModHdr:
                        foundModHdr = False
                    else:
                        modName = match.group(1)
                        if len(modName) == 36:
                            modName = self.dictGuidNameXref[modName.upper()]
                    self.dictModBase['%s:TEXT' % modName] = int(match.group(2), 16)
                    self.dictModBase['%s:DATA' % modName] = int(match.group(3), 16)
            else:
                foundModHdr = False
                match = re.match('^\\s+(0x[a-z0-9]+)\\s+([_a-zA-Z0-9]+)', rptLine)
                if match is not None:
                    self.dictSymbolAddress['%s:%s' % (modName, match.group(2))] = match.group(1)
            rptLine = fdIn.readline()
        fdIn.close()
        return 0

    def parseModMapFile(self, moduleName, mapFile):
        if False:
            i = 10
            return i + 15
        modSymbols = {}
        fdIn = open(mapFile, 'r')
        reportLines = fdIn.readlines()
        fdIn.close()
        moduleEntryPoint = '__ModuleEntryPoint'
        reportLine = reportLines[0]
        if reportLine.strip().find('Archive member included') != -1:
            patchMapFileMatchString = '\\s+(0x[0-9a-fA-F]{16})\\s+([^\\s][^0x][_a-zA-Z0-9\\-]+)\\s'
            matchKeyGroupIndex = 2
            matchSymbolGroupIndex = 1
            prefix = '_'
        else:
            patchMapFileMatchString = '^\\s[0-9a-fA-F]{4}:[0-9a-fA-F]{8}\\s+(\\w+)\\s+([0-9a-fA-F]{8,16}\\s+)'
            matchKeyGroupIndex = 1
            matchSymbolGroupIndex = 2
            prefix = ''
        for reportLine in reportLines:
            match = re.match(patchMapFileMatchString, reportLine)
            if match is not None:
                modSymbols[prefix + match.group(matchKeyGroupIndex)] = match.group(matchSymbolGroupIndex)
        handleNext = False
        if matchSymbolGroupIndex == 1:
            for reportLine in reportLines:
                if handleNext:
                    handleNext = False
                    pcdName = match.group(1)
                    match = re.match('\\s+(0x[0-9a-fA-F]{16})\\s+', reportLine)
                    if match is not None:
                        modSymbols[prefix + pcdName] = match.group(1)
                else:
                    match = re.match('^\\s\\.data\\.(_gPcd_BinaryPatch[_a-zA-Z0-9\\-]+)', reportLine)
                    if match is not None:
                        handleNext = True
                        continue
        if not moduleEntryPoint in modSymbols:
            if matchSymbolGroupIndex == 2:
                if not '_ModuleEntryPoint' in modSymbols:
                    return 1
                else:
                    moduleEntryPoint = '_ModuleEntryPoint'
            else:
                return 1
        modEntry = '%s:%s' % (moduleName, moduleEntryPoint)
        if not modEntry in self.dictSymbolAddress:
            modKey = '%s:ENTRY' % moduleName
            if modKey in self.dictModBase:
                baseOffset = self.dictModBase['%s:ENTRY' % moduleName] - int(modSymbols[moduleEntryPoint], 16)
            else:
                return 2
        else:
            baseOffset = int(self.dictSymbolAddress[modEntry], 16) - int(modSymbols[moduleEntryPoint], 16)
        for symbol in modSymbols:
            fullSym = '%s:%s' % (moduleName, symbol)
            if not fullSym in self.dictSymbolAddress:
                self.dictSymbolAddress[fullSym] = '0x00%08x' % (baseOffset + int(modSymbols[symbol], 16))
        return 0

    def parseGuidXrefFile(self, xrefFile):
        if False:
            for i in range(10):
                print('nop')
        fdIn = open(xrefFile, 'r')
        rptLine = fdIn.readline()
        while rptLine != '':
            match = re.match('([0-9a-fA-F\\-]+)\\s([_a-zA-Z0-9]+)', rptLine)
            if match is not None:
                self.dictGuidNameXref[match.group(1).upper()] = match.group(2)
            rptLine = fdIn.readline()
        fdIn.close()
        return 0

    def getCurr(self):
        if False:
            print('Hello World!')
        try:
            return self.string[self.index]
        except Exception:
            return ''

    def isLast(self):
        if False:
            print('Hello World!')
        return self.index == len(self.string)

    def moveNext(self):
        if False:
            while True:
                i = 10
        self.index += 1

    def skipSpace(self):
        if False:
            while True:
                i = 10
        while not self.isLast():
            if self.getCurr() in ' \t':
                self.moveNext()
            else:
                return

    def parseValue(self):
        if False:
            return 10
        self.skipSpace()
        var = ''
        while not self.isLast():
            char = self.getCurr()
            if char.lower() in '_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:-':
                var += char
                self.moveNext()
            else:
                break
        if ':' in var:
            partList = var.split(':')
            lenList = len(partList)
            if lenList != 2 and lenList != 3:
                raise Exception('Unrecognized expression %s' % var)
            modName = partList[lenList - 2]
            modOff = partList[lenList - 1]
            if '-' not in modName and modOff[0] in '0123456789':
                var = self.getModGuid(modName) + ':' + modOff
            if '-' in var:
                value = self.getGuidOff(var)
            else:
                value = self.getSymbols(var)
                self.synUsed = True
        elif var[0] in '0123456789':
            value = self.getNumber(var)
        else:
            value = self.getVariable(var)
        return int(value)

    def parseSingleOp(self):
        if False:
            i = 10
            return i + 15
        self.skipSpace()
        char = self.getCurr()
        if char == '~':
            self.moveNext()
            return ~self.parseBrace()
        else:
            return self.parseValue()

    def parseBrace(self):
        if False:
            while True:
                i = 10
        self.skipSpace()
        char = self.getCurr()
        parenthesisType = self.parenthesisOpenSet.find(char)
        if parenthesisType >= 0:
            self.moveNext()
            value = self.parseExpr()
            self.skipSpace()
            if self.getCurr() != self.parenthesisCloseSet[parenthesisType]:
                raise Exception('No closing brace')
            self.moveNext()
            if parenthesisType == 1:
                value = self.getContent(value)
            elif parenthesisType == 2:
                value = self.toAddress(value)
            elif parenthesisType == 3:
                value = self.toOffset(value)
            return value
        else:
            return self.parseSingleOp()

    def parseMul(self):
        if False:
            print('Hello World!')
        values = [self.parseBrace()]
        while True:
            self.skipSpace()
            char = self.getCurr()
            if char == '*':
                self.moveNext()
                values.append(self.parseBrace())
            else:
                break
        value = 1
        for each in values:
            value *= each
        return value

    def parseAndOr(self):
        if False:
            for i in range(10):
                print('nop')
        value = self.parseMul()
        op = None
        while True:
            self.skipSpace()
            char = self.getCurr()
            if char == '&':
                self.moveNext()
                value &= self.parseMul()
            elif char == '|':
                div_index = self.index
                self.moveNext()
                value |= self.parseMul()
            else:
                break
        return value

    def parseAddMinus(self):
        if False:
            return 10
        values = [self.parseAndOr()]
        while True:
            self.skipSpace()
            char = self.getCurr()
            if char == '+':
                self.moveNext()
                values.append(self.parseAndOr())
            elif char == '-':
                self.moveNext()
                values.append(-1 * self.parseAndOr())
            else:
                break
        return sum(values)

    def parseExpr(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parseAddMinus()

    def getResult(self):
        if False:
            while True:
                i = 10
        value = self.parseExpr()
        self.skipSpace()
        if not self.isLast():
            raise Exception("Unexpected character found '%s'" % self.getCurr())
        return value

    def getModGuid(self, var):
        if False:
            i = 10
            return i + 15
        guid = (guid for (guid, name) in self.dictGuidNameXref.items() if name == var)
        try:
            value = guid.next()
        except Exception:
            raise Exception('Unknown module name %s !' % var)
        return value

    def getVariable(self, var):
        if False:
            i = 10
            return i + 15
        value = self.dictVariable.get(var, None)
        if value == None:
            raise Exception("Unrecognized variable '%s'" % var)
        return value

    def getNumber(self, var):
        if False:
            return 10
        var = var.strip()
        if var.startswith('0x'):
            value = int(var, 16)
        else:
            value = int(var, 10)
        return value

    def getContent(self, value):
        if False:
            for i in range(10):
                print('nop')
        return readDataFromFile(self.fdFile, self.toOffset(value), 4)

    def toAddress(self, value):
        if False:
            i = 10
            return i + 15
        if value < self.fdSize:
            value = value + self.fdBase
        return value

    def toOffset(self, value):
        if False:
            while True:
                i = 10
        offset = None
        for fvInfo in self.fvList:
            if value >= fvInfo['Base'] and value < fvInfo['Base'] + fvInfo['Size']:
                offset = value - fvInfo['Base'] + fvInfo['Offset']
        if not offset:
            if value >= self.fdBase and value < self.fdBase + self.fdSize:
                offset = value - self.fdBase
            else:
                offset = value
        if offset >= self.fdSize:
            raise Exception('Invalid file offset 0x%08x !' % value)
        return offset

    def getGuidOff(self, value):
        if False:
            while True:
                i = 10
        symbolName = value.split(':')
        if len(symbolName) == 3:
            fvName = symbolName[0].upper()
            keyName = '%s:%s' % (fvName, symbolName[1])
            offStr = symbolName[2]
        elif len(symbolName) == 2:
            keyName = symbolName[0]
            offStr = symbolName[1]
        if keyName in self.dictFfsOffset:
            value = int(self.dictFfsOffset[keyName], 16) + int(offStr, 16) & 4294967295
        else:
            raise Exception('Unknown GUID %s !' % value)
        return value

    def getSymbols(self, value):
        if False:
            return 10
        if value in self.dictSymbolAddress:
            ret = int(self.dictSymbolAddress[value], 16)
        else:
            raise Exception('Unknown symbol %s !' % value)
        return ret

    def evaluate(self, expression, isOffset):
        if False:
            i = 10
            return i + 15
        self.index = 0
        self.synUsed = False
        self.string = expression
        value = self.getResult()
        if isOffset:
            if self.synUsed:
                value = self.toOffset(value)
            if value & 2147483648:
                offset = (~value & 4294967295) + 1
                if offset < self.fdSize:
                    value = self.fdSize - offset
            if value >= self.fdSize:
                raise Exception('Invalid offset expression !')
        return value & 4294967295

def Usage():
    if False:
        print('Hello World!')
    print('PatchFv Version 0.60')
    print('Usage: \n\tPatchFv FvBuildDir [FvFileBaseNames:]FdFileBaseNameToPatch "Offset, Value"')
    print('\tPatchFv FdFileDir FdFileName "Offset, Value"')

def main():
    if False:
        print('Hello World!')
    symTables = Symbols()
    if len(sys.argv) < 4:
        Usage()
        return 1
    if symTables.createDicts(sys.argv[1], sys.argv[2]) != 0:
        print('ERROR: Failed to create symbol dictionary!!')
        return 2
    fdFile = symTables.getFdFile()
    fdSize = symTables.getFdSize()
    try:
        ret = IsFspHeaderValid(fdFile)
        if ret == False:
            raise Exception('The FSP header is not valid. Stop patching FD.')
        comment = ''
        for fvFile in sys.argv[3:]:
            items = fvFile.split(',')
            if len(items) < 2:
                raise Exception("Expect more arguments for '%s'!" % fvFile)
            comment = ''
            command = ''
            params = []
            for item in items:
                item = item.strip()
                if item.startswith('@'):
                    comment = item[1:]
                elif item.startswith('$'):
                    command = item[1:]
                else:
                    if len(params) == 0:
                        isOffset = True
                    else:
                        isOffset = False
                    params.append(symTables.evaluate(item, isOffset))
            if command == '':
                if len(params) == 2:
                    offset = params[0]
                    value = params[1]
                    oldvalue = readDataFromFile(fdFile, offset, 4)
                    ret = patchDataInFile(fdFile, offset, value, 4) - 4
                else:
                    raise Exception('Patch command needs 2 parameters !')
                if ret:
                    raise Exception('Patch failed for offset 0x%08X' % offset)
                else:
                    print('Patched offset 0x%08X:[%08X] with value 0x%08X  # %s' % (offset, oldvalue, value, comment))
            elif command == 'COPY':
                if len(params) == 3:
                    src = symTables.toOffset(params[0])
                    dest = symTables.toOffset(params[1])
                    clen = symTables.toOffset(params[2])
                    if dest + clen <= fdSize and src + clen <= fdSize:
                        oldvalue = readDataFromFile(fdFile, src, clen)
                        ret = patchDataInFile(fdFile, dest, oldvalue, clen) - clen
                    else:
                        raise Exception('Copy command OFFSET or LENGTH parameter is invalid !')
                else:
                    raise Exception('Copy command needs 3 parameters !')
                if ret:
                    raise Exception('Copy failed from offset 0x%08X to offset 0x%08X!' % (src, dest))
                else:
                    print('Copied %d bytes from offset 0x%08X ~ offset 0x%08X  # %s' % (clen, src, dest, comment))
            else:
                raise Exception('Unknown command %s!' % command)
        return 0
    except Exception as ex:
        print('ERROR: %s' % ex)
        return 1
if __name__ == '__main__':
    sys.exit(main())