from __future__ import division
from __future__ import print_function
import sys
from struct import unpack
import ntpath
from six import b
from impacket import LOG
from impacket.structure import Structure, hexdump
ROOT_KEY = 44
REG_NONE = 0
REG_SZ = 1
REG_EXPAND_SZ = 2
REG_BINARY = 3
REG_DWORD = 4
REG_MULTISZ = 7
REG_QWORD = 11

class REG_REGF(Structure):
    structure = (('Magic', '"regf'), ('Unknown', '<L=0'), ('Unknown2', '<L=0'), ('lastChange', '<Q=0'), ('MajorVersion', '<L=0'), ('MinorVersion', '<L=0'), ('0', '<L=0'), ('11', '<L=0'), ('OffsetFirstRecord', '<L=0'), ('DataSize', '<L=0'), ('1111', '<L=0'), ('Name', '48s=""'), ('Remaining1', '411s=b""'), ('CheckSum', '<L=0xffffffff'), ('Remaining2', '3585s=b""'))

class REG_HBIN(Structure):
    structure = (('Magic', '"hbin'), ('OffsetFirstHBin', '<L=0'), ('OffsetNextHBin', '<L=0'), ('BlockSize', '<L=0'))

class REG_HBINBLOCK(Structure):
    structure = (('DataBlockSize', '<l=0'), ('_Data', '_-Data', 'self["DataBlockSize"]*(-1)-4'), ('Data', ':'))

class REG_NK(Structure):
    structure = (('Magic', '"nk'), ('Type', '<H=0'), ('lastChange', '<Q=0'), ('Unknown', '<L=0'), ('OffsetParent', '<l=0'), ('NumSubKeys', '<L=0'), ('Unknown2', '<L=0'), ('OffsetSubKeyLf', '<l=0'), ('Unknown3', '<L=0'), ('NumValues', '<L=0'), ('OffsetValueList', '<l=0'), ('OffsetSkRecord', '<l=0'), ('OffsetClassName', '<l=0'), ('UnUsed', '20s=b""'), ('NameLength', '<H=0'), ('ClassNameLength', '<H=0'), ('_KeyName', '_-KeyName', 'self["NameLength"]'), ('KeyName', ':'))

class REG_VK(Structure):
    structure = (('Magic', '"vk'), ('NameLength', '<H=0'), ('DataLen', '<l=0'), ('OffsetData', '<L=0'), ('ValueType', '<L=0'), ('Flag', '<H=0'), ('UnUsed', '<H=0'), ('_Name', '_-Name', 'self["NameLength"]'), ('Name', ':'))

class REG_LF(Structure):
    structure = (('Magic', '"lf'), ('NumKeys', '<H=0'), ('HashRecords', ':'))

class REG_LH(Structure):
    structure = (('Magic', '"lh'), ('NumKeys', '<H=0'), ('HashRecords', ':'))

class REG_RI(Structure):
    structure = (('Magic', '"ri'), ('NumKeys', '<H=0'), ('HashRecords', ':'))

class REG_SK(Structure):
    structure = (('Magic', '"sk'), ('UnUsed', '<H=0'), ('OffsetPreviousSk', '<l=0'), ('OffsetNextSk', '<l=0'), ('UsageCounter', '<L=0'), ('SizeSk', '<L=0'), ('Data', ':'))

class REG_HASH(Structure):
    structure = (('OffsetNk', '<L=0'), ('KeyName', '4s=b""'))
StructMappings = {b'nk': REG_NK, b'vk': REG_VK, b'lf': REG_LF, b'lh': REG_LH, b'ri': REG_RI, b'sk': REG_SK}

class Registry:

    def __init__(self, hive, isRemote=False):
        if False:
            for i in range(10):
                print('nop')
        self.__hive = hive
        if isRemote is True:
            self.fd = self.__hive
            self.__hive.open()
        else:
            self.fd = open(hive, 'rb')
        data = self.fd.read(4096)
        self.__regf = REG_REGF(data)
        self.indent = ''
        self.rootKey = self.__findRootKey()
        if self.rootKey is None:
            LOG.error("Can't find root key!")
        elif self.__regf['MajorVersion'] != 1 and self.__regf['MinorVersion'] > 5:
            LOG.warning('Unsupported version (%d.%d) - things might not work!' % (self.__regf['MajorVersion'], self.__regf['MinorVersion']))

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.fd.close()

    def __del__(self):
        if False:
            print('Hello World!')
        self.close()

    def __findRootKey(self):
        if False:
            while True:
                i = 10
        self.fd.seek(0, 0)
        data = self.fd.read(4096)
        while len(data) > 0:
            try:
                hbin = REG_HBIN(data[:32])
                data += self.fd.read(hbin['OffsetNextHBin'] - 4096)
                data = data[32:]
                blocks = self.__processDataBlocks(data)
                for block in blocks:
                    if isinstance(block, REG_NK):
                        if block['Type'] == ROOT_KEY:
                            return block
            except Exception as e:
                pass
            data = self.fd.read(4096)
        return None

    def __getBlock(self, offset):
        if False:
            return 10
        self.fd.seek(4096 + offset, 0)
        sizeBytes = self.fd.read(4)
        data = sizeBytes + self.fd.read(unpack('<l', sizeBytes)[0] * -1 - 4)
        if len(data) == 0:
            return None
        else:
            block = REG_HBINBLOCK(data)
            if block['Data'][:2] in StructMappings:
                return StructMappings[block['Data'][:2]](block['Data'])
            else:
                LOG.debug('Unknown type 0x%s' % block['Data'][:2])
                return block
            return None

    def __getValueBlocks(self, offset, count):
        if False:
            for i in range(10):
                print('nop')
        valueList = []
        res = []
        self.fd.seek(4096 + offset, 0)
        for i in range(count):
            valueList.append(unpack('<l', self.fd.read(4))[0])
        for valueOffset in valueList:
            if valueOffset > 0:
                block = self.__getBlock(valueOffset)
                res.append(block)
        return res

    def __getData(self, offset, count):
        if False:
            while True:
                i = 10
        self.fd.seek(4096 + offset, 0)
        return self.fd.read(count)[4:]

    def __processDataBlocks(self, data):
        if False:
            return 10
        res = []
        while len(data) > 0:
            blockSize = unpack('<l', data[:4])[0]
            block = REG_HBINBLOCK()
            if blockSize > 0:
                tmpList = list(block.structure)
                tmpList[1] = ('_Data', '_-Data', 'self["DataBlockSize"]-4')
                block.structure = tuple(tmpList)
            block.fromString(data)
            blockLen = len(block)
            if block['Data'][:2] in StructMappings:
                block = StructMappings[block['Data'][:2]](block['Data'])
            res.append(block)
            data = data[blockLen:]
        return res

    def __getValueData(self, rec):
        if False:
            for i in range(10):
                print('nop')
        if rec['DataLen'] == 0:
            return ''
        if rec['DataLen'] < 0:
            return rec['OffsetData']
        else:
            return self.__getData(rec['OffsetData'], rec['DataLen'] + 4)

    def __getLhHash(self, key):
        if False:
            i = 10
            return i + 15
        res = 0
        for bb in key.upper():
            res *= 37
            res += ord(bb)
        return res % 4294967296

    def __compareHash(self, magic, hashData, key):
        if False:
            return 10
        if magic == 'lf':
            hashRec = REG_HASH(hashData)
            if hashRec['KeyName'].strip(b'\x00') == b(key[:4]):
                return hashRec['OffsetNk']
        elif magic == 'lh':
            hashRec = REG_HASH(hashData)
            if unpack('<L', hashRec['KeyName'])[0] == self.__getLhHash(key):
                return hashRec['OffsetNk']
        elif magic == 'ri':
            offset = unpack('<L', hashData[:4])[0]
            nk = self.__getBlock(offset)
            if nk['KeyName'] == key:
                return offset
        else:
            LOG.critical('UNKNOWN Magic %s' % magic)
            sys.exit(1)
        return None

    def __findSubKey(self, parentKey, subKey):
        if False:
            while True:
                i = 10
        lf = self.__getBlock(parentKey['OffsetSubKeyLf'])
        if lf is not None:
            data = lf['HashRecords']
            if lf['Magic'] == 'ri':
                records = b''
                for i in range(lf['NumKeys']):
                    offset = unpack('<L', data[:4])[0]
                    l = self.__getBlock(offset)
                    records = records + l['HashRecords'][:l['NumKeys'] * 8]
                    data = data[4:]
                data = records
            for record in range(parentKey['NumSubKeys']):
                hashRec = data[:8]
                res = self.__compareHash(lf['Magic'], hashRec, subKey)
                if res is not None:
                    nk = self.__getBlock(res)
                    if nk['KeyName'].decode('utf-8') == subKey:
                        return nk
                data = data[8:]
        return None

    def __walkSubNodes(self, rec):
        if False:
            return 10
        nk = self.__getBlock(rec['OffsetNk'])
        if isinstance(nk, REG_NK):
            print('%s%s' % (self.indent, nk['KeyName'].decode('utf-8')))
            self.indent += '  '
            if nk['OffsetSubKeyLf'] < 0:
                self.indent = self.indent[:-2]
                return
            lf = self.__getBlock(nk['OffsetSubKeyLf'])
        else:
            lf = nk
        data = lf['HashRecords']
        if lf['Magic'] == 'ri':
            records = ''
            for i in range(lf['NumKeys']):
                offset = unpack('<L', data[:4])[0]
                l = self.__getBlock(offset)
                records = records + l['HashRecords'][:l['NumKeys'] * 8]
                data = data[4:]
            data = records
        for key in range(lf['NumKeys']):
            hashRec = REG_HASH(data[:8])
            self.__walkSubNodes(hashRec)
            data = data[8:]
        if isinstance(nk, REG_NK):
            self.indent = self.indent[:-2]

    def walk(self, parentKey):
        if False:
            for i in range(10):
                print('nop')
        key = self.findKey(parentKey)
        if key is None or key['OffsetSubKeyLf'] < 0:
            return
        lf = self.__getBlock(key['OffsetSubKeyLf'])
        data = lf['HashRecords']
        for record in range(lf['NumKeys']):
            hashRec = REG_HASH(data[:8])
            self.__walkSubNodes(hashRec)
            data = data[8:]

    def findKey(self, key):
        if False:
            while True:
                i = 10
        if key[0] == '\\' and len(key) > 1:
            key = key[1:]
        parentKey = self.rootKey
        if len(key) > 0 and key[0] != '\\':
            for subKey in key.split('\\'):
                res = self.__findSubKey(parentKey, subKey)
                if res is not None:
                    parentKey = res
                else:
                    return None
        return parentKey

    def printValue(self, valueType, valueData):
        if False:
            return 10
        if valueType in [REG_SZ, REG_EXPAND_SZ, REG_MULTISZ]:
            if isinstance(valueData, int):
                print('NULL')
            else:
                print('%s' % valueData.decode('utf-16le'))
        elif valueType == REG_BINARY:
            print('')
            hexdump(valueData, self.indent)
        elif valueType == REG_DWORD:
            print('%d' % valueData)
        elif valueType == REG_QWORD:
            print('%d' % unpack('<Q', valueData)[0])
        elif valueType == REG_NONE:
            try:
                if len(valueData) > 1:
                    print('')
                    hexdump(valueData, self.indent)
                else:
                    print(' NULL')
            except:
                print(' NULL')
        else:
            print('Unknown Type 0x%x!' % valueType)
            hexdump(valueData)

    def enumKey(self, parentKey):
        if False:
            print('Hello World!')
        res = []
        if parentKey['NumSubKeys'] > 0:
            lf = self.__getBlock(parentKey['OffsetSubKeyLf'])
            data = lf['HashRecords']
            if lf['Magic'] == 'ri':
                records = ''
                for i in range(lf['NumKeys']):
                    offset = unpack('<L', data[:4])[0]
                    l = self.__getBlock(offset)
                    records = records + l['HashRecords'][:l['NumKeys'] * 8]
                    data = data[4:]
                data = records
            for i in range(parentKey['NumSubKeys']):
                hashRec = REG_HASH(data[:8])
                nk = self.__getBlock(hashRec['OffsetNk'])
                data = data[8:]
                res.append('%s' % nk['KeyName'].decode('utf-8'))
        return res

    def enumValues(self, key):
        if False:
            print('Hello World!')
        resp = []
        if key['NumValues'] > 0:
            valueList = self.__getValueBlocks(key['OffsetValueList'], key['NumValues'] + 1)
            for value in valueList:
                if value['Flag'] > 0:
                    resp.append(value['Name'])
                else:
                    resp.append(b'default')
        return resp

    def getValue(self, keyValue):
        if False:
            print('Hello World!')
        regKey = ntpath.dirname(keyValue)
        regValue = ntpath.basename(keyValue)
        key = self.findKey(regKey)
        if key is None:
            return None
        if key['NumValues'] > 0:
            valueList = self.__getValueBlocks(key['OffsetValueList'], key['NumValues'] + 1)
            for value in valueList:
                if value['Name'] == b(regValue):
                    return (value['ValueType'], self.__getValueData(value))
                elif regValue == 'default' and value['Flag'] <= 0:
                    return (value['ValueType'], self.__getValueData(value))
        return None

    def getClass(self, className):
        if False:
            print('Hello World!')
        key = self.findKey(className)
        if key is None:
            return None
        if key['OffsetClassName'] > 0:
            value = self.__getBlock(key['OffsetClassName'])
            return value['Data']