from __future__ import division
from __future__ import print_function
import os
import sys
import logging
import struct
import argparse
import cmd
import ntpath
try:
    import pyreadline as readline
except ImportError:
    import readline
from six import PY2, text_type
from datetime import datetime
from impacket.examples import logger
from impacket import version
from impacket.structure import Structure

def pretty_print(x):
    if False:
        while True:
            i = 10
    visible = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
    return x if x in visible else '.'

def hexdump(data):
    if False:
        i = 10
        return i + 15
    x = str(data)
    strLen = len(x)
    i = 0
    while i < strLen:
        print('%04x  ' % i, end=' ')
        for j in range(16):
            if i + j < strLen:
                print('%02X' % ord(x[i + j]), end=' ')
            else:
                print('  ', end=' ')
            if j % 16 == 7:
                print('', end=' ')
        print(' ', end=' ')
        print(''.join((pretty_print(x) for x in x[i:i + 16])))
        i += 16
FIXED_MFTS = 16
UNUSED = 0
STANDARD_INFORMATION = 16
ATTRIBUTE_LIST = 32
FILE_NAME = 48
OBJECT_ID = 64
SECURITY_DESCRIPTOR = 80
VOLUME_NAME = 96
VOLUME_INFORMATION = 112
DATA = 128
INDEX_ROOT = 144
INDEX_ALLOCATION = 160
BITMAP = 176
REPARSE_POINT = 192
EA_INFORMATION = 208
EA = 224
PROPERTY_SET = 240
LOGGED_UTILITY_STREAM = 256
FIRST_USER_DEFINED_ATTRIBUTE = 4096
END = 4294967295
ATTR_IS_COMPRESSED = 1
ATTR_COMPRESSION_MASK = 255
ATTR_IS_ENCRYPTED = 16384
ATTR_IS_SPARSE = 32768
FILE_NAME_POSIX = 0
FILE_NAME_WIN32 = 1
FILE_NAME_DOS = 2
FILE_NAME_WIN32_AND_DOS = 3
MFT_RECORD_IN_USE = 1
MFT_RECORD_IS_DIRECTORY = 2
MFT_RECORD_IS_4 = 4
MFT_RECORD_IS_VIEW_INDEX = 8
MFT_REC_SPACE_FILLER = 1048575
FILE_ATTR_READONLY = 1
FILE_ATTR_HIDDEN = 2
FILE_ATTR_SYSTEM = 4
FILE_ATTR_DIRECTORY = 16
FILE_ATTR_ARCHIVE = 32
FILE_ATTR_DEVICE = 64
FILE_ATTR_NORMAL = 128
FILE_ATTR_TEMPORARY = 256
FILE_ATTR_SPARSE_FILE = 512
FILE_ATTR_REPARSE_POINT = 1024
FILE_ATTR_COMPRESSED = 2048
FILE_ATTR_OFFLINE = 4096
FILE_ATTR_NOT_CONTENT_INDEXED = 8192
FILE_ATTR_ENCRYPTED = 16384
FILE_ATTR_VALID_FLAGS = 32695
FILE_ATTR_VALID_SET_FLAGS = 12711
FILE_ATTR_I30_INDEX_PRESENT = 268435456
FILE_ATTR_VIEW_INDEX_PRESENT = 536870912
FILE_MFT = 0
FILE_MFTMirr = 1
FILE_LogFile = 2
FILE_Volume = 3
FILE_AttrDef = 4
FILE_Root = 5
FILE_Bitmap = 6
FILE_Boot = 7
FILE_BadClus = 8
FILE_Secure = 9
FILE_UpCase = 10
FILE_Extend = 11
SMALL_INDEX = 0
LARGE_INDEX = 1
LEAF_NODE = 0
INDEX_NODE = 1
NODE_MASK = 0
INDEX_ENTRY_NODE = 1
INDEX_ENTRY_END = 2
INDEX_ENTRY_SPACE_FILLER = 65535

class NTFS_BPB(Structure):
    structure = (('BytesPerSector', '<H=0'), ('SectorsPerCluster', 'B=0'), ('ReservedSectors', '<H=0'), ('Reserved', '3s=b""'), ('Reserved2', '2s=b""'), ('MediaDescription', 'B=0'), ('Reserved3', '2s=b""'), ('Reserved4', '<H=0'), ('Reserved5', '<H=0'), ('Reserved6', '<L=0'), ('Reserved7', '4s=b""'))

class NTFS_EXTENDED_BPB(Structure):
    structure = (('Reserved', '4s=b""'), ('TotalSectors', '<Q=0'), ('MFTClusterNumber', '<Q=0'), ('MFTMirrClusterNumber', '<Q=0'), ('ClusterPerFileRecord', 'b=0'), ('Reserved2', '3s=b""'), ('ClusterPerIndexBuffer', '<b=0'), ('Reserved3', '3s=b""'), ('VolumeSerialNumber', '8s=b""'), ('CheckSum', '4s=b""'))

class NTFS_BOOT_SECTOR(Structure):
    structure = (('JmpInstr', '3s=b""'), ('OEM_ID', '8s=b""'), ('BPB', '25s=b""'), ('ExtendedBPB', '48s=b""'), ('Bootstrap', '426s=b""'), ('EOS', '<H=0'))

class NTFS_MFT_RECORD(Structure):
    structure = (('MagicLabel', '4s=b""'), ('USROffset', '<H=0'), ('USRSize', '<H=0'), ('LogSeqNum', '<Q=0'), ('SeqNum', '<H=0'), ('LinkCount', '<H=0'), ('AttributesOffset', '<H=0'), ('Flags', '<H=0'), ('BytesInUse', '<L=0'), ('BytesAllocated', '<L=0'), ('BaseMftRecord', '<Q=0'), ('NextAttrInstance', '<H=0'), ('Reserved', '<H=0'), ('RecordNumber', '<L=0'))

class NTFS_ATTRIBUTE_RECORD(Structure):
    commonHdr = (('Type', '<L=0'), ('Length', '<L=0'), ('NonResident', 'B=0'), ('NameLength', 'B=0'), ('NameOffset', '<H=0'), ('Flags', '<H=0'), ('Instance', '<H=0'))
    structure = ()

class NTFS_ATTRIBUTE_RECORD_NON_RESIDENT(Structure):
    structure = (('LowestVCN', '<Q=0'), ('HighestVCN', '<Q=0'), ('DataRunsOffset', '<H=0'), ('CompressionUnit', '<H=0'), ('Reserved1', '4s=""'), ('AllocatedSize', '<Q=0'), ('DataSize', '<Q=0'), ('InitializedSize', '<Q=0'))

class NTFS_ATTRIBUTE_RECORD_RESIDENT(Structure):
    structure = (('ValueLen', '<L=0'), ('ValueOffset', '<H=0'), ('Flags', 'B=0'), ('Reserved', 'B=0'))

class NTFS_FILE_NAME_ATTR(Structure):
    structure = (('ParentDirectory', '<Q=0'), ('CreationTime', '<Q=0'), ('LastDataChangeTime', '<Q=0'), ('LastMftChangeTime', '<Q=0'), ('LastAccessTime', '<Q=0'), ('AllocatedSize', '<Q=0'), ('DataSize', '<Q=0'), ('FileAttributes', '<L=0'), ('EaSize', '<L=0'), ('FileNameLen', 'B=0'), ('FileNameType', 'B=0'), ('_FileName', '_-FileName', 'self["FileNameLen"]*2'), ('FileName', ':'))

class NTFS_STANDARD_INFORMATION(Structure):
    structure = (('CreationTime', '<Q=0'), ('LastDataChangeTime', '<Q=0'), ('LastMftChangeTime', '<Q=0'), ('LastAccessTime', '<Q=0'), ('FileAttributes', '<L=0'))

class NTFS_INDEX_HEADER(Structure):
    structure = (('EntriesOffset', '<L=0'), ('IndexLength', '<L=0'), ('AllocatedSize', '<L=0'), ('Flags', 'B=0'), ('Reserved', '3s=b""'))

class NTFS_INDEX_ROOT(Structure):
    structure = (('Type', '<L=0'), ('CollationRule', '<L=0'), ('IndexBlockSize', '<L=0'), ('ClustersPerIndexBlock', 'B=0'), ('Reserved', '3s=b""'), ('Index', ':', NTFS_INDEX_HEADER))

class NTFS_INDEX_ALLOCATION(Structure):
    structure = (('Magic', '4s=b""'), ('USROffset', '<H=0'), ('USRSize', '<H=0'), ('Lsn', '<Q=0'), ('IndexVcn', '<Q=0'), ('Index', ':', NTFS_INDEX_HEADER))

class NTFS_INDEX_ENTRY_HEADER(Structure):
    structure = (('IndexedFile', '<Q=0'), ('Length', '<H=0'), ('KeyLength', '<H=0'), ('Flags', '<H=0'), ('Reserved', '<H=0'))

class NTFS_INDEX_ENTRY(Structure):
    alignment = 8
    structure = (('EntryHeader', ':', NTFS_INDEX_ENTRY_HEADER), ('_Key', '_-Key', 'self["EntryHeader"]["KeyLength"]'), ('Key', ':'), ('_Vcn', '_-Vcn', '(self["EntryHeader"]["Flags"] & 1)*8'), ('Vcn', ':'))

class NTFS_DATA_RUN(Structure):
    structure = (('LCN', '<q=0'), ('Clusters', '<Q=0'), ('StartVCN', '<Q=0'), ('LastVCN', '<Q=0'))

def getUnixTime(t):
    if False:
        print('Hello World!')
    t -= 116444736000000000
    t //= 10000000
    return t

class Attribute:

    def __init__(self, iNode, data):
        if False:
            print('Hello World!')
        self.AttributeName = None
        self.NTFSVolume = iNode.NTFSVolume
        self.AttributeHeader = NTFS_ATTRIBUTE_RECORD(data)
        if self.AttributeHeader['NameLength'] > 0 and self.AttributeHeader['Type'] != END:
            self.AttributeName = data[self.AttributeHeader['NameOffset']:][:self.AttributeHeader['NameLength'] * 2].decode('utf-16le')

    def getFlags(self):
        if False:
            return 10
        return self.AttributeHeader['Flags']

    def getName(self):
        if False:
            i = 10
            return i + 15
        return self.AttributeName

    def isNonResident(self):
        if False:
            print('Hello World!')
        return self.AttributeHeader['NonResident']

    def dump(self):
        if False:
            while True:
                i = 10
        return self.AttributeHeader.dump()

    def getTotalSize(self):
        if False:
            return 10
        return self.AttributeHeader['Length']

    def getType(self):
        if False:
            while True:
                i = 10
        return self.AttributeHeader['Type']

class AttributeResident(Attribute):

    def __init__(self, iNode, data):
        if False:
            return 10
        logging.debug('Inside AttributeResident: iNode: %s' % iNode.INodeNumber)
        Attribute.__init__(self, iNode, data)
        self.ResidentHeader = NTFS_ATTRIBUTE_RECORD_RESIDENT(data[len(self.AttributeHeader):])
        self.AttrValue = data[self.ResidentHeader['ValueOffset']:][:self.ResidentHeader['ValueLen']]

    def dump(self):
        if False:
            return 10
        return self.ResidentHeader.dump()

    def getFlags(self):
        if False:
            return 10
        return self.ResidentHeader['Flags']

    def getValue(self):
        if False:
            return 10
        return self.AttrValue

    def read(self, offset, length):
        if False:
            while True:
                i = 10
        logging.debug('Inside Read: offset: %d, length: %d' % (offset, length))
        return self.AttrValue[offset:][:length]

    def getDataSize(self):
        if False:
            return 10
        return len(self.AttrValue)

class AttributeNonResident(Attribute):

    def __init__(self, iNode, data):
        if False:
            return 10
        logging.debug('Inside AttributeNonResident: iNode: %s' % iNode.INodeNumber)
        Attribute.__init__(self, iNode, data)
        self.NonResidentHeader = NTFS_ATTRIBUTE_RECORD_NON_RESIDENT(data[len(self.AttributeHeader):])
        self.AttrValue = data[self.NonResidentHeader['DataRunsOffset']:][:self.NonResidentHeader['AllocatedSize']]
        self.DataRuns = []
        self.ClusterSize = 0
        self.parseDataRuns()

    def dump(self):
        if False:
            while True:
                i = 10
        return self.NonResidentHeader.dump()

    def getDataSize(self):
        if False:
            i = 10
            return i + 15
        return self.NonResidentHeader['InitializedSize']

    def getValue(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def parseDataRuns(self):
        if False:
            for i in range(10):
                print('nop')
        value = self.AttrValue
        if value is not None:
            VCN = 0
            LCN = 0
            LCNOffset = 0
            while value[0:1] != b'\x00':
                LCN += LCNOffset
                dr = NTFS_DATA_RUN()
                size = struct.unpack('B', value[0:1])[0]
                value = value[1:]
                lengthBytes = size & 15
                offsetBytes = size >> 4
                length = value[:lengthBytes]
                length = struct.unpack('<Q', value[:lengthBytes] + b'\x00' * (8 - len(length)))[0]
                value = value[lengthBytes:]
                fillWith = b'\x00'
                if struct.unpack('B', value[offsetBytes - 1:offsetBytes])[0] & 128:
                    fillWith = b'\xff'
                LCNOffset = value[:offsetBytes] + fillWith * (8 - len(value[:offsetBytes]))
                LCNOffset = struct.unpack('<q', LCNOffset)[0]
                value = value[offsetBytes:]
                dr['LCN'] = LCN + LCNOffset
                dr['Clusters'] = length
                dr['StartVCN'] = VCN
                dr['LastVCN'] = VCN + length - 1
                VCN += length
                self.DataRuns.append(dr)
                if len(value) == 0:
                    break

    def readClusters(self, clusters, lcn):
        if False:
            i = 10
            return i + 15
        logging.debug('Inside ReadClusters: clusters:%d, lcn:%d' % (clusters, lcn))
        if lcn == -1:
            return '\x00' * clusters * self.ClusterSize
        self.NTFSVolume.volumeFD.seek(lcn * self.ClusterSize, 0)
        buf = self.NTFSVolume.volumeFD.read(clusters * self.ClusterSize)
        while len(buf) < clusters * self.ClusterSize:
            buf += self.NTFSVolume.volumeFD.read(clusters * self.ClusterSize - len(buf))
        if len(buf) == 0:
            return None
        return buf

    def readVCN(self, vcn, numOfClusters):
        if False:
            i = 10
            return i + 15
        logging.debug('Inside ReadVCN: vcn: %d, numOfClusters: %d' % (vcn, numOfClusters))
        buf = b''
        clustersLeft = numOfClusters
        for dr in self.DataRuns:
            if vcn >= dr['StartVCN'] and vcn <= dr['LastVCN']:
                vcnsToRead = dr['LastVCN'] - vcn + 1
                if numOfClusters > vcnsToRead:
                    clustersToRead = vcnsToRead
                else:
                    clustersToRead = numOfClusters
                tmpBuf = self.readClusters(clustersToRead, dr['LCN'] + (vcn - dr['StartVCN']))
                if tmpBuf is not None:
                    buf += tmpBuf
                    clustersLeft -= clustersToRead
                    vcn += clustersToRead
                else:
                    break
                if clustersLeft == 0:
                    break
        return buf

    def read(self, offset, length):
        if False:
            return 10
        logging.debug('Inside Read: offset: %d, length: %d' % (offset, length))
        buf = b''
        curLength = length
        self.ClusterSize = self.NTFSVolume.BPB['BytesPerSector'] * self.NTFSVolume.BPB['SectorsPerCluster']
        vcnToStart = offset // self.ClusterSize
        if offset % self.ClusterSize:
            bufTemp = self.readVCN(vcnToStart, 1)
            if bufTemp == b'':
                return None
            buf = bufTemp[offset % self.ClusterSize:]
            curLength -= len(buf)
            vcnToStart += 1
        if curLength <= 0:
            return buf[:length]
        if curLength // self.ClusterSize:
            bufTemp = self.readVCN(vcnToStart, curLength // self.ClusterSize)
            if bufTemp == b'':
                return None
            if len(bufTemp) > curLength:
                buf = buf + bufTemp[:curLength]
            else:
                buf = buf + bufTemp
            vcnToStart += curLength // self.ClusterSize
            curLength -= len(bufTemp)
        if curLength > 0:
            bufTemp = self.readVCN(vcnToStart, 1)
            buf = buf + bufTemp[:curLength]
        if buf == b'':
            return None
        else:
            return buf

class AttributeStandardInfo:

    def __init__(self, attribute):
        if False:
            print('Hello World!')
        logging.debug('Inside AttributeStandardInfo')
        self.Attribute = attribute
        self.StandardInfo = NTFS_STANDARD_INFORMATION(self.Attribute.AttrValue)

    def getFileAttributes(self):
        if False:
            while True:
                i = 10
        return self.StandardInfo['FileAttributes']

    def getFileTime(self):
        if False:
            return 10
        if self.StandardInfo['LastDataChangeTime'] > 0:
            return datetime.fromtimestamp(getUnixTime(self.StandardInfo['LastDataChangeTime']))
        else:
            return 0

    def dump(self):
        if False:
            i = 10
            return i + 15
        return self.StandardInfo.dump()

class AttributeFileName:

    def __init__(self, attribute):
        if False:
            while True:
                i = 10
        logging.debug('Inside AttributeFileName')
        self.Attribute = attribute
        self.FileNameRecord = NTFS_FILE_NAME_ATTR(self.Attribute.AttrValue)

    def getFileNameType(self):
        if False:
            return 10
        return self.FileNameRecord['FileNameType']

    def getFileAttributes(self):
        if False:
            print('Hello World!')
        return self.FileNameRecord['FileAttributes']

    def getFileName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.FileNameRecord['FileName'].decode('utf-16le')

    def getFileSize(self):
        if False:
            for i in range(10):
                print('nop')
        return self.FileNameRecord['DataSize']

    def getFlags(self):
        if False:
            i = 10
            return i + 15
        return self.FileNameRecord['FileAttributes']

    def dump(self):
        if False:
            print('Hello World!')
        return self.FileNameRecord.dump()

class AttributeIndexAllocation:

    def __init__(self, attribute):
        if False:
            i = 10
            return i + 15
        logging.debug('Inside AttributeIndexAllocation')
        self.Attribute = attribute

    def dump(self):
        if False:
            while True:
                i = 10
        print(self.Attribute.dump())
        for i in self.Attribute.DataRuns:
            print(i.dump())

    def read(self, offset, length):
        if False:
            print('Hello World!')
        return self.Attribute.read(offset, length)

class AttributeIndexRoot:

    def __init__(self, attribute):
        if False:
            i = 10
            return i + 15
        logging.debug('Inside AttributeIndexRoot')
        self.Attribute = attribute
        self.IndexRootRecord = NTFS_INDEX_ROOT(attribute.AttrValue)
        self.IndexEntries = []
        self.parseIndexEntries()

    def parseIndexEntries(self):
        if False:
            while True:
                i = 10
        data = self.Attribute.AttrValue[len(self.IndexRootRecord):]
        while True:
            ie = IndexEntry(data)
            self.IndexEntries.append(ie)
            if ie.isLastNode():
                break
            data = data[ie.getSize():]

    def dump(self):
        if False:
            print('Hello World!')
        self.IndexRootRecord.dump()
        for i in self.IndexEntries:
            i.dump()

    def getType(self):
        if False:
            print('Hello World!')
        return self.IndexRootRecord['Type']

class IndexEntry:

    def __init__(self, entry):
        if False:
            while True:
                i = 10
        self.entry = NTFS_INDEX_ENTRY(entry)

    def isSubNode(self):
        if False:
            print('Hello World!')
        return self.entry['EntryHeader']['Flags'] & INDEX_ENTRY_NODE

    def isLastNode(self):
        if False:
            print('Hello World!')
        return self.entry['EntryHeader']['Flags'] & INDEX_ENTRY_END

    def getVCN(self):
        if False:
            i = 10
            return i + 15
        return struct.unpack('<Q', self.entry['Vcn'])[0]

    def getSize(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.entry)

    def getKey(self):
        if False:
            for i in range(10):
                print('nop')
        return self.entry['Key']

    def getINodeNumber(self):
        if False:
            i = 10
            return i + 15
        return self.entry['EntryHeader']['IndexedFile'] & 281474976710655

    def dump(self):
        if False:
            return 10
        self.entry.dump()

class INODE:

    def __init__(self, NTFSVolume):
        if False:
            i = 10
            return i + 15
        self.NTFSVolume = NTFSVolume
        self.INodeNumber = None
        self.Attributes = {}
        self.AttributesRaw = None
        self.AttributesLastPos = None
        self.FileAttributes = 0
        self.LastDataChangeTime = None
        self.FileName = None
        self.FileSize = 0

    def isDirectory(self):
        if False:
            return 10
        return self.FileAttributes & FILE_ATTR_I30_INDEX_PRESENT

    def isCompressed(self):
        if False:
            print('Hello World!')
        return self.FileAttributes & FILE_ATTR_COMPRESSED

    def isEncrypted(self):
        if False:
            i = 10
            return i + 15
        return self.FileAttributes & FILE_ATTR_ENCRYPTED

    def isSparse(self):
        if False:
            i = 10
            return i + 15
        return self.FileAttributes & FILE_ATTR_SPARSE_FILE

    def displayName(self):
        if False:
            i = 10
            return i + 15
        if self.LastDataChangeTime is not None and self.FileName is not None:
            try:
                print('%s %s %15d %s ' % (self.getPrintableAttributes(), self.LastDataChangeTime.isoformat(' '), self.FileSize, self.FileName))
            except Exception as e:
                logging.error('Exception when trying to display inode %d: %s' % (self.INodeNumber, str(e)))

    def getPrintableAttributes(self):
        if False:
            print('Hello World!')
        mask = ''
        if self.FileAttributes & FILE_ATTR_I30_INDEX_PRESENT:
            mask += 'd'
        else:
            mask += '-'
        if self.FileAttributes & FILE_ATTR_HIDDEN:
            mask += 'h'
        else:
            mask += '-'
        if self.FileAttributes & FILE_ATTR_SYSTEM:
            mask += 'S'
        else:
            mask += '-'
        if self.isCompressed():
            mask += 'C'
        else:
            mask += '-'
        if self.isEncrypted():
            mask += 'E'
        else:
            mask += '-'
        if self.isSparse():
            mask += 's'
        else:
            mask += '-'
        return mask

    def parseAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        attr = self.searchAttribute(STANDARD_INFORMATION, None)
        if attr is not None:
            si = AttributeStandardInfo(attr)
            self.Attributes[STANDARD_INFORMATION] = si
            self.FileAttributes |= si.getFileAttributes()
            self.LastDataChangeTime = si.getFileTime()
            self.Attributes[STANDARD_INFORMATION] = si
        attr = self.searchAttribute(FILE_NAME, None)
        while attr is not None:
            fn = AttributeFileName(attr)
            if fn.getFileNameType() != FILE_NAME_DOS:
                self.FileName = fn.getFileName()
                self.FileSize = fn.getFileSize()
                self.FileAttributes |= fn.getFileAttributes()
                self.Attributes[FILE_NAME] = fn
                break
            attr = self.searchAttribute(FILE_NAME, None, True)
        attr = self.searchAttribute(INDEX_ALLOCATION, u'$I30')
        if attr is not None:
            ia = AttributeIndexAllocation(attr)
            self.Attributes[INDEX_ALLOCATION] = ia
        attr = self.searchAttribute(INDEX_ROOT, u'$I30')
        if attr is not None:
            ir = AttributeIndexRoot(attr)
            self.Attributes[INDEX_ROOT] = ir

    def searchAttribute(self, attributeType, attributeName, findNext=False):
        if False:
            return 10
        logging.debug('Inside searchAttribute: type: 0x%x, name: %s' % (attributeType, attributeName))
        record = None
        if findNext is True:
            data = self.AttributesLastPos
        else:
            data = self.AttributesRaw
        while True:
            if len(data) <= 8:
                record = None
                break
            record = Attribute(self, data)
            if record.getType() == END:
                record = None
                break
            if record.getTotalSize() == 0:
                record = None
                break
            if record.getType() == attributeType and record.getName() == attributeName:
                if record.isNonResident() == 1:
                    record = AttributeNonResident(self, data)
                else:
                    record = AttributeResident(self, data)
                self.AttributesLastPos = data[record.getTotalSize():]
                break
            data = data[record.getTotalSize():]
        return record

    def PerformFixUp(self, record, buf, numSectors):
        if False:
            i = 10
            return i + 15
        logging.debug('Inside PerformFixUp...')
        magicNum = struct.unpack('<H', buf[record['USROffset']:][:2])[0]
        sequenceArray = buf[record['USROffset'] + 2:][:record['USRSize'] * 2]
        dataList = list(buf)
        index = 0
        for i in range(0, numSectors * 2, 2):
            index += self.NTFSVolume.SectorSize - 2
            lastBytes = struct.unpack('<H', buf[index:][:2])[0]
            if lastBytes != magicNum:
                logging.error("Magic number 0x%x doesn't match with 0x%x" % (magicNum, lastBytes))
                return None
            dataList[index] = sequenceArray[i]
            dataList[index + 1] = sequenceArray[i + 1]
            index += 2
        if PY2:
            return ''.join(dataList)
        else:
            return bytes(dataList)

    def parseIndexBlocks(self, vcn):
        if False:
            while True:
                i = 10
        IndexEntries = []
        if INDEX_ALLOCATION in self.Attributes:
            ia = self.Attributes[INDEX_ALLOCATION]
            data = ia.read(vcn * self.NTFSVolume.IndexBlockSize, self.NTFSVolume.IndexBlockSize)
            if data:
                iaRec = NTFS_INDEX_ALLOCATION(data)
                sectorsPerIB = self.NTFSVolume.IndexBlockSize // self.NTFSVolume.SectorSize
                data = self.PerformFixUp(iaRec, data, sectorsPerIB)
                if data is None:
                    return []
                data = data[len(iaRec) - len(NTFS_INDEX_HEADER()) + iaRec['Index']['EntriesOffset']:]
                while True:
                    ie = IndexEntry(data)
                    IndexEntries.append(ie)
                    if ie.isLastNode():
                        break
                    data = data[ie.getSize():]
        return IndexEntries

    def walkSubNodes(self, vcn):
        if False:
            for i in range(10):
                print('nop')
        logging.debug('Inside walkSubNodes: vcn %s' % vcn)
        entries = self.parseIndexBlocks(vcn)
        files = []
        for entry in entries:
            if entry.isSubNode():
                files += self.walkSubNodes(entry.getVCN())
            elif len(entry.getKey()) > 0 and entry.getINodeNumber() > 16:
                fn = NTFS_FILE_NAME_ATTR(entry.getKey())
                if fn['FileNameType'] != FILE_NAME_DOS:
                    files.append(fn)
        return files

    def walk(self):
        if False:
            while True:
                i = 10
        logging.debug('Inside Walk... ')
        files = []
        if INDEX_ROOT in self.Attributes:
            ir = self.Attributes[INDEX_ROOT]
            if ir.getType() & FILE_NAME:
                for ie in ir.IndexEntries:
                    if ie.isSubNode():
                        files += self.walkSubNodes(ie.getVCN())
                return files
        else:
            return None

    def findFirstSubNode(self, vcn, toSearch):
        if False:
            print('Hello World!')

        def getFileName(entry):
            if False:
                i = 10
                return i + 15
            if len(entry.getKey()) > 0 and entry.getINodeNumber() > 16:
                fn = NTFS_FILE_NAME_ATTR(entry.getKey())
                if fn['FileNameType'] != FILE_NAME_DOS:
                    return fn['FileName'].decode('utf-16le').upper()
            return None
        entries = self.parseIndexBlocks(vcn)
        for ie in entries:
            name = getFileName(ie)
            if name is not None:
                if name == toSearch:
                    return ie
                if toSearch < name:
                    if ie.isSubNode():
                        res = self.findFirstSubNode(ie.getVCN(), toSearch)
                        if res is not None:
                            return res
                    else:
                        return None
            elif ie.isSubNode():
                res = self.findFirstSubNode(ie.getVCN(), toSearch)
                if res is not None:
                    return res

    def findFirst(self, fileName):
        if False:
            print('Hello World!')

        def getFileName(entry):
            if False:
                for i in range(10):
                    print('nop')
            if len(entry.getKey()) > 0 and entry.getINodeNumber() > 16:
                fn = NTFS_FILE_NAME_ATTR(entry.getKey())
                if fn['FileNameType'] != FILE_NAME_DOS:
                    return fn['FileName'].decode('utf-16le').upper()
            return None
        toSearch = text_type(fileName.upper())
        if INDEX_ROOT in self.Attributes:
            ir = self.Attributes[INDEX_ROOT]
            if ir.getType() & FILE_NAME or 1 == 1:
                for ie in ir.IndexEntries:
                    name = getFileName(ie)
                    if name is not None:
                        if name == toSearch:
                            return ie
                        if toSearch < name:
                            if ie.isSubNode():
                                res = self.findFirstSubNode(ie.getVCN(), toSearch)
                                if res is not None:
                                    return res
                            else:
                                return None
                    elif ie.isSubNode():
                        res = self.findFirstSubNode(ie.getVCN(), toSearch)
                        if res is not None:
                            return res

    def getStream(self, name):
        if False:
            i = 10
            return i + 15
        return self.searchAttribute(DATA, name, findNext=False)

class NTFS:

    def __init__(self, volumeName):
        if False:
            print('Hello World!')
        self.__volumeName = volumeName
        self.__bootSector = None
        self.__MFTStart = None
        self.volumeFD = None
        self.BPB = None
        self.ExtendedBPB = None
        self.RecordSize = None
        self.IndexBlockSize = None
        self.SectorSize = None
        self.MFTINode = None
        self.mountVolume()

    def mountVolume(self):
        if False:
            print('Hello World!')
        logging.debug('Mounting volume...')
        self.volumeFD = open(self.__volumeName, 'rb')
        self.readBootSector()
        self.MFTINode = self.getINode(FILE_MFT)
        attr = self.MFTINode.searchAttribute(DATA, None)
        if attr is None:
            del self.MFTINode
            self.MFTINode = None

    def readBootSector(self):
        if False:
            i = 10
            return i + 15
        logging.debug('Reading Boot Sector for %s' % self.__volumeName)
        self.volumeFD.seek(0, 0)
        data = self.volumeFD.read(512)
        while len(data) < 512:
            data += self.volumeFD.read(512)
        self.__bootSector = NTFS_BOOT_SECTOR(data)
        self.BPB = NTFS_BPB(self.__bootSector['BPB'])
        self.ExtendedBPB = NTFS_EXTENDED_BPB(self.__bootSector['ExtendedBPB'])
        self.SectorSize = self.BPB['BytesPerSector']
        self.__MFTStart = self.BPB['BytesPerSector'] * self.BPB['SectorsPerCluster'] * self.ExtendedBPB['MFTClusterNumber']
        if self.ExtendedBPB['ClusterPerFileRecord'] > 0:
            self.RecordSize = self.BPB['BytesPerSector'] * self.BPB['SectorsPerCluster'] * self.ExtendedBPB['ClusterPerFileRecord']
        else:
            self.RecordSize = 1 << -self.ExtendedBPB['ClusterPerFileRecord']
        if self.ExtendedBPB['ClusterPerIndexBuffer'] > 0:
            self.IndexBlockSize = self.BPB['BytesPerSector'] * self.BPB['SectorsPerCluster'] * self.ExtendedBPB['ClusterPerIndexBuffer']
        else:
            self.IndexBlockSize = 1 << -self.ExtendedBPB['ClusterPerIndexBuffer']
        logging.debug('MFT should start at position %d' % self.__MFTStart)

    def getINode(self, iNodeNum):
        if False:
            return 10
        logging.debug('Trying to fetch inode %d' % iNodeNum)
        newINode = INODE(self)
        recordLen = self.RecordSize
        if self.MFTINode and iNodeNum > FIXED_MFTS:
            attr = self.MFTINode.searchAttribute(DATA, None)
            record = attr.read(iNodeNum * self.RecordSize, self.RecordSize)
        else:
            diskPosition = self.__MFTStart + iNodeNum * self.RecordSize
            self.volumeFD.seek(diskPosition, 0)
            record = self.volumeFD.read(recordLen)
            while len(record) < recordLen:
                record += self.volumeFD.read(recordLen - len(record))
        mftRecord = NTFS_MFT_RECORD(record)
        record = newINode.PerformFixUp(mftRecord, record, self.RecordSize // self.SectorSize)
        newINode.INodeNumber = iNodeNum
        newINode.AttributesRaw = record[mftRecord['AttributesOffset'] - recordLen:]
        newINode.parseAttributes()
        return newINode

class MiniShell(cmd.Cmd):

    def __init__(self, volume):
        if False:
            while True:
                i = 10
        cmd.Cmd.__init__(self)
        self.volumePath = volume
        self.volume = NTFS(volume)
        self.rootINode = self.volume.getINode(5)
        self.prompt = '\\>'
        self.intro = 'Type help for list of commands'
        self.currentINode = self.rootINode
        self.completion = []
        self.pwd = '\\'
        self.do_ls('', False)
        self.last_output = ''

    def emptyline(self):
        if False:
            print('Hello World!')
        pass

    def onecmd(self, s):
        if False:
            print('Hello World!')
        retVal = False
        try:
            retVal = cmd.Cmd.onecmd(self, s)
        except Exception as e:
            logging.debug('Exception:', exc_info=True)
            logging.error(str(e))
        return retVal

    def do_exit(self, line):
        if False:
            return 10
        return True

    def do_shell(self, line):
        if False:
            while True:
                i = 10
        output = os.popen(line).read()
        print(output)
        self.last_output = output

    def do_help(self, line):
        if False:
            for i in range(10):
                print('nop')
        print('\n cd {path} - changes the current directory to {path}\n pwd - shows current remote directory\n ls  - lists all the files in the current directory\n lcd - change local directory\n get {filename} - downloads the filename from the current path\n cat {filename} - prints the contents of filename\n hexdump {filename} - hexdumps the contents of filename\n exit - terminates the server process (and this session)\n\n')

    def do_lcd(self, line):
        if False:
            for i in range(10):
                print('nop')
        if line == '':
            print(os.getcwd())
        else:
            os.chdir(line)
            print(os.getcwd())

    def do_cd(self, line):
        if False:
            while True:
                i = 10
        p = line.replace('/', '\\')
        oldpwd = self.pwd
        newPath = ntpath.normpath(ntpath.join(self.pwd, p))
        if newPath == self.pwd:
            return
        common = ntpath.commonprefix([newPath, oldpwd])
        if common == oldpwd:
            res = self.findPathName(ntpath.normpath(p))
        else:
            res = self.findPathName(newPath)
        if res is None:
            logging.error('Directory not found')
            self.pwd = oldpwd
            return
        if res.isDirectory() == 0:
            logging.error('Not a directory!')
            self.pwd = oldpwd
            return
        else:
            self.currentINode = res
            self.do_ls('', False)
            self.pwd = ntpath.join(self.pwd, p)
            self.pwd = ntpath.normpath(self.pwd)
            self.prompt = self.pwd + '>'

    def findPathName(self, pathName):
        if False:
            for i in range(10):
                print('nop')
        if pathName == '\\':
            return self.rootINode
        tmpINode = self.currentINode
        parts = pathName.split('\\')
        for part in parts:
            if part == '':
                tmpINode = self.rootINode
            else:
                res = tmpINode.findFirst(part)
                if res is None:
                    return res
                else:
                    tmpINode = self.volume.getINode(res.getINodeNumber())
        return tmpINode

    def do_pwd(self, line):
        if False:
            i = 10
            return i + 15
        print(self.pwd)

    def do_ls(self, line, display=True):
        if False:
            for i in range(10):
                print('nop')
        entries = self.currentINode.walk()
        self.completion = []
        for entry in entries:
            inode = INODE(self.volume)
            inode.FileAttributes = entry['FileAttributes']
            inode.FileSize = entry['DataSize']
            inode.LastDataChangeTime = datetime.fromtimestamp(getUnixTime(entry['LastDataChangeTime']))
            inode.FileName = entry['FileName'].decode('utf-16le')
            if display is True:
                inode.displayName()
            self.completion.append((inode.FileName, inode.isDirectory()))

    def complete_cd(self, text, line, begidx, endidx):
        if False:
            return 10
        return self.complete_get(text, line, begidx, endidx, include=2)

    def complete_cat(self, text, line, begidx, endidx):
        if False:
            return 10
        return self.complete_get(text, line, begidx, endidx)

    def complete_hexdump(self, text, line, begidx, endidx):
        if False:
            i = 10
            return i + 15
        return self.complete_get(text, line, begidx, endidx)

    def complete_get(self, text, line, begidx, endidx, include=1):
        if False:
            print('Hello World!')
        items = []
        if include == 1:
            mask = 0
        else:
            mask = FILE_ATTR_I30_INDEX_PRESENT
        for i in self.completion:
            if i[1] == mask:
                items.append(i[0])
        if text:
            return [item for item in items if item.upper().startswith(text.upper())]
        else:
            return items

    def do_hexdump(self, line):
        if False:
            for i in range(10):
                print('nop')
        return self.do_cat(line, command=hexdump)

    def do_cat(self, line, command=sys.stdout.write):
        if False:
            for i in range(10):
                print('nop')
        pathName = line.replace('/', '\\')
        pathName = ntpath.normpath(ntpath.join(self.pwd, pathName))
        res = self.findPathName(pathName)
        if res is None:
            logging.error('Not found!')
            return
        if res.isDirectory() > 0:
            logging.error("It's a directory!")
            return
        if res.isCompressed() or res.isEncrypted() or res.isSparse():
            logging.error('Cannot handle compressed/encrypted/sparse files! :(')
            return
        stream = res.getStream(None)
        chunks = 4096 * 10
        written = 0
        for i in range(stream.getDataSize() // chunks):
            buf = stream.read(i * chunks, chunks)
            written += len(buf)
            command(buf)
        if stream.getDataSize() % chunks:
            buf = stream.read(written, stream.getDataSize() % chunks)
            command(buf.decode('latin-1'))
        logging.info('%d bytes read' % stream.getDataSize())

    def do_get(self, line):
        if False:
            for i in range(10):
                print('nop')
        pathName = line.replace('/', '\\')
        pathName = ntpath.normpath(ntpath.join(self.pwd, pathName))
        fh = open(ntpath.basename(pathName), 'wb')
        self.do_cat(line, command=fh.write)
        fh.close()

def main():
    if False:
        i = 10
        return i + 15
    print(version.BANNER)
    logger.init()
    parser = argparse.ArgumentParser(add_help=True, description='NTFS explorer (read-only)')
    parser.add_argument('volume', action='store', help='NTFS volume to open (e.g. \\\\.\\C: or /dev/disk1s1)')
    parser.add_argument('-extract', action='store', help='extracts pathname (e.g. \\windows\\system32\\config\\sam)')
    parser.add_argument('-debug', action='store_true', help='Turn DEBUG output ON')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    options = parser.parse_args()
    if options.debug is True:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(version.getInstallationPath())
    else:
        logging.getLogger().setLevel(logging.INFO)
    shell = MiniShell(options.volume)
    if options.extract is not None:
        shell.onecmd('get %s' % options.extract)
    else:
        shell.cmdloop()
if __name__ == '__main__':
    main()
    sys.exit(1)