from __future__ import division
from __future__ import print_function
from impacket import LOG
try:
    from collections import OrderedDict
except:
    try:
        from ordereddict.ordereddict import OrderedDict
    except:
        from ordereddict import OrderedDict
from impacket.structure import Structure, hexdump
from struct import unpack
from binascii import hexlify
from six import b
FILE_TYPE_DATABASE = 0
FILE_TYPE_STREAMING_FILE = 1
JET_dbstateJustCreated = 1
JET_dbstateDirtyShutdown = 2
JET_dbstateCleanShutdown = 3
JET_dbstateBeingConverted = 4
JET_dbstateForceDetach = 5
FLAGS_ROOT = 1
FLAGS_LEAF = 2
FLAGS_PARENT = 4
FLAGS_EMPTY = 8
FLAGS_SPACE_TREE = 32
FLAGS_INDEX = 64
FLAGS_LONG_VALUE = 128
FLAGS_NEW_FORMAT = 8192
FLAGS_NEW_CHECKSUM = 8192
TAG_UNKNOWN = 1
TAG_DEFUNCT = 2
TAG_COMMON = 4
DATABASE_PAGE_NUMBER = 1
CATALOG_PAGE_NUMBER = 4
CATALOG_BACKUP_PAGE_NUMBER = 24
DATABASE_FDP = 1
CATALOG_FDP = 2
CATALOG_BACKUP_FDP = 3
CATALOG_TYPE_TABLE = 1
CATALOG_TYPE_COLUMN = 2
CATALOG_TYPE_INDEX = 3
CATALOG_TYPE_LONG_VALUE = 4
CATALOG_TYPE_CALLBACK = 5
JET_coltypNil = 0
JET_coltypBit = 1
JET_coltypUnsignedByte = 2
JET_coltypShort = 3
JET_coltypLong = 4
JET_coltypCurrency = 5
JET_coltypIEEESingle = 6
JET_coltypIEEEDouble = 7
JET_coltypDateTime = 8
JET_coltypBinary = 9
JET_coltypText = 10
JET_coltypLongBinary = 11
JET_coltypLongText = 12
JET_coltypSLV = 13
JET_coltypUnsignedLong = 14
JET_coltypLongLong = 15
JET_coltypGUID = 16
JET_coltypUnsignedShort = 17
JET_coltypMax = 18
ColumnTypeToName = {JET_coltypNil: 'NULL', JET_coltypBit: 'Boolean', JET_coltypUnsignedByte: 'Signed byte', JET_coltypShort: 'Signed short', JET_coltypLong: 'Signed long', JET_coltypCurrency: 'Currency', JET_coltypIEEESingle: 'Single precision FP', JET_coltypIEEEDouble: 'Double precision FP', JET_coltypDateTime: 'DateTime', JET_coltypBinary: 'Binary', JET_coltypText: 'Text', JET_coltypLongBinary: 'Long Binary', JET_coltypLongText: 'Long Text', JET_coltypSLV: 'Obsolete', JET_coltypUnsignedLong: 'Unsigned long', JET_coltypLongLong: 'Long long', JET_coltypGUID: 'GUID', JET_coltypUnsignedShort: 'Unsigned short', JET_coltypMax: 'Max'}
ColumnTypeSize = {JET_coltypNil: None, JET_coltypBit: (1, 'B'), JET_coltypUnsignedByte: (1, 'B'), JET_coltypShort: (2, '<h'), JET_coltypLong: (4, '<l'), JET_coltypCurrency: (8, '<Q'), JET_coltypIEEESingle: (4, '<f'), JET_coltypIEEEDouble: (8, '<d'), JET_coltypDateTime: (8, '<Q'), JET_coltypBinary: None, JET_coltypText: None, JET_coltypLongBinary: None, JET_coltypLongText: None, JET_coltypSLV: None, JET_coltypUnsignedLong: (4, '<L'), JET_coltypLongLong: (8, '<Q'), JET_coltypGUID: (16, '16s'), JET_coltypUnsignedShort: (2, '<H'), JET_coltypMax: None}
TAGGED_DATA_TYPE_VARIABLE_SIZE = 1
TAGGED_DATA_TYPE_COMPRESSED = 2
TAGGED_DATA_TYPE_STORED = 4
TAGGED_DATA_TYPE_MULTI_VALUE = 8
TAGGED_DATA_TYPE_WHO_KNOWS = 10
CODEPAGE_UNICODE = 1200
CODEPAGE_ASCII = 20127
CODEPAGE_WESTERN = 1252
StringCodePages = {CODEPAGE_UNICODE: 'utf-16le', CODEPAGE_ASCII: 'ascii', CODEPAGE_WESTERN: 'cp1252'}
TABLE_CURSOR = {'TableData': b'', 'FatherDataPageNumber': 0, 'CurrentPageData': b'', 'CurrentTag': 0}

class ESENT_JET_SIGNATURE(Structure):
    structure = (('Random', '<L=0'), ('CreationTime', '<Q=0'), ('NetBiosName', '16s=b""'))

class ESENT_DB_HEADER(Structure):
    structure = (('CheckSum', '<L=0'), ('Signature', '"ïÍ«\x89'), ('Version', '<L=0'), ('FileType', '<L=0'), ('DBTime', '<Q=0'), ('DBSignature', ':', ESENT_JET_SIGNATURE), ('DBState', '<L=0'), ('ConsistentPosition', '<Q=0'), ('ConsistentTime', '<Q=0'), ('AttachTime', '<Q=0'), ('AttachPosition', '<Q=0'), ('DetachTime', '<Q=0'), ('DetachPosition', '<Q=0'), ('LogSignature', ':', ESENT_JET_SIGNATURE), ('Unknown', '<L=0'), ('PreviousBackup', '24s=b""'), ('PreviousIncBackup', '24s=b""'), ('CurrentFullBackup', '24s=b""'), ('ShadowingDisables', '<L=0'), ('LastObjectID', '<L=0'), ('WindowsMajorVersion', '<L=0'), ('WindowsMinorVersion', '<L=0'), ('WindowsBuildNumber', '<L=0'), ('WindowsServicePackNumber', '<L=0'), ('FileFormatRevision', '<L=0'), ('PageSize', '<L=0'), ('RepairCount', '<L=0'), ('RepairTime', '<Q=0'), ('Unknown2', '28s=b""'), ('ScrubTime', '<Q=0'), ('RequiredLog', '<Q=0'), ('UpgradeExchangeFormat', '<L=0'), ('UpgradeFreePages', '<L=0'), ('UpgradeSpaceMapPages', '<L=0'), ('CurrentShadowBackup', '24s=b""'), ('CreationFileFormatVersion', '<L=0'), ('CreationFileFormatRevision', '<L=0'), ('Unknown3', '16s=b""'), ('OldRepairCount', '<L=0'), ('ECCCount', '<L=0'), ('LastECCTime', '<Q=0'), ('OldECCFixSuccessCount', '<L=0'), ('ECCFixErrorCount', '<L=0'), ('LastECCFixErrorTime', '<Q=0'), ('OldECCFixErrorCount', '<L=0'), ('BadCheckSumErrorCount', '<L=0'), ('LastBadCheckSumTime', '<Q=0'), ('OldCheckSumErrorCount', '<L=0'), ('CommittedLog', '<L=0'), ('PreviousShadowCopy', '24s=b""'), ('PreviousDifferentialBackup', '24s=b""'), ('Unknown4', '40s=b""'), ('NLSMajorVersion', '<L=0'), ('NLSMinorVersion', '<L=0'), ('Unknown5', '148s=b""'), ('UnknownFlags', '<L=0'))

class ESENT_PAGE_HEADER(Structure):
    structure_2003_SP0 = (('CheckSum', '<L=0'), ('PageNumber', '<L=0'))
    structure_0x620_0x0b = (('CheckSum', '<L=0'), ('ECCCheckSum', '<L=0'))
    structure_win7 = (('CheckSum', '<Q=0'),)
    common = (('LastModificationTime', '<Q=0'), ('PreviousPageNumber', '<L=0'), ('NextPageNumber', '<L=0'), ('FatherDataPage', '<L=0'), ('AvailableDataSize', '<H=0'), ('AvailableUncommittedDataSize', '<H=0'), ('FirstAvailableDataOffset', '<H=0'), ('FirstAvailablePageTag', '<H=0'), ('PageFlags', '<L=0'))
    extended_win7 = (('ExtendedCheckSum1', '<Q=0'), ('ExtendedCheckSum2', '<Q=0'), ('ExtendedCheckSum3', '<Q=0'), ('PageNumber', '<Q=0'), ('Unknown', '<Q=0'))

    def __init__(self, version, revision, pageSize=8192, data=None):
        if False:
            print('Hello World!')
        if version < 1568 or (version == 1568 and revision < 11):
            self.structure = self.structure_2003_SP0 + self.common
        elif version == 1568 and revision < 17:
            self.structure = self.structure_0x620_0x0b + self.common
        else:
            self.structure = self.structure_win7 + self.common
            if pageSize > 8192:
                self.structure += self.extended_win7
        Structure.__init__(self, data)

class ESENT_ROOT_HEADER(Structure):
    structure = (('InitialNumberOfPages', '<L=0'), ('ParentFatherDataPage', '<L=0'), ('ExtentSpace', '<L=0'), ('SpaceTreePageNumber', '<L=0'))

class ESENT_BRANCH_HEADER(Structure):
    structure = (('CommonPageKey', ':'),)

class ESENT_BRANCH_ENTRY(Structure):
    common = (('CommonPageKeySize', '<H=0'),)
    structure = (('LocalPageKeySize', '<H=0'), ('_LocalPageKey', '_-LocalPageKey', 'self["LocalPageKeySize"]'), ('LocalPageKey', ':'), ('ChildPageNumber', '<L=0'))

    def __init__(self, flags, data=None):
        if False:
            for i in range(10):
                print('nop')
        if flags & TAG_COMMON > 0:
            self.structure = self.common + self.structure
        Structure.__init__(self, data)

class ESENT_LEAF_HEADER(Structure):
    structure = (('CommonPageKey', ':'),)

class ESENT_LEAF_ENTRY(Structure):
    common = (('CommonPageKeySize', '<H=0'),)
    structure = (('LocalPageKeySize', '<H=0'), ('_LocalPageKey', '_-LocalPageKey', 'self["LocalPageKeySize"]'), ('LocalPageKey', ':'), ('EntryData', ':'))

    def __init__(self, flags, data=None):
        if False:
            while True:
                i = 10
        if flags & TAG_COMMON > 0:
            self.structure = self.common + self.structure
        Structure.__init__(self, data)

class ESENT_SPACE_TREE_HEADER(Structure):
    structure = (('Unknown', '<Q=0'),)

class ESENT_SPACE_TREE_ENTRY(Structure):
    structure = (('PageKeySize', '<H=0'), ('LastPageNumber', '<L=0'), ('NumberOfPages', '<L=0'))

class ESENT_INDEX_ENTRY(Structure):
    structure = (('RecordPageKey', ':'),)

class ESENT_DATA_DEFINITION_HEADER(Structure):
    structure = (('LastFixedSize', '<B=0'), ('LastVariableDataType', '<B=0'), ('VariableSizeOffset', '<H=0'))

class ESENT_CATALOG_DATA_DEFINITION_ENTRY(Structure):
    fixed = (('FatherDataPageID', '<L=0'), ('Type', '<H=0'), ('Identifier', '<L=0'))
    column_stuff = (('ColumnType', '<L=0'), ('SpaceUsage', '<L=0'), ('ColumnFlags', '<L=0'), ('CodePage', '<L=0'))
    other = (('FatherDataPageNumber', '<L=0'),)
    table_stuff = (('SpaceUsage', '<L=0'),)
    index_stuff = (('SpaceUsage', '<L=0'), ('IndexFlags', '<L=0'), ('Locale', '<L=0'))
    lv_stuff = (('SpaceUsage', '<L=0'),)
    common = (('Trailing', ':'),)

    def __init__(self, data):
        if False:
            print('Hello World!')
        dataType = unpack('<H', data[4:][:2])[0]
        self.structure = self.fixed
        if dataType == CATALOG_TYPE_TABLE:
            self.structure += self.other + self.table_stuff
        elif dataType == CATALOG_TYPE_COLUMN:
            self.structure += self.column_stuff
        elif dataType == CATALOG_TYPE_INDEX:
            self.structure += self.other + self.index_stuff
        elif dataType == CATALOG_TYPE_LONG_VALUE:
            self.structure += self.other + self.lv_stuff
        elif dataType == CATALOG_TYPE_CALLBACK:
            raise Exception('CallBack types not supported!')
        else:
            LOG.error('Unknown catalog type 0x%x' % dataType)
            self.structure = ()
            Structure.__init__(self, data)
        self.structure += self.common
        Structure.__init__(self, data)

def getUnixTime(t):
    if False:
        while True:
            i = 10
    t -= 116444736000000000
    t //= 10000000
    return t

class ESENT_PAGE:

    def __init__(self, db, data=None):
        if False:
            print('Hello World!')
        self.__DBHeader = db
        self.data = data
        self.record = None
        if data is not None:
            self.record = ESENT_PAGE_HEADER(self.__DBHeader['Version'], self.__DBHeader['FileFormatRevision'], self.__DBHeader['PageSize'], data)

    def printFlags(self):
        if False:
            return 10
        flags = self.record['PageFlags']
        if flags & FLAGS_EMPTY:
            print('\tEmpty')
        if flags & FLAGS_INDEX:
            print('\tIndex')
        if flags & FLAGS_LEAF:
            print('\tLeaf')
        else:
            print('\tBranch')
        if flags & FLAGS_LONG_VALUE:
            print('\tLong Value')
        if flags & FLAGS_NEW_CHECKSUM:
            print('\tNew Checksum')
        if flags & FLAGS_NEW_FORMAT:
            print('\tNew Format')
        if flags & FLAGS_PARENT:
            print('\tParent')
        if flags & FLAGS_ROOT:
            print('\tRoot')
        if flags & FLAGS_SPACE_TREE:
            print('\tSpace Tree')

    def dump(self):
        if False:
            return 10
        baseOffset = len(self.record)
        self.record.dump()
        tags = self.data[-4 * self.record['FirstAvailablePageTag']:]
        print('FLAGS: ')
        self.printFlags()
        print()
        for i in range(self.record['FirstAvailablePageTag']):
            tag = tags[-4:]
            if self.__DBHeader['Version'] == 1568 and self.__DBHeader['FileFormatRevision'] > 11 and (self.__DBHeader['PageSize'] > 8192):
                valueSize = unpack('<H', tag[:2])[0] & 32767
                valueOffset = unpack('<H', tag[2:])[0] & 32767
                hexdump(self.data[baseOffset + valueOffset:][:6])
                pageFlags = ord(self.data[baseOffset + valueOffset:][1]) >> 5
            else:
                valueSize = unpack('<H', tag[:2])[0] & 8191
                pageFlags = (unpack('<H', tag[2:])[0] & 57344) >> 13
                valueOffset = unpack('<H', tag[2:])[0] & 8191
            print('TAG %-8d offset:0x%-6x flags:0x%-4x valueSize:0x%x' % (i, valueOffset, pageFlags, valueSize))
            tags = tags[:-4]
        if self.record['PageFlags'] & FLAGS_ROOT > 0:
            rootHeader = ESENT_ROOT_HEADER(self.getTag(0)[1])
            rootHeader.dump()
        elif self.record['PageFlags'] & FLAGS_LEAF == 0:
            (flags, data) = self.getTag(0)
            branchHeader = ESENT_BRANCH_HEADER(data)
            branchHeader.dump()
        else:
            (flags, data) = self.getTag(0)
            if self.record['PageFlags'] & FLAGS_SPACE_TREE > 0:
                spaceTreeHeader = ESENT_SPACE_TREE_HEADER(data)
                spaceTreeHeader.dump()
            else:
                leafHeader = ESENT_LEAF_HEADER(data)
                leafHeader.dump()
        for tagNum in range(1, self.record['FirstAvailablePageTag']):
            (flags, data) = self.getTag(tagNum)
            if self.record['PageFlags'] & FLAGS_LEAF == 0:
                branchEntry = ESENT_BRANCH_ENTRY(flags, data)
                branchEntry.dump()
            elif self.record['PageFlags'] & FLAGS_LEAF > 0:
                if self.record['PageFlags'] & FLAGS_SPACE_TREE > 0:
                    spaceTreeEntry = ESENT_SPACE_TREE_ENTRY(data)
                elif self.record['PageFlags'] & FLAGS_INDEX > 0:
                    indexEntry = ESENT_INDEX_ENTRY(data)
                elif self.record['PageFlags'] & FLAGS_LONG_VALUE > 0:
                    raise Exception('Long value still not supported')
                else:
                    leafEntry = ESENT_LEAF_ENTRY(flags, data)
                    dataDefinitionHeader = ESENT_DATA_DEFINITION_HEADER(leafEntry['EntryData'])
                    dataDefinitionHeader.dump()
                    catalogEntry = ESENT_CATALOG_DATA_DEFINITION_ENTRY(leafEntry['EntryData'][len(dataDefinitionHeader):])
                    catalogEntry.dump()
                    hexdump(leafEntry['EntryData'])

    def getTag(self, tagNum):
        if False:
            while True:
                i = 10
        if self.record['FirstAvailablePageTag'] < tagNum:
            raise Exception('Trying to grab an unknown tag 0x%x' % tagNum)
        tags = self.data[-4 * self.record['FirstAvailablePageTag']:]
        baseOffset = len(self.record)
        for i in range(tagNum):
            tags = tags[:-4]
        tag = tags[-4:]
        if self.__DBHeader['Version'] == 1568 and self.__DBHeader['FileFormatRevision'] >= 17 and (self.__DBHeader['PageSize'] > 8192):
            valueSize = unpack('<H', tag[:2])[0] & 32767
            valueOffset = unpack('<H', tag[2:])[0] & 32767
            tmpData = bytearray(self.data[baseOffset + valueOffset:][:valueSize])
            pageFlags = tmpData[1] >> 5
            tmpData[1] = tmpData[1:2][0] & 31
            tmpData = bytes(tmpData)
            tagData = tmpData
        else:
            valueSize = unpack('<H', tag[:2])[0] & 8191
            pageFlags = (unpack('<H', tag[2:])[0] & 57344) >> 13
            valueOffset = unpack('<H', tag[2:])[0] & 8191
            tagData = self.data[baseOffset + valueOffset:][:valueSize]
        return (pageFlags, tagData)

class ESENT_DB:

    def __init__(self, fileName, pageSize=8192, isRemote=False):
        if False:
            print('Hello World!')
        self.__fileName = fileName
        self.__pageSize = pageSize
        self.__DB = None
        self.__DBHeader = None
        self.__totalPages = None
        self.__tables = OrderedDict()
        self.__currentTable = None
        self.__isRemote = isRemote
        self.mountDB()

    def mountDB(self):
        if False:
            i = 10
            return i + 15
        LOG.debug('Mounting DB...')
        if self.__isRemote is True:
            self.__DB = self.__fileName
            self.__DB.open()
        else:
            self.__DB = open(self.__fileName, 'rb')
        mainHeader = self.getPage(-1)
        self.__DBHeader = ESENT_DB_HEADER(mainHeader)
        self.__pageSize = self.__DBHeader['PageSize']
        self.__DB.seek(0, 2)
        self.__totalPages = self.__DB.tell() // self.__pageSize - 2
        LOG.debug('Database Version:0x%x, Revision:0x%x' % (self.__DBHeader['Version'], self.__DBHeader['FileFormatRevision']))
        LOG.debug('Page Size: %d' % self.__pageSize)
        LOG.debug('Total Pages in file: %d' % self.__totalPages)
        self.parseCatalog(CATALOG_PAGE_NUMBER)

    def printCatalog(self):
        if False:
            i = 10
            return i + 15
        indent = '    '
        print('Database version: 0x%x, 0x%x' % (self.__DBHeader['Version'], self.__DBHeader['FileFormatRevision']))
        print('Page size: %d ' % self.__pageSize)
        print('Number of pages: %d' % self.__totalPages)
        print()
        print('Catalog for %s' % self.__fileName)
        for table in list(self.__tables.keys()):
            print('[%s]' % table.decode('utf8'))
            print('%sColumns ' % indent)
            for column in list(self.__tables[table]['Columns'].keys()):
                record = self.__tables[table]['Columns'][column]['Record']
                print('%s%-5d%-30s%s' % (indent * 2, record['Identifier'], column.decode('utf-8'), ColumnTypeToName[record['ColumnType']]))
            print('%sIndexes' % indent)
            for index in list(self.__tables[table]['Indexes'].keys()):
                print('%s%s' % (indent * 2, index.decode('utf-8')))
            print('')

    def __addItem(self, entry):
        if False:
            return 10
        dataDefinitionHeader = ESENT_DATA_DEFINITION_HEADER(entry['EntryData'])
        catalogEntry = ESENT_CATALOG_DATA_DEFINITION_ENTRY(entry['EntryData'][len(dataDefinitionHeader):])
        itemName = self.__parseItemName(entry)
        if catalogEntry['Type'] == CATALOG_TYPE_TABLE:
            self.__tables[itemName] = OrderedDict()
            self.__tables[itemName]['TableEntry'] = entry
            self.__tables[itemName]['Columns'] = OrderedDict()
            self.__tables[itemName]['Indexes'] = OrderedDict()
            self.__tables[itemName]['LongValues'] = OrderedDict()
            self.__currentTable = itemName
        elif catalogEntry['Type'] == CATALOG_TYPE_COLUMN:
            self.__tables[self.__currentTable]['Columns'][itemName] = entry
            self.__tables[self.__currentTable]['Columns'][itemName]['Header'] = dataDefinitionHeader
            self.__tables[self.__currentTable]['Columns'][itemName]['Record'] = catalogEntry
        elif catalogEntry['Type'] == CATALOG_TYPE_INDEX:
            self.__tables[self.__currentTable]['Indexes'][itemName] = entry
        elif catalogEntry['Type'] == CATALOG_TYPE_LONG_VALUE:
            self.__addLongValue(entry)
        else:
            raise Exception('Unknown type 0x%x' % catalogEntry['Type'])

    def __parseItemName(self, entry):
        if False:
            for i in range(10):
                print('nop')
        dataDefinitionHeader = ESENT_DATA_DEFINITION_HEADER(entry['EntryData'])
        if dataDefinitionHeader['LastVariableDataType'] > 127:
            numEntries = dataDefinitionHeader['LastVariableDataType'] - 127
        else:
            numEntries = dataDefinitionHeader['LastVariableDataType']
        itemLen = unpack('<H', entry['EntryData'][dataDefinitionHeader['VariableSizeOffset']:][:2])[0]
        itemName = entry['EntryData'][dataDefinitionHeader['VariableSizeOffset']:][2 * numEntries:][:itemLen]
        return itemName

    def __addLongValue(self, entry):
        if False:
            print('Hello World!')
        dataDefinitionHeader = ESENT_DATA_DEFINITION_HEADER(entry['EntryData'])
        lvLen = unpack('<H', entry['EntryData'][dataDefinitionHeader['VariableSizeOffset']:][:2])[0]
        lvName = entry['EntryData'][dataDefinitionHeader['VariableSizeOffset']:][7:][:lvLen]
        self.__tables[self.__currentTable]['LongValues'][lvName] = entry

    def parsePage(self, page):
        if False:
            while True:
                i = 10
        for tagNum in range(1, page.record['FirstAvailablePageTag']):
            (flags, data) = page.getTag(tagNum)
            if page.record['PageFlags'] & FLAGS_LEAF > 0:
                if page.record['PageFlags'] & FLAGS_SPACE_TREE > 0:
                    pass
                elif page.record['PageFlags'] & FLAGS_INDEX > 0:
                    pass
                elif page.record['PageFlags'] & FLAGS_LONG_VALUE > 0:
                    pass
                else:
                    leafEntry = ESENT_LEAF_ENTRY(flags, data)
                    self.__addItem(leafEntry)

    def parseCatalog(self, pageNum):
        if False:
            for i in range(10):
                print('nop')
        page = self.getPage(pageNum)
        self.parsePage(page)
        for i in range(1, page.record['FirstAvailablePageTag']):
            (flags, data) = page.getTag(i)
            if page.record['PageFlags'] & FLAGS_LEAF == 0:
                branchEntry = ESENT_BRANCH_ENTRY(flags, data)
                self.parseCatalog(branchEntry['ChildPageNumber'])

    def readHeader(self):
        if False:
            for i in range(10):
                print('nop')
        LOG.debug('Reading Boot Sector for %s' % self.__volumeName)

    def getPage(self, pageNum):
        if False:
            while True:
                i = 10
        LOG.debug('Trying to fetch page %d (0x%x)' % (pageNum, (pageNum + 1) * self.__pageSize))
        self.__DB.seek((pageNum + 1) * self.__pageSize, 0)
        data = self.__DB.read(self.__pageSize)
        while len(data) < self.__pageSize:
            remaining = self.__pageSize - len(data)
            data += self.__DB.read(remaining)
        if pageNum <= 0:
            return data
        else:
            return ESENT_PAGE(self.__DBHeader, data)

    def close(self):
        if False:
            while True:
                i = 10
        self.__DB.close()

    def openTable(self, tableName):
        if False:
            i = 10
            return i + 15
        if isinstance(tableName, bytes) is not True:
            tableName = b(tableName)
        if tableName in self.__tables:
            entry = self.__tables[tableName]['TableEntry']
            dataDefinitionHeader = ESENT_DATA_DEFINITION_HEADER(entry['EntryData'])
            catalogEntry = ESENT_CATALOG_DATA_DEFINITION_ENTRY(entry['EntryData'][len(dataDefinitionHeader):])
            pageNum = catalogEntry['FatherDataPageNumber']
            done = False
            while done is False:
                page = self.getPage(pageNum)
                if page.record['FirstAvailablePageTag'] <= 1:
                    done = True
                for i in range(1, page.record['FirstAvailablePageTag']):
                    (flags, data) = page.getTag(i)
                    if page.record['PageFlags'] & FLAGS_LEAF == 0:
                        branchEntry = ESENT_BRANCH_ENTRY(flags, data)
                        pageNum = branchEntry['ChildPageNumber']
                        break
                    else:
                        done = True
                        break
            cursor = TABLE_CURSOR
            cursor['TableData'] = self.__tables[tableName]
            cursor['FatherDataPageNumber'] = catalogEntry['FatherDataPageNumber']
            cursor['CurrentPageData'] = page
            cursor['CurrentTag'] = 0
            return cursor
        else:
            return None

    def __getNextTag(self, cursor):
        if False:
            i = 10
            return i + 15
        page = cursor['CurrentPageData']
        if cursor['CurrentTag'] >= page.record['FirstAvailablePageTag']:
            return None
        (flags, data) = page.getTag(cursor['CurrentTag'])
        if page.record['PageFlags'] & FLAGS_LEAF > 0:
            if page.record['PageFlags'] & FLAGS_SPACE_TREE > 0:
                raise Exception('FLAGS_SPACE_TREE > 0')
            elif page.record['PageFlags'] & FLAGS_INDEX > 0:
                raise Exception('FLAGS_INDEX > 0')
            elif page.record['PageFlags'] & FLAGS_LONG_VALUE > 0:
                raise Exception('FLAGS_LONG_VALUE > 0')
            else:
                leafEntry = ESENT_LEAF_ENTRY(flags, data)
                return leafEntry
        return None

    def getNextRow(self, cursor, filter_tables=None):
        if False:
            return 10
        cursor['CurrentTag'] += 1
        tag = self.__getNextTag(cursor)
        if tag is None:
            page = cursor['CurrentPageData']
            if page.record['NextPageNumber'] == 0:
                return None
            else:
                cursor['CurrentPageData'] = self.getPage(page.record['NextPageNumber'])
                cursor['CurrentTag'] = 0
                return self.getNextRow(cursor, filter_tables=filter_tables)
        else:
            return self.__tagToRecord(cursor, tag['EntryData'], filter_tables=filter_tables)

    def __tagToRecord(self, cursor, tag, filter_tables=None):
        if False:
            print('Hello World!')
        record = OrderedDict()
        taggedItems = OrderedDict()
        taggedItemsParsed = False
        dataDefinitionHeader = ESENT_DATA_DEFINITION_HEADER(tag)
        variableDataBytesProcessed = (dataDefinitionHeader['LastVariableDataType'] - 127) * 2
        prevItemLen = 0
        tagLen = len(tag)
        fixedSizeOffset = len(dataDefinitionHeader)
        variableSizeOffset = dataDefinitionHeader['VariableSizeOffset']
        columns = cursor['TableData']['Columns']
        for column in list(columns.keys()):
            if filter_tables is not None:
                if column not in filter_tables:
                    continue
            columnRecord = columns[column]['Record']
            if columnRecord['Identifier'] <= dataDefinitionHeader['LastFixedSize']:
                record[column] = tag[fixedSizeOffset:][:columnRecord['SpaceUsage']]
                fixedSizeOffset += columnRecord['SpaceUsage']
            elif 127 < columnRecord['Identifier'] <= dataDefinitionHeader['LastVariableDataType']:
                index = columnRecord['Identifier'] - 127 - 1
                itemLen = unpack('<H', tag[variableSizeOffset + index * 2:][:2])[0]
                if itemLen & 32768:
                    itemLen = prevItemLen
                    record[column] = None
                else:
                    itemValue = tag[variableSizeOffset + variableDataBytesProcessed:][:itemLen - prevItemLen]
                    record[column] = itemValue
                variableDataBytesProcessed += itemLen - prevItemLen
                prevItemLen = itemLen
            elif columnRecord['Identifier'] > 255:
                if taggedItemsParsed is False and variableDataBytesProcessed + variableSizeOffset < tagLen:
                    index = variableDataBytesProcessed + variableSizeOffset
                    endOfVS = self.__pageSize
                    firstOffsetTag = (unpack('<H', tag[index + 2:][:2])[0] & 16383) + variableDataBytesProcessed + variableSizeOffset
                    while True:
                        taggedIdentifier = unpack('<H', tag[index:][:2])[0]
                        index += 2
                        taggedOffset = unpack('<H', tag[index:][:2])[0] & 16383
                        if self.__DBHeader['Version'] == 1568 and self.__DBHeader['FileFormatRevision'] >= 17 and (self.__DBHeader['PageSize'] > 8192):
                            flagsPresent = 1
                        else:
                            flagsPresent = unpack('<H', tag[index:][:2])[0] & 16384
                        index += 2
                        if taggedOffset < endOfVS:
                            endOfVS = taggedOffset
                        taggedItems[taggedIdentifier] = (taggedOffset, tagLen, flagsPresent)
                        if index >= firstOffsetTag:
                            break
                    prevKey = list(taggedItems.keys())[0]
                    for i in range(1, len(taggedItems)):
                        (offset0, length, flags) = taggedItems[prevKey]
                        (offset, _, _) = list(taggedItems.items())[i][1]
                        taggedItems[prevKey] = (offset0, offset - offset0, flags)
                        prevKey = list(taggedItems.keys())[i]
                    taggedItemsParsed = True
                if columnRecord['Identifier'] in taggedItems:
                    offsetItem = variableDataBytesProcessed + variableSizeOffset + taggedItems[columnRecord['Identifier']][0]
                    itemSize = taggedItems[columnRecord['Identifier']][1]
                    if taggedItems[columnRecord['Identifier']][2] > 0:
                        itemFlag = ord(tag[offsetItem:offsetItem + 1])
                        offsetItem += 1
                        itemSize -= 1
                    else:
                        itemFlag = 0
                    if itemFlag & TAGGED_DATA_TYPE_COMPRESSED:
                        LOG.error('Unsupported tag column: %s, flag:0x%x' % (column, itemFlag))
                        record[column] = None
                    elif itemFlag & TAGGED_DATA_TYPE_MULTI_VALUE:
                        LOG.debug('Multivalue detected in column %s, returning raw results' % column)
                        record[column] = (hexlify(tag[offsetItem:][:itemSize]),)
                    else:
                        record[column] = tag[offsetItem:][:itemSize]
                else:
                    record[column] = None
            else:
                record[column] = None
            if type(record[column]) is tuple:
                record[column] = record[column][0]
            elif columnRecord['ColumnType'] == JET_coltypText or columnRecord['ColumnType'] == JET_coltypLongText:
                if record[column] is not None:
                    if columnRecord['CodePage'] not in StringCodePages:
                        raise Exception('Unknown codepage 0x%x' % columnRecord['CodePage'])
                    stringDecoder = StringCodePages[columnRecord['CodePage']]
                    try:
                        record[column] = record[column].decode(stringDecoder)
                    except Exception:
                        LOG.debug('Exception:', exc_info=True)
                        LOG.debug('Fixing Record[%r][%d]: %r' % (column, columnRecord['ColumnType'], record[column]))
                        record[column] = record[column].decode(stringDecoder, 'replace')
                        pass
            else:
                unpackData = ColumnTypeSize[columnRecord['ColumnType']]
                if record[column] is not None:
                    if unpackData is None:
                        record[column] = hexlify(record[column])
                    else:
                        unpackStr = unpackData[1]
                        record[column] = unpack(unpackStr, record[column])[0]
        return record