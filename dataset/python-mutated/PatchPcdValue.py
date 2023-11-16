import Common.LongFilePathOs as os
from Common.LongFilePathSupport import OpenLongFilePath as open
import sys
from optparse import OptionParser
from optparse import make_option
from Common.BuildToolError import *
import Common.EdkLogger as EdkLogger
from Common.BuildVersion import gBUILD_VERSION
import array
from Common.DataType import *
__version_number__ = '0.10' + ' ' + gBUILD_VERSION
__version__ = '%prog Version ' + __version_number__
__copyright__ = 'Copyright (c) 2010 - 2018, Intel Corporation. All rights reserved.'

def PatchBinaryFile(FileName, ValueOffset, TypeName, ValueString, MaxSize=0):
    if False:
        print('Hello World!')
    FileHandle = open(FileName, 'rb')
    FileHandle.seek(0, 2)
    FileLength = FileHandle.tell()
    FileHandle.close()
    TypeName = TypeName.upper()
    ValueLength = 0
    if TypeName == 'BOOLEAN':
        ValueLength = 1
    elif TypeName == TAB_UINT8:
        ValueLength = 1
    elif TypeName == TAB_UINT16:
        ValueLength = 2
    elif TypeName == TAB_UINT32:
        ValueLength = 4
    elif TypeName == TAB_UINT64:
        ValueLength = 8
    elif TypeName == TAB_VOID:
        if MaxSize == 0:
            return (OPTION_MISSING, 'PcdMaxSize is not specified for VOID* type PCD.')
        ValueLength = int(MaxSize)
    else:
        return (PARAMETER_INVALID, 'PCD type %s is not valid.' % CommandOptions.PcdTypeName)
    if ValueOffset + ValueLength > FileLength:
        return (PARAMETER_INVALID, 'PcdOffset + PcdMaxSize(DataType) is larger than the input file size.')
    FileHandle = open(FileName, 'rb')
    ByteArray = array.array('B')
    ByteArray.fromfile(FileHandle, FileLength)
    FileHandle.close()
    OrigByteList = ByteArray.tolist()
    ByteList = ByteArray.tolist()
    for Index in range(ValueLength):
        ByteList[ValueOffset + Index] = 0
    SavedStr = ValueString
    ValueString = ValueString.upper()
    ValueNumber = 0
    if TypeName == 'BOOLEAN':
        try:
            if ValueString == 'TRUE':
                ValueNumber = 1
            elif ValueString == 'FALSE':
                ValueNumber = 0
            ValueNumber = int(ValueString, 0)
            if ValueNumber != 0:
                ValueNumber = 1
        except:
            return (PARAMETER_INVALID, 'PCD Value %s is not valid dec or hex string.' % ValueString)
        ByteList[ValueOffset] = ValueNumber
    elif TypeName in TAB_PCD_CLEAN_NUMERIC_TYPES:
        try:
            ValueNumber = int(ValueString, 0)
        except:
            return (PARAMETER_INVALID, 'PCD Value %s is not valid dec or hex string.' % ValueString)
        for Index in range(ValueLength):
            ByteList[ValueOffset + Index] = ValueNumber % 256
            ValueNumber = ValueNumber // 256
    elif TypeName == TAB_VOID:
        ValueString = SavedStr
        if ValueString.startswith('L"'):
            Index = 0
            for ByteString in ValueString[2:-1]:
                if Index + 2 >= ValueLength:
                    break
                ByteList[ValueOffset + Index] = ord(ByteString)
                Index = Index + 2
        elif ValueString.startswith('{') and ValueString.endswith('}'):
            ValueList = ValueString[1:len(ValueString) - 1].split(',')
            Index = 0
            try:
                for ByteString in ValueList:
                    ByteString = ByteString.strip()
                    if ByteString.upper().startswith('0X'):
                        ByteValue = int(ByteString, 16)
                    else:
                        ByteValue = int(ByteString)
                    ByteList[ValueOffset + Index] = ByteValue % 256
                    Index = Index + 1
                    if Index >= ValueLength:
                        break
            except:
                return (PARAMETER_INVALID, 'PCD Value %s is not valid dec or hex string array.' % ValueString)
        else:
            Index = 0
            for ByteString in ValueString[1:-1]:
                if Index + 1 >= ValueLength:
                    break
                ByteList[ValueOffset + Index] = ord(ByteString)
                Index = Index + 1
    if ByteList != OrigByteList:
        ByteArray = array.array('B')
        ByteArray.fromlist(ByteList)
        FileHandle = open(FileName, 'wb')
        ByteArray.tofile(FileHandle)
        FileHandle.close()
    return (0, 'Patch Value into File %s successfully.' % FileName)

def Options():
    if False:
        i = 10
        return i + 15
    OptionList = [make_option('-f', '--offset', dest='PcdOffset', action='store', type='int', help='Start offset to the image is used to store PCD value.'), make_option('-u', '--value', dest='PcdValue', action='store', help='PCD value will be updated into the image.'), make_option('-t', '--type', dest='PcdTypeName', action='store', help='The name of PCD data type may be one of VOID*,BOOLEAN, UINT8, UINT16, UINT32, UINT64.'), make_option('-s', '--maxsize', dest='PcdMaxSize', action='store', type='int', help='Max size of data buffer is taken by PCD value.It must be set when PCD type is VOID*.'), make_option('-v', '--verbose', dest='LogLevel', action='store_const', const=EdkLogger.VERBOSE, help='Run verbosely'), make_option('-d', '--debug', dest='LogLevel', type='int', help='Run with debug information'), make_option('-q', '--quiet', dest='LogLevel', action='store_const', const=EdkLogger.QUIET, help='Run quietly'), make_option('-?', action='help', help='show this help message and exit')]
    UsageString = '%prog -f Offset -u Value -t Type [-s MaxSize] <input_file>'
    Parser = OptionParser(description=__copyright__, version=__version__, option_list=OptionList, usage=UsageString)
    Parser.set_defaults(LogLevel=EdkLogger.INFO)
    (Options, Args) = Parser.parse_args()
    if len(Args) == 0:
        EdkLogger.error('PatchPcdValue', PARAMETER_INVALID, ExtraData=Parser.get_usage())
    InputFile = Args[len(Args) - 1]
    return (Options, InputFile)

def Main():
    if False:
        print('Hello World!')
    try:
        EdkLogger.Initialize()
        (CommandOptions, InputFile) = Options()
        if CommandOptions.LogLevel < EdkLogger.DEBUG_9:
            EdkLogger.SetLevel(CommandOptions.LogLevel + 1)
        else:
            EdkLogger.SetLevel(CommandOptions.LogLevel)
        if not os.path.exists(InputFile):
            EdkLogger.error('PatchPcdValue', FILE_NOT_FOUND, ExtraData=InputFile)
            return 1
        if CommandOptions.PcdOffset is None or CommandOptions.PcdValue is None or CommandOptions.PcdTypeName is None:
            EdkLogger.error('PatchPcdValue', OPTION_MISSING, ExtraData='PcdOffset or PcdValue of PcdTypeName is not specified.')
            return 1
        if CommandOptions.PcdTypeName.upper() not in TAB_PCD_NUMERIC_TYPES_VOID:
            EdkLogger.error('PatchPcdValue', PARAMETER_INVALID, ExtraData='PCD type %s is not valid.' % CommandOptions.PcdTypeName)
            return 1
        if CommandOptions.PcdTypeName.upper() == TAB_VOID and CommandOptions.PcdMaxSize is None:
            EdkLogger.error('PatchPcdValue', OPTION_MISSING, ExtraData='PcdMaxSize is not specified for VOID* type PCD.')
            return 1
        (ReturnValue, ErrorInfo) = PatchBinaryFile(InputFile, CommandOptions.PcdOffset, CommandOptions.PcdTypeName, CommandOptions.PcdValue, CommandOptions.PcdMaxSize)
        if ReturnValue != 0:
            EdkLogger.error('PatchPcdValue', ReturnValue, ExtraData=ErrorInfo)
            return 1
        return 0
    except:
        return 1
if __name__ == '__main__':
    r = Main()
    sys.exit(r)