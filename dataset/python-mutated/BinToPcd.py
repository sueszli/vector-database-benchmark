"""
BinToPcd
"""
from __future__ import print_function
import sys
import argparse
import re
import xdrlib
import io
import struct
import math
__prog__ = 'BinToPcd'
__copyright__ = 'Copyright (c) 2016 - 2018, Intel Corporation. All rights reserved.'
__description__ = 'Convert one or more binary files to a VOID* PCD value or DSC file VOID* PCD statement.\n'
if __name__ == '__main__':

    def ValidateUnsignedInteger(Argument):
        if False:
            while True:
                i = 10
        try:
            Value = int(Argument, 0)
        except:
            Message = '{Argument} is not a valid integer value.'.format(Argument=Argument)
            raise argparse.ArgumentTypeError(Message)
        if Value < 0:
            Message = '{Argument} is a negative value.'.format(Argument=Argument)
            raise argparse.ArgumentTypeError(Message)
        return Value

    def ValidatePcdName(Argument):
        if False:
            print('Hello World!')
        if re.split('[a-zA-Z\\_][a-zA-Z0-9\\_]*\\.[a-zA-Z\\_][a-zA-Z0-9\\_]*', Argument) != ['', '']:
            Message = '{Argument} is not in the form <PcdTokenSpaceGuidCName>.<PcdCName>'.format(Argument=Argument)
            raise argparse.ArgumentTypeError(Message)
        return Argument

    def ValidateGuidName(Argument):
        if False:
            print('Hello World!')
        if re.split('[a-zA-Z\\_][a-zA-Z0-9\\_]*', Argument) != ['', '']:
            Message = '{Argument} is not a valid GUID C name'.format(Argument=Argument)
            raise argparse.ArgumentTypeError(Message)
        return Argument

    def XdrPackBuffer(buffer):
        if False:
            for i in range(10):
                print('nop')
        packed_bytes = io.BytesIO()
        for unpacked_bytes in buffer:
            n = len(unpacked_bytes)
            packed_bytes.write(struct.pack('>L', n))
            data = unpacked_bytes[:n]
            n = math.ceil(n / 4) * 4
            data = data + (n - len(data)) * b'\x00'
            packed_bytes.write(data)
        return packed_bytes.getvalue()

    def ByteArray(Buffer, Xdr=False):
        if False:
            i = 10
            return i + 15
        if Xdr:
            Buffer = bytearray(XdrPackBuffer(Buffer))
        else:
            Buffer = bytearray(b''.join(Buffer))
        return ('{' + ', '.join(['0x{Byte:02X}'.format(Byte=Item) for Item in Buffer]) + '}', len(Buffer))
    parser = argparse.ArgumentParser(prog=__prog__, description=__description__ + __copyright__, conflict_handler='resolve')
    parser.add_argument('-i', '--input', dest='InputFile', type=argparse.FileType('rb'), action='append', required=True, help='Input binary filename.  Multiple input files are combined into a single PCD.')
    parser.add_argument('-o', '--output', dest='OutputFile', type=argparse.FileType('w'), help='Output filename for PCD value or PCD statement')
    parser.add_argument('-p', '--pcd', dest='PcdName', type=ValidatePcdName, help='Name of the PCD in the form <PcdTokenSpaceGuidCName>.<PcdCName>')
    parser.add_argument('-t', '--type', dest='PcdType', default=None, choices=['VPD', 'HII'], help='PCD statement type (HII or VPD).  Default is standard.')
    parser.add_argument('-m', '--max-size', dest='MaxSize', type=ValidateUnsignedInteger, help='Maximum size of the PCD.  Ignored with --type HII.')
    parser.add_argument('-f', '--offset', dest='Offset', type=ValidateUnsignedInteger, help='VPD offset if --type is VPD.  UEFI Variable offset if --type is HII.  Must be 8-byte aligned.')
    parser.add_argument('-n', '--variable-name', dest='VariableName', help='UEFI variable name.  Only used with --type HII.')
    parser.add_argument('-g', '--variable-guid', type=ValidateGuidName, dest='VariableGuid', help='UEFI variable GUID C name.  Only used with --type HII.')
    parser.add_argument('-x', '--xdr', dest='Xdr', action='store_true', help='Encode PCD using the Variable-Length Opaque Data format of RFC 4506 External Data Representation Standard (XDR)')
    parser.add_argument('-v', '--verbose', dest='Verbose', action='store_true', help='Increase output messages')
    parser.add_argument('-q', '--quiet', dest='Quiet', action='store_true', help='Reduce output messages')
    parser.add_argument('--debug', dest='Debug', type=int, metavar='[0-9]', choices=range(0, 10), default=0, help='Set debug level')
    args = parser.parse_args()
    Buffer = []
    for File in args.InputFile:
        try:
            Buffer.append(File.read())
            File.close()
        except:
            print('BinToPcd: error: can not read binary input file {File}'.format(File=File))
            sys.exit(1)
    (PcdValue, PcdSize) = ByteArray(Buffer, args.Xdr)
    if args.PcdName is None:
        Pcd = PcdValue
        if args.Verbose:
            print('BinToPcd: Convert binary file to PCD Value')
    elif args.PcdType is None:
        if args.MaxSize is None:
            Pcd = '  {Name}|{Value}'.format(Name=args.PcdName, Value=PcdValue)
        elif args.MaxSize < PcdSize:
            print('BinToPcd: error: argument --max-size is smaller than input file.')
            sys.exit(1)
        else:
            Pcd = '  {Name}|{Value}|VOID*|{Size}'.format(Name=args.PcdName, Value=PcdValue, Size=args.MaxSize)
        if args.Verbose:
            print('BinToPcd: Convert binary file to PCD statement compatible with PCD sections:')
            print('    [PcdsFixedAtBuild]')
            print('    [PcdsPatchableInModule]')
            print('    [PcdsDynamicDefault]')
            print('    [PcdsDynamicExDefault]')
    elif args.PcdType == 'VPD':
        if args.MaxSize is None:
            args.MaxSize = PcdSize
        if args.MaxSize < PcdSize:
            print('BinToPcd: error: argument --max-size is smaller than input file.')
            sys.exit(1)
        if args.Offset is None:
            Pcd = '  {Name}|*|{Size}|{Value}'.format(Name=args.PcdName, Size=args.MaxSize, Value=PcdValue)
        else:
            if args.Offset % 8 != 0:
                print('BinToPcd: error: argument --offset must be 8-byte aligned.')
                sys.exit(1)
            Pcd = '  {Name}|{Offset}|{Size}|{Value}'.format(Name=args.PcdName, Offset=args.Offset, Size=args.MaxSize, Value=PcdValue)
        if args.Verbose:
            print('BinToPcd: Convert binary file to PCD statement compatible with PCD sections')
            print('    [PcdsDynamicVpd]')
            print('    [PcdsDynamicExVpd]')
    elif args.PcdType == 'HII':
        if args.VariableGuid is None or args.VariableName is None:
            print('BinToPcd: error: arguments --variable-guid and --variable-name are required for --type HII.')
            sys.exit(1)
        if args.Offset is None:
            args.Offset = 0
        if args.Offset % 8 != 0:
            print('BinToPcd: error: argument --offset must be 8-byte aligned.')
            sys.exit(1)
        Pcd = '  {Name}|L"{VarName}"|{VarGuid}|{Offset}|{Value}'.format(Name=args.PcdName, VarName=args.VariableName, VarGuid=args.VariableGuid, Offset=args.Offset, Value=PcdValue)
        if args.Verbose:
            print('BinToPcd: Convert binary file to PCD statement compatible with PCD sections')
            print('    [PcdsDynamicHii]')
            print('    [PcdsDynamicExHii]')
    try:
        args.OutputFile.write(Pcd)
        args.OutputFile.close()
    except:
        print(Pcd)