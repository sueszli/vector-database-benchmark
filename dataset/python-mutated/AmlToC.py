import argparse
import Common.EdkLogger as EdkLogger
from Common.BuildToolError import *
import sys
import os
__description__ = '\nConvert an AML file to a .c file containing the AML bytecode stored in a C\narray. By default, Tables\\Dsdt.aml will generate Tables\\Dsdt.c.\nTables\\Dsdt.c will contain a C array named "dsdt_aml_code" that contains\nthe AML bytecode.\n'

def ParseArgs():
    if False:
        for i in range(10):
            print('nop')
    Parser = argparse.ArgumentParser(description=__description__)
    Parser.add_argument(dest='InputFile', help='Path to an input AML file to generate a .c file from.')
    Parser.add_argument('-o', '--out-dir', dest='OutDir', help="Output directory where the .c file will be generated. Default is the input file's directory.")
    Args = Parser.parse_args()
    SplitInputName = ''
    if not os.path.exists(Args.InputFile):
        EdkLogger.error(__file__, FILE_OPEN_FAILURE, ExtraData=Args.InputFile)
        return None
    else:
        with open(Args.InputFile, 'rb') as fIn:
            Signature = str(fIn.read(4))
            if 'DSDT' not in Signature and 'SSDT' not in Signature:
                EdkLogger.info('Invalid file type. File does not have a valid DSDT or SSDT signature: {}'.format(Args.InputFile))
                return None
    SplitInputName = os.path.splitext(Args.InputFile)
    BaseName = os.path.basename(SplitInputName[0])
    if not Args.OutDir:
        Args.OutputFile = os.path.join(os.path.dirname(Args.InputFile), BaseName + '.c')
    else:
        if not os.path.exists(Args.OutDir):
            os.mkdir(Args.OutDir)
        Args.OutputFile = os.path.join(Args.OutDir, BaseName + '.c')
    Args.BaseName = BaseName
    return Args

def AmlToC(InputFile, OutputFile, BaseName):
    if False:
        return 10
    ArrayName = BaseName.lower() + '_aml_code'
    FileHeader = '\n// This file has been generated from:\n//   -Python script: {}\n//   -Input AML file: {}\n\n'
    with open(InputFile, 'rb') as fIn, open(OutputFile, 'w') as fOut:
        fOut.write(FileHeader.format(os.path.abspath(InputFile), os.path.abspath(__file__)))
        fOut.write('unsigned char {}[] = {{\n  '.format(ArrayName))
        cnt = 0
        byte = fIn.read(1)
        while len(byte) != 0:
            fOut.write('0x{0:02X}, '.format(ord(byte)))
            cnt += 1
            if cnt % 8 == 0:
                fOut.write('\n  ')
            byte = fIn.read(1)
        fOut.write('\n};\n')

def Main():
    if False:
        i = 10
        return i + 15
    EdkLogger.Initialize()
    try:
        CommandArguments = ParseArgs()
        if not CommandArguments:
            return 1
        AmlToC(CommandArguments.InputFile, CommandArguments.OutputFile, CommandArguments.BaseName)
    except Exception as e:
        print(e)
        return 1
    return 0
if __name__ == '__main__':
    r = Main()
    if r < 0 or r > 127:
        r = 1
    sys.exit(r)