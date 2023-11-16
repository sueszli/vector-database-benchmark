import Common.LongFilePathOs as os
import sys
import re
from io import BytesIO
import codecs
from optparse import OptionParser
from optparse import make_option
from Common.BuildToolError import *
from Common.Misc import *
from Common.DataType import *
from Common.BuildVersion import gBUILD_VERSION
import Common.EdkLogger as EdkLogger
from Common.LongFilePathSupport import OpenLongFilePath as open
__version_number__ = '0.10' + ' ' + gBUILD_VERSION
__version__ = '%prog Version ' + __version_number__
__copyright__ = 'Copyright (c) 2007-2018, Intel Corporation. All rights reserved.'
gLineControlDirective = re.compile('^\\s*#(?:line)?\\s+([0-9]+)\\s+"*([^"]*)"')
gTypedefPattern = re.compile('^\\s*typedef\\s+struct(\\s+\\w+)?\\s*[{]*$', re.MULTILINE)
gPragmaPattern = re.compile('^\\s*#pragma\\s+pack', re.MULTILINE)
gTypedef_SinglePattern = re.compile('^\\s*typedef', re.MULTILINE)
gTypedef_MulPattern = re.compile('^\\s*(typedef)?\\s+(struct|union)(\\s+\\w+)?\\s*[{]*$', re.MULTILINE)
gHexNumberPattern = re.compile('(?<=[^a-zA-Z0-9_])(0[xX])([0-9a-fA-F]+)(U(?=$|[^a-zA-Z0-9_]))?')
gDecNumberPattern = re.compile('(?<=[^a-zA-Z0-9_])([0-9]+)U(?=$|[^a-zA-Z0-9_])')
gLongNumberPattern = re.compile('(?<=[^a-zA-Z0-9_])(0[xX][0-9a-fA-F]+|[0-9]+)U?LL(?=$|[^a-zA-Z0-9_])')
gAslIncludePattern = re.compile('^(\\s*)[iI]nclude\\s*\\("?([^"\\(\\)]+)"\\)', re.MULTILINE)
gAslCIncludePattern = re.compile('^(\\s*)#include\\s*[<"]\\s*([-\\\\/\\w.]+)\\s*([>"])', re.MULTILINE)
gIncludePattern = re.compile('^[ \\t]*[%]?[ \\t]*include(?:[ \\t]*(?:\\\\(?:\\r\\n|\\r|\\n))*[ \\t]*)*(?:\\(?[\\"<]?[ \\t]*)([-\\w.\\\\/() \\t]+)(?:[ \\t]*[\\">]?\\)?)', re.MULTILINE | re.UNICODE | re.IGNORECASE)
gIncludedAslFile = []

def TrimPreprocessedFile(Source, Target, ConvertHex, TrimLong):
    if False:
        i = 10
        return i + 15
    CreateDirectory(os.path.dirname(Target))
    try:
        with open(Source, 'r') as File:
            Lines = File.readlines()
    except IOError:
        EdkLogger.error('Trim', FILE_OPEN_FAILURE, ExtraData=Source)
    except:
        EdkLogger.error('Trim', AUTOGEN_ERROR, 'TrimPreprocessedFile: Error while processing file', File=Source)
    PreprocessedFile = ''
    InjectedFile = ''
    LineIndexOfOriginalFile = None
    NewLines = []
    LineControlDirectiveFound = False
    for Index in range(len(Lines)):
        Line = Lines[Index]
        MatchList = gLineControlDirective.findall(Line)
        if MatchList != []:
            MatchList = MatchList[0]
            if len(MatchList) == 2:
                LineNumber = int(MatchList[0], 0)
                InjectedFile = MatchList[1]
                InjectedFile = os.path.normpath(InjectedFile)
                InjectedFile = os.path.normcase(InjectedFile)
                if PreprocessedFile == '':
                    PreprocessedFile = InjectedFile
            LineControlDirectiveFound = True
            continue
        elif PreprocessedFile == '' or InjectedFile != PreprocessedFile:
            continue
        if LineIndexOfOriginalFile is None:
            LineIndexOfOriginalFile = Index
            EdkLogger.verbose('Found original file content starting from line %d' % (LineIndexOfOriginalFile + 1))
        if TrimLong:
            Line = gLongNumberPattern.sub('\\1', Line)
        if ConvertHex:
            Line = gHexNumberPattern.sub('0\\2h', Line)
        else:
            Line = gHexNumberPattern.sub('\\1\\2', Line)
        Line = gDecNumberPattern.sub('\\1', Line)
        if LineNumber is not None:
            EdkLogger.verbose('Got line directive: line=%d' % LineNumber)
            if LineNumber <= len(NewLines):
                NewLines[LineNumber - 1] = Line
            else:
                if LineNumber > len(NewLines) + 1:
                    for LineIndex in range(len(NewLines), LineNumber - 1):
                        NewLines.append(TAB_LINE_BREAK)
                NewLines.append(Line)
            LineNumber = None
            EdkLogger.verbose('Now we have lines: %d' % len(NewLines))
        else:
            NewLines.append(Line)
    if not LineControlDirectiveFound and NewLines == []:
        MulPatternFlag = False
        SinglePatternFlag = False
        Brace = 0
        for Index in range(len(Lines)):
            Line = Lines[Index]
            if MulPatternFlag == False and gTypedef_MulPattern.search(Line) is None:
                if SinglePatternFlag == False and gTypedef_SinglePattern.search(Line) is None:
                    if gPragmaPattern.search(Line) is None:
                        NewLines.append(Line)
                    continue
                elif SinglePatternFlag == False:
                    SinglePatternFlag = True
                if Line.find(';') >= 0:
                    SinglePatternFlag = False
            elif MulPatternFlag == False:
                MulPatternFlag = True
            if Line.find('{') >= 0:
                Brace += 1
            elif Line.find('}') >= 0:
                Brace -= 1
            if Brace == 0 and Line.find(';') >= 0:
                MulPatternFlag = False
    try:
        with open(Target, 'w') as File:
            File.writelines(NewLines)
    except:
        EdkLogger.error('Trim', FILE_OPEN_FAILURE, ExtraData=Target)

def TrimPreprocessedVfr(Source, Target):
    if False:
        print('Hello World!')
    CreateDirectory(os.path.dirname(Target))
    try:
        with open(Source, 'r') as File:
            Lines = File.readlines()
    except:
        EdkLogger.error('Trim', FILE_OPEN_FAILURE, ExtraData=Source)
    FoundTypedef = False
    Brace = 0
    TypedefStart = 0
    TypedefEnd = 0
    for Index in range(len(Lines)):
        Line = Lines[Index]
        if Line.strip() == 'formset':
            break
        if FoundTypedef == False and (Line.find('#line') == 0 or Line.find('# ') == 0):
            Lines[Index] = '\n'
            continue
        if FoundTypedef == False and gTypedefPattern.search(Line) is None:
            if gPragmaPattern.search(Line) is None:
                Lines[Index] = '\n'
            continue
        elif FoundTypedef == False:
            FoundTypedef = True
            TypedefStart = Index
        if Line.find('{') >= 0:
            Brace += 1
        elif Line.find('}') >= 0:
            Brace -= 1
        if Brace == 0 and Line.find(';') >= 0:
            FoundTypedef = False
            TypedefEnd = Index
            if Line.strip('} ;\r\n') in [TAB_GUID, 'EFI_PLABEL', 'PAL_CALL_RETURN']:
                for i in range(TypedefStart, TypedefEnd + 1):
                    Lines[i] = '\n'
    try:
        with open(Target, 'w') as File:
            File.writelines(Lines)
    except:
        EdkLogger.error('Trim', FILE_OPEN_FAILURE, ExtraData=Target)

def DoInclude(Source, Indent='', IncludePathList=[], LocalSearchPath=None, IncludeFileList=None, filetype=None):
    if False:
        while True:
            i = 10
    NewFileContent = []
    if IncludeFileList is None:
        IncludeFileList = []
    try:
        if LocalSearchPath:
            SearchPathList = [LocalSearchPath] + IncludePathList
        else:
            SearchPathList = IncludePathList
        for IncludePath in SearchPathList:
            IncludeFile = os.path.join(IncludePath, Source)
            if os.path.isfile(IncludeFile):
                try:
                    with open(IncludeFile, 'r') as File:
                        F = File.readlines()
                except:
                    with codecs.open(IncludeFile, 'r', encoding='utf-8') as File:
                        F = File.readlines()
                break
        else:
            EdkLogger.error('Trim', 'Failed to find include file %s' % Source)
            return []
    except:
        EdkLogger.error('Trim', FILE_OPEN_FAILURE, ExtraData=Source)
        return []
    IncludeFile = os.path.abspath(os.path.normpath(IncludeFile))
    if IncludeFile in gIncludedAslFile:
        EdkLogger.warn('Trim', 'Circular include', ExtraData='%s -> %s' % (' -> '.join(gIncludedAslFile), IncludeFile))
        return []
    gIncludedAslFile.append(IncludeFile)
    IncludeFileList.append(IncludeFile.strip())
    for Line in F:
        LocalSearchPath = None
        if filetype == 'ASL':
            Result = gAslIncludePattern.findall(Line)
            if len(Result) == 0:
                Result = gAslCIncludePattern.findall(Line)
                if len(Result) == 0 or os.path.splitext(Result[0][1])[1].lower() not in ['.asl', '.asi']:
                    NewFileContent.append('%s%s' % (Indent, Line))
                    continue
                if Result[0][2] == '"':
                    LocalSearchPath = os.path.dirname(IncludeFile)
            CurrentIndent = Indent + Result[0][0]
            IncludedFile = Result[0][1]
            NewFileContent.extend(DoInclude(IncludedFile, CurrentIndent, IncludePathList, LocalSearchPath, IncludeFileList, filetype))
            NewFileContent.append('\n')
        elif filetype == 'ASM':
            Result = gIncludePattern.findall(Line)
            if len(Result) == 0:
                NewFileContent.append('%s%s' % (Indent, Line))
                continue
            IncludedFile = Result[0]
            IncludedFile = IncludedFile.strip()
            IncludedFile = os.path.normpath(IncludedFile)
            NewFileContent.extend(DoInclude(IncludedFile, '', IncludePathList, LocalSearchPath, IncludeFileList, filetype))
            NewFileContent.append('\n')
    gIncludedAslFile.pop()
    return NewFileContent

def TrimAslFile(Source, Target, IncludePathFile, AslDeps=False):
    if False:
        for i in range(10):
            print('nop')
    CreateDirectory(os.path.dirname(Target))
    SourceDir = os.path.dirname(Source)
    if SourceDir == '':
        SourceDir = '.'
    IncludePathList = [SourceDir]
    if IncludePathFile:
        try:
            LineNum = 0
            with open(IncludePathFile, 'r') as File:
                FileLines = File.readlines()
            for Line in FileLines:
                LineNum += 1
                if Line.startswith('/I') or Line.startswith('-I'):
                    IncludePathList.append(Line[2:].strip())
                else:
                    EdkLogger.warn('Trim', 'Invalid include line in include list file.', IncludePathFile, LineNum)
        except:
            EdkLogger.error('Trim', FILE_OPEN_FAILURE, ExtraData=IncludePathFile)
    AslIncludes = []
    Lines = DoInclude(Source, '', IncludePathList, IncludeFileList=AslIncludes, filetype='ASL')
    AslIncludes = [item for item in AslIncludes if item != Source]
    SaveFileOnChange(os.path.join(os.path.dirname(Target), os.path.basename(Source)) + '.trim.deps', ' \\\n'.join([Source + ':'] + AslIncludes), False)
    Lines.insert(0, '#undef MIN\n#undef MAX\n')
    try:
        with open(Target, 'w') as File:
            File.writelines(Lines)
    except:
        EdkLogger.error('Trim', FILE_OPEN_FAILURE, ExtraData=Target)

def TrimAsmFile(Source, Target, IncludePathFile):
    if False:
        for i in range(10):
            print('nop')
    CreateDirectory(os.path.dirname(Target))
    SourceDir = os.path.dirname(Source)
    if SourceDir == '':
        SourceDir = '.'
    IncludePathList = [SourceDir]
    if IncludePathFile:
        try:
            LineNum = 0
            with open(IncludePathFile, 'r') as File:
                FileLines = File.readlines()
            for Line in FileLines:
                LineNum += 1
                if Line.startswith('/I') or Line.startswith('-I'):
                    IncludePathList.append(Line[2:].strip())
                else:
                    EdkLogger.warn('Trim', 'Invalid include line in include list file.', IncludePathFile, LineNum)
        except:
            EdkLogger.error('Trim', FILE_OPEN_FAILURE, ExtraData=IncludePathFile)
    AsmIncludes = []
    Lines = DoInclude(Source, '', IncludePathList, IncludeFileList=AsmIncludes, filetype='ASM')
    AsmIncludes = [item for item in AsmIncludes if item != Source]
    if AsmIncludes:
        SaveFileOnChange(os.path.join(os.path.dirname(Target), os.path.basename(Source)) + '.trim.deps', ' \\\n'.join([Source + ':'] + AsmIncludes), False)
    try:
        with open(Target, 'w') as File:
            File.writelines(Lines)
    except:
        EdkLogger.error('Trim', FILE_OPEN_FAILURE, ExtraData=Target)

def GenerateVfrBinSec(ModuleName, DebugDir, OutputFile):
    if False:
        i = 10
        return i + 15
    VfrNameList = []
    if os.path.isdir(DebugDir):
        for (CurrentDir, Dirs, Files) in os.walk(DebugDir):
            for FileName in Files:
                (Name, Ext) = os.path.splitext(FileName)
                if Ext == '.c' and Name != 'AutoGen':
                    VfrNameList.append(Name + 'Bin')
    VfrNameList.append(ModuleName + 'Strings')
    EfiFileName = os.path.join(DebugDir, ModuleName + '.efi')
    MapFileName = os.path.join(DebugDir, ModuleName + '.map')
    VfrUniOffsetList = GetVariableOffset(MapFileName, EfiFileName, VfrNameList)
    if not VfrUniOffsetList:
        return
    try:
        fInputfile = open(OutputFile, 'wb+')
    except:
        EdkLogger.error('Trim', FILE_OPEN_FAILURE, 'File open failed for %s' % OutputFile, None)
    fStringIO = BytesIO()
    for Item in VfrUniOffsetList:
        if Item[0].find('Strings') != -1:
            UniGuid = b'\xe0\xc5\x13\x89\xf63\x86M\x9b\xf1C\xef\x89\xfc\x06f'
            fStringIO.write(UniGuid)
            UniValue = pack('Q', int(Item[1], 16))
            fStringIO.write(UniValue)
        else:
            VfrGuid = b'\xb4|\xbc\xd0Gj_I\xaa\x11q\x07F\xda\x06\xa2'
            fStringIO.write(VfrGuid)
            type(Item[1])
            VfrValue = pack('Q', int(Item[1], 16))
            fStringIO.write(VfrValue)
    try:
        fInputfile.write(fStringIO.getvalue())
    except:
        EdkLogger.error('Trim', FILE_WRITE_FAILURE, 'Write data to file %s failed, please check whether the file been locked or using by other applications.' % OutputFile, None)
    fStringIO.close()
    fInputfile.close()

def Options():
    if False:
        i = 10
        return i + 15
    OptionList = [make_option('-s', '--source-code', dest='FileType', const='SourceCode', action='store_const', help='The input file is preprocessed source code, including C or assembly code'), make_option('-r', '--vfr-file', dest='FileType', const='Vfr', action='store_const', help='The input file is preprocessed VFR file'), make_option('--Vfr-Uni-Offset', dest='FileType', const='VfrOffsetBin', action='store_const', help='The input file is EFI image'), make_option('--asl-deps', dest='AslDeps', const='True', action='store_const', help='Generate Asl dependent files.'), make_option('-a', '--asl-file', dest='FileType', const='Asl', action='store_const', help='The input file is ASL file'), make_option('--asm-file', dest='FileType', const='Asm', action='store_const', help='The input file is asm file'), make_option('-c', '--convert-hex', dest='ConvertHex', action='store_true', help='Convert standard hex format (0xabcd) to MASM format (abcdh)'), make_option('-l', '--trim-long', dest='TrimLong', action='store_true', help='Remove postfix of long number'), make_option('-i', '--include-path-file', dest='IncludePathFile', help='The input file is include path list to search for ASL include file'), make_option('-o', '--output', dest='OutputFile', help='File to store the trimmed content'), make_option('--ModuleName', dest='ModuleName', help="The module's BASE_NAME"), make_option('--DebugDir', dest='DebugDir', help='Debug Output directory to store the output files'), make_option('-v', '--verbose', dest='LogLevel', action='store_const', const=EdkLogger.VERBOSE, help='Run verbosely'), make_option('-d', '--debug', dest='LogLevel', type='int', help='Run with debug information'), make_option('-q', '--quiet', dest='LogLevel', action='store_const', const=EdkLogger.QUIET, help='Run quietly'), make_option('-?', action='help', help='show this help message and exit')]
    UsageString = '%prog [-s|-r|-a|--Vfr-Uni-Offset] [-c] [-v|-d <debug_level>|-q] [-i <include_path_file>] [-o <output_file>] [--ModuleName <ModuleName>] [--DebugDir <DebugDir>] [<input_file>]'
    Parser = OptionParser(description=__copyright__, version=__version__, option_list=OptionList, usage=UsageString)
    Parser.set_defaults(FileType='Vfr')
    Parser.set_defaults(ConvertHex=False)
    Parser.set_defaults(LogLevel=EdkLogger.INFO)
    (Options, Args) = Parser.parse_args()
    if Options.FileType == 'VfrOffsetBin':
        if len(Args) == 0:
            return (Options, '')
        elif len(Args) > 1:
            EdkLogger.error('Trim', OPTION_NOT_SUPPORTED, ExtraData=Parser.get_usage())
    if len(Args) == 0:
        EdkLogger.error('Trim', OPTION_MISSING, ExtraData=Parser.get_usage())
    if len(Args) > 1:
        EdkLogger.error('Trim', OPTION_NOT_SUPPORTED, ExtraData=Parser.get_usage())
    InputFile = Args[0]
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
    except FatalError as X:
        return 1
    try:
        if CommandOptions.FileType == 'Vfr':
            if CommandOptions.OutputFile is None:
                CommandOptions.OutputFile = os.path.splitext(InputFile)[0] + '.iii'
            TrimPreprocessedVfr(InputFile, CommandOptions.OutputFile)
        elif CommandOptions.FileType == 'Asl':
            if CommandOptions.OutputFile is None:
                CommandOptions.OutputFile = os.path.splitext(InputFile)[0] + '.iii'
            TrimAslFile(InputFile, CommandOptions.OutputFile, CommandOptions.IncludePathFile, CommandOptions.AslDeps)
        elif CommandOptions.FileType == 'VfrOffsetBin':
            GenerateVfrBinSec(CommandOptions.ModuleName, CommandOptions.DebugDir, CommandOptions.OutputFile)
        elif CommandOptions.FileType == 'Asm':
            TrimAsmFile(InputFile, CommandOptions.OutputFile, CommandOptions.IncludePathFile)
        else:
            if CommandOptions.OutputFile is None:
                CommandOptions.OutputFile = os.path.splitext(InputFile)[0] + '.iii'
            TrimPreprocessedFile(InputFile, CommandOptions.OutputFile, CommandOptions.ConvertHex, CommandOptions.TrimLong)
    except FatalError as X:
        import platform
        import traceback
        if CommandOptions is not None and CommandOptions.LogLevel <= EdkLogger.DEBUG_9:
            EdkLogger.quiet('(Python %s on %s) ' % (platform.python_version(), sys.platform) + traceback.format_exc())
        return 1
    except:
        import traceback
        import platform
        EdkLogger.error('\nTrim', CODE_ERROR, 'Unknown fatal error when trimming [%s]' % InputFile, ExtraData='\n(Please send email to %s for help, attaching following call stack trace!)\n' % MSG_EDKII_MAIL_ADDR, RaiseError=False)
        EdkLogger.quiet('(Python %s on %s) ' % (platform.python_version(), sys.platform) + traceback.format_exc())
        return 1
    return 0
if __name__ == '__main__':
    r = Main()
    if r < 0 or r > 127:
        r = 1
    sys.exit(r)