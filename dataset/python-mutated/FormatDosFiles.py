from __future__ import print_function
import argparse
import os
import os.path
import re
import sys
import copy
__prog__ = 'FormatDosFiles'
__version__ = '%s Version %s' % (__prog__, '0.10 ')
__copyright__ = 'Copyright (c) 2018-2019, Intel Corporation. All rights reserved.'
__description__ = 'Convert source files to meet the EDKII C Coding Standards Specification.\n'
DEFAULT_EXT_LIST = ['.h', '.c', '.nasm', '.nasmb', '.asm', '.S', '.inf', '.dec', '.dsc', '.fdf', '.uni', '.asl', '.aslc', '.vfr', '.idf', '.txt', '.bat', '.py']

def FormatFile(FilePath, Args):
    if False:
        print('Hello World!')
    with open(FilePath, 'rb') as Fd:
        Content = Fd.read()
        Content = re.sub(b'([^\\r])\\n', b'\\1\\r\\n', Content)
        Content = re.sub(b'^\\n', b'\\r\\n', Content, flags=re.MULTILINE)
        Content = re.sub(b'([^\\r\\n])$', b'\\1\\r\\n', Content)
        Content = re.sub(b'[ \\t]+(\\r\\n)', b'\\1', Content, flags=re.MULTILINE)
        Content = re.sub(b'\t', b'  ', Content)
        with open(FilePath, 'wb') as Fd:
            Fd.write(Content)
            if not Args.Quiet:
                print(FilePath)

def FormatFilesInDir(DirPath, ExtList, Args):
    if False:
        for i in range(10):
            print('nop')
    FileList = []
    ExcludeDir = DirPath
    for (DirPath, DirNames, FileNames) in os.walk(DirPath):
        if Args.Exclude:
            DirNames[:] = [d for d in DirNames if d not in Args.Exclude]
            FileNames[:] = [f for f in FileNames if f not in Args.Exclude]
            Continue = False
            for Path in Args.Exclude:
                Path = Path.strip('\\').strip('/')
                if not os.path.isdir(Path) and (not os.path.isfile(Path)):
                    Path = os.path.join(ExcludeDir, Path)
                if os.path.isdir(Path) and Path.endswith(DirPath):
                    DirNames[:] = []
                    Continue = True
                elif os.path.isfile(Path):
                    FilePaths = FileNames
                    for ItemPath in FilePaths:
                        FilePath = os.path.join(DirPath, ItemPath)
                        if Path.endswith(FilePath):
                            FileNames.remove(ItemPath)
            if Continue:
                continue
        for FileName in [f for f in FileNames if any((f.endswith(ext) for ext in ExtList))]:
            FileList.append(os.path.join(DirPath, FileName))
    for File in FileList:
        FormatFile(File, Args)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__prog__, description=__description__ + __copyright__, conflict_handler='resolve')
    parser.add_argument('Path', nargs='+', help='the path for files to be converted.It could be directory or file path.')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--append-extensions', dest='AppendExt', nargs='+', help='append file extensions filter to default extensions. (Example: .txt .c .h)')
    parser.add_argument('--override-extensions', dest='OverrideExt', nargs='+', help='override file extensions filter on default extensions. (Example: .txt .c .h)')
    parser.add_argument('-v', '--verbose', dest='Verbose', action='store_true', help='increase output messages')
    parser.add_argument('-q', '--quiet', dest='Quiet', action='store_true', help='reduce output messages')
    parser.add_argument('--debug', dest='Debug', type=int, metavar='[0-9]', choices=range(0, 10), default=0, help='set debug level')
    parser.add_argument('--exclude', dest='Exclude', nargs='+', help='directory name or file name which will be excluded')
    args = parser.parse_args()
    DefaultExt = copy.copy(DEFAULT_EXT_LIST)
    if args.OverrideExt is not None:
        DefaultExt = args.OverrideExt
    if args.AppendExt is not None:
        DefaultExt = list(set(DefaultExt + args.AppendExt))
    for Path in args.Path:
        if not os.path.exists(Path):
            print('not exists path: {0}'.format(Path))
            sys.exit(1)
        if os.path.isdir(Path):
            FormatFilesInDir(Path, DefaultExt, args)
        elif os.path.isfile(Path):
            FormatFile(Path, args)