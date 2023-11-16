"""fuse_gtest_files.py v0.2.0
Fuses Google Test source code into a .h file and a .cc file.

SYNOPSIS
       fuse_gtest_files.py [GTEST_ROOT_DIR] OUTPUT_DIR

       Scans GTEST_ROOT_DIR for Google Test source code, and generates
       two files: OUTPUT_DIR/gtest/gtest.h and OUTPUT_DIR/gtest/gtest-all.cc.
       Then you can build your tests by adding OUTPUT_DIR to the include
       search path and linking with OUTPUT_DIR/gtest/gtest-all.cc.  These
       two files contain everything you need to use Google Test.  Hence
       you can "install" Google Test by copying them to wherever you want.

       GTEST_ROOT_DIR can be omitted and defaults to the parent
       directory of the directory holding this script.

EXAMPLES
       ./fuse_gtest_files.py fused_gtest
       ./fuse_gtest_files.py path/to/unpacked/gtest fused_gtest

This tool is experimental.  In particular, it assumes that there is no
conditional inclusion of Google Test headers.  Please report any
problems to googletestframework@googlegroups.com.  You can read
https://github.com/google/googletest/blob/master/googletest/docs/advanced.md for
more information.
"""
__author__ = 'wan@google.com (Zhanyong Wan)'
import os
import re
try:
    from sets import Set as set
except ImportError:
    pass
import sys
DEFAULT_GTEST_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
INCLUDE_GTEST_FILE_REGEX = re.compile('^\\s*#\\s*include\\s*"(gtest/.+)"')
INCLUDE_SRC_FILE_REGEX = re.compile('^\\s*#\\s*include\\s*"(src/.+)"')
GTEST_H_SEED = 'include/gtest/gtest.h'
GTEST_SPI_H_SEED = 'include/gtest/gtest-spi.h'
GTEST_ALL_CC_SEED = 'src/gtest-all.cc'
GTEST_H_OUTPUT = 'gtest/gtest.h'
GTEST_ALL_CC_OUTPUT = 'gtest/gtest-all.cc'

def VerifyFileExists(directory, relative_path):
    if False:
        print('Hello World!')
    'Verifies that the given file exists; aborts on failure.\n\n  relative_path is the file path relative to the given directory.\n  '
    if not os.path.isfile(os.path.join(directory, relative_path)):
        print('ERROR: Cannot find %s in directory %s.' % (relative_path, directory))
        print('Please either specify a valid project root directory or omit it on the command line.')
        sys.exit(1)

def ValidateGTestRootDir(gtest_root):
    if False:
        return 10
    'Makes sure gtest_root points to a valid gtest root directory.\n\n  The function aborts the program on failure.\n  '
    VerifyFileExists(gtest_root, GTEST_H_SEED)
    VerifyFileExists(gtest_root, GTEST_ALL_CC_SEED)

def VerifyOutputFile(output_dir, relative_path):
    if False:
        while True:
            i = 10
    'Verifies that the given output file path is valid.\n\n  relative_path is relative to the output_dir directory.\n  '
    output_file = os.path.join(output_dir, relative_path)
    if os.path.exists(output_file):
        print('%s already exists in directory %s - overwrite it? (y/N) ' % (relative_path, output_dir))
        answer = sys.stdin.readline().strip()
        if answer not in ['y', 'Y']:
            print('ABORTED.')
            sys.exit(1)
    parent_directory = os.path.dirname(output_file)
    if not os.path.isdir(parent_directory):
        os.makedirs(parent_directory)

def ValidateOutputDir(output_dir):
    if False:
        i = 10
        return i + 15
    'Makes sure output_dir points to a valid output directory.\n\n  The function aborts the program on failure.\n  '
    VerifyOutputFile(output_dir, GTEST_H_OUTPUT)
    VerifyOutputFile(output_dir, GTEST_ALL_CC_OUTPUT)

def FuseGTestH(gtest_root, output_dir):
    if False:
        return 10
    'Scans folder gtest_root to generate gtest/gtest.h in output_dir.'
    output_file = open(os.path.join(output_dir, GTEST_H_OUTPUT), 'w')
    processed_files = set()

    def ProcessFile(gtest_header_path):
        if False:
            while True:
                i = 10
        'Processes the given gtest header file.'
        if gtest_header_path in processed_files:
            return
        processed_files.add(gtest_header_path)
        for line in open(os.path.join(gtest_root, gtest_header_path), 'r'):
            m = INCLUDE_GTEST_FILE_REGEX.match(line)
            if m:
                ProcessFile('include/' + m.group(1))
            else:
                output_file.write(line)
    ProcessFile(GTEST_H_SEED)
    output_file.close()

def FuseGTestAllCcToFile(gtest_root, output_file):
    if False:
        for i in range(10):
            print('nop')
    'Scans folder gtest_root to generate gtest/gtest-all.cc in output_file.'
    processed_files = set()

    def ProcessFile(gtest_source_file):
        if False:
            while True:
                i = 10
        'Processes the given gtest source file.'
        if gtest_source_file in processed_files:
            return
        processed_files.add(gtest_source_file)
        for line in open(os.path.join(gtest_root, gtest_source_file), 'r'):
            m = INCLUDE_GTEST_FILE_REGEX.match(line)
            if m:
                if 'include/' + m.group(1) == GTEST_SPI_H_SEED:
                    ProcessFile(GTEST_SPI_H_SEED)
                elif not GTEST_H_SEED in processed_files:
                    processed_files.add(GTEST_H_SEED)
                    output_file.write('#include "%s"\n' % (GTEST_H_OUTPUT,))
            else:
                m = INCLUDE_SRC_FILE_REGEX.match(line)
                if m:
                    ProcessFile(m.group(1))
                else:
                    output_file.write(line)
    ProcessFile(GTEST_ALL_CC_SEED)

def FuseGTestAllCc(gtest_root, output_dir):
    if False:
        while True:
            i = 10
    'Scans folder gtest_root to generate gtest/gtest-all.cc in output_dir.'
    output_file = open(os.path.join(output_dir, GTEST_ALL_CC_OUTPUT), 'w')
    FuseGTestAllCcToFile(gtest_root, output_file)
    output_file.close()

def FuseGTest(gtest_root, output_dir):
    if False:
        for i in range(10):
            print('nop')
    'Fuses gtest.h and gtest-all.cc.'
    ValidateGTestRootDir(gtest_root)
    ValidateOutputDir(output_dir)
    FuseGTestH(gtest_root, output_dir)
    FuseGTestAllCc(gtest_root, output_dir)

def main():
    if False:
        for i in range(10):
            print('nop')
    argc = len(sys.argv)
    if argc == 2:
        FuseGTest(DEFAULT_GTEST_ROOT_DIR, sys.argv[1])
    elif argc == 3:
        FuseGTest(sys.argv[1], sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
if __name__ == '__main__':
    main()