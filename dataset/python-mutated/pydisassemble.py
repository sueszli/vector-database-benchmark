from __future__ import print_function
import getopt
import os
import sys
from uncompyle6.code_fns import disassemble_file
from uncompyle6.version import __version__
(program, ext) = os.path.splitext(os.path.basename(__file__))
__doc__ = '\nUsage:\n  {0} [OPTIONS]... FILE\n  {0} [--help | -h | -V | --version]\n\nDisassemble/Tokenize FILE with in the way that is done to\nassist uncompyle6 in parsing the instruction stream. For example\ninstructions with variable-length arguments like CALL_FUNCTION and\nBUILD_LIST have argument counts appended to the instruction name, and\nCOME_FROM psuedo instructions are inserted into the instruction stream.\nBit flag values encoded in an operand are expanding, EXTENDED_ARG\nvalue are folded into the following instruction operand.\n\nLike the parser, you may find this more high-level and or helpful.\nHowever if you want a true disassembler see the Standard built-in\nPython library module "dis", or pydisasm from the cross-version\nPython bytecode package "xdis".\n\nExamples:\n  {0} foo.pyc\n  {0} foo.py    # same thing as above but find the file\n  {0} foo.pyc bar.pyc  # disassemble foo.pyc and bar.pyc\n\nSee also `pydisasm\' from the `xdis\' package.\n\nOptions:\n  -V | --version     show version and stop\n  -h | --help        show this message\n\n'.format(program)
PATTERNS = ('*.pyc', '*.pyo')

def main():
    if False:
        print('Hello World!')
    Usage_short = 'usage: %s FILE...\nType -h for for full help.' % program
    if len(sys.argv) == 1:
        print('No file(s) given', file=sys.stderr)
        print(Usage_short, file=sys.stderr)
        sys.exit(1)
    try:
        (opts, files) = getopt.getopt(sys.argv[1:], 'hVU', ['help', 'version', 'uncompyle6'])
    except getopt.GetoptError as e:
        print('%s: %s' % (os.path.basename(sys.argv[0]), e), file=sys.stderr)
        sys.exit(-1)
    for (opt, val) in opts:
        if opt in ('-h', '--help'):
            print(__doc__)
            sys.exit(1)
        elif opt in ('-V', '--version'):
            print('%s %s' % (program, __version__))
            sys.exit(0)
        else:
            print(opt)
            print(Usage_short, file=sys.stderr)
            sys.exit(1)
    for file in files:
        if os.path.exists(files[0]):
            disassemble_file(file, sys.stdout)
        else:
            print("Can't read %s - skipping" % files[0], file=sys.stderr)
            pass
        pass
    return
if __name__ == '__main__':
    main()