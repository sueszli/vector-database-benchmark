"""A pwd implementation for xonsh."""
import os
from xonsh.built_ins import XSH

def pwd(args, stdin, stdout, stderr):
    if False:
        for i in range(10):
            print('nop')
    'A pwd implementation'
    e = XSH.env['PWD']
    if '-h' in args or '--help' in args:
        print(PWD_HELP, file=stdout)
        return 0
    if '-P' in args:
        e = os.path.realpath(e)
    print(e, file=stdout)
    return 0
PWD_HELP = 'Usage: pwd [OPTION]...\nPrint the full filename of the current working directory.\n\n  -P, --physical   avoid all symlinks\n      --help       display this help and exit\n\nThis version of pwd was written in Python for the xonsh project: http://xon.sh\nBased on pwd from GNU coreutils: http://www.gnu.org/software/coreutils/'