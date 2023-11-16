"""A tty implementation for xonsh"""
import os
import sys

def tty(args, stdin, stdout, stderr):
    if False:
        for i in range(10):
            print('nop')
    'A tty command for xonsh.'
    if '--help' in args:
        print(TTY_HELP, file=stdout)
        return 0
    silent = False
    for i in ('-s', '--silent', '--quiet'):
        if i in args:
            silent = True
            args.remove(i)
    if len(args) > 0:
        if not silent:
            for i in args:
                print(f'tty: Invalid option: {i}', file=stderr)
            print("Try 'tty --help' for more information", file=stderr)
        return 2
    try:
        fd = stdin.fileno()
    except:
        fd = sys.stdin.fileno()
    if not os.isatty(fd):
        if not silent:
            print('not a tty', file=stdout)
        return 1
    if not silent:
        try:
            print(os.ttyname(fd), file=stdout)
        except:
            return 3
    return 0
TTY_HELP = 'Usage: tty [OPTION]...\nPrint the file name of the terminal connected to standard input.\n\n  -s, --silent, --quiet   print nothing, only return an exit status\n      --help     display this help and exit\n\nThis version of tty was written in Python for the xonsh project: http://xon.sh\nBased on tty from GNU coreutils: http://www.gnu.org/software/coreutils/'