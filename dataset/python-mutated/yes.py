"""An implementation of yes for xonsh."""

def yes(args, stdin, stdout, stderr):
    if False:
        while True:
            i = 10
    'A yes command.'
    if '--help' in args:
        print(YES_HELP, file=stdout)
        return 0
    to_print = ['y'] if len(args) == 0 else [str(i) for i in args]
    while True:
        print(*to_print, file=stdout)
    return 0
YES_HELP = "Usage: yes [STRING]...\n  or:  yes OPTION\nRepeatedly output a line with all specified STRING(s), or 'y'.\n\n      --help     display this help and exit\n\nThis version of yes was written in Python for the xonsh project: http://xon.sh\nBased on yes from GNU coreutils: http://www.gnu.org/software/coreutils/"