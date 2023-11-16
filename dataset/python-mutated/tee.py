"""A tee implementation for xonsh."""

def tee(args, stdin, stdout, stderr):
    if False:
        while True:
            i = 10
    'A tee command for xonsh.'
    mode = 'w'
    if '-a' in args:
        args.remove('-a')
        mode = 'a'
    if '--append' in args:
        args.remove('--append')
        mode = 'a'
    if '--help' in args:
        print(TEE_HELP, file=stdout)
        return 0
    if stdin is None:
        msg = 'tee was not piped stdin, must have input stream to read from.'
        print(msg, file=stderr)
        return 1
    errors = False
    files = []
    for i in args:
        if i == '-':
            files.append(stdout)
        else:
            try:
                files.append(open(i, mode))
            except:
                print(f'tee: failed to open {i}', file=stderr)
                errors = True
    files.append(stdout)
    while True:
        r = stdin.read(1024)
        if r == '':
            break
        for i in files:
            i.write(r)
    for i in files:
        if i != stdout:
            i.close()
    return int(errors)
TEE_HELP = 'This version of tee was written in Python for the xonsh project: http://xon.sh\nBased on tee from GNU coreutils: http://www.gnu.org/software/coreutils/\n\nUsage: tee [OPTION]... [FILE]...\nCopy standard input to each FILE, and also to standard output.\n\n  -a, --append              append to the given FILEs, do not overwrite\n      --help     display this help and exit\n\nIf a FILE is -, copy again to standard output.'