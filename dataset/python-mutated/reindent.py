"""reindent [-d][-r][-v] [ path ... ]

-d (--dryrun)   Dry run.   Analyze, but don't make any changes to, files.
-r (--recurse)  Recurse.   Search for all .py files in subdirectories too.
-n (--nobackup) No backup. Does not make a ".bak" file before reindenting.
-v (--verbose)  Verbose.   Print informative msgs; else no output.
   (--newline)  Newline.   Specify the newline character to use (CRLF, LF).
                           Default is the same as the original file.
-h (--help)     Help.      Print this usage information and exit.

Change Python (.py) files to use 4-space indents and no hard tab characters.
Also trim excess spaces and tabs from ends of lines, and remove empty lines
at the end of files.  Also ensure the last line ends with a newline.

If no paths are given on the command line, reindent operates as a filter,
reading a single source file from standard input and writing the transformed
source to standard output.  In this case, the -d, -r and -v flags are
ignored.

You can pass one or more file and/or directory paths.  When a directory
path, all .py files within the directory will be examined, and, if the -r
option is given, likewise recursively for subdirectories.

If output is not to standard output, reindent overwrites files in place,
renaming the originals with a .bak extension.  If it finds nothing to
change, the file is left alone.  If reindent does change a file, the changed
file is a fixed-point for future runs (i.e., running reindent on the
resulting .py file won't change it again).

The hard part of reindenting is figuring out what to do with comment
lines.  So long as the input files get a clean bill of health from
tabnanny.py, reindent should do a good job.

The backup file is a copy of the one that is being reindented. The ".bak"
file is generated with shutil.copy(), but some corner cases regarding
user/group and permissions could leave the backup file more readable than
you'd prefer. You can always use the --nobackup option to prevent this.
"""
__version__ = '1'
import tokenize
import os
import shutil
import sys
verbose = False
recurse = False
dryrun = False
makebackup = True
spec_newline = None

def usage(msg=None):
    if False:
        print('Hello World!')
    if msg is None:
        msg = __doc__
    print(msg, file=sys.stderr)

def errprint(*args):
    if False:
        print('Hello World!')
    sys.stderr.write(' '.join((str(arg) for arg in args)))
    sys.stderr.write('\n')

def main():
    if False:
        i = 10
        return i + 15
    import getopt
    global verbose, recurse, dryrun, makebackup, spec_newline
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'drnvh', ['dryrun', 'recurse', 'nobackup', 'verbose', 'newline=', 'help'])
    except getopt.error as msg:
        usage(msg)
        return
    for (o, a) in opts:
        if o in ('-d', '--dryrun'):
            dryrun = True
        elif o in ('-r', '--recurse'):
            recurse = True
        elif o in ('-n', '--nobackup'):
            makebackup = False
        elif o in ('-v', '--verbose'):
            verbose = True
        elif o in ('--newline',):
            if not a.upper() in ('CRLF', 'LF'):
                usage()
                return
            spec_newline = dict(CRLF='\r\n', LF='\n')[a.upper()]
        elif o in ('-h', '--help'):
            usage()
            return
    if not args:
        r = Reindenter(sys.stdin)
        r.run()
        r.write(sys.stdout)
        return
    for arg in args:
        check(arg)

def check(file):
    if False:
        while True:
            i = 10
    if os.path.isdir(file) and (not os.path.islink(file)):
        if verbose:
            print('listing directory', file)
        names = os.listdir(file)
        for name in names:
            fullname = os.path.join(file, name)
            if recurse and os.path.isdir(fullname) and (not os.path.islink(fullname)) and (not os.path.split(fullname)[1].startswith('.')) or name.lower().endswith('.py'):
                check(fullname)
        return
    if verbose:
        print('checking', file, '...', end=' ')
    with open(file, 'rb') as f:
        try:
            (encoding, _) = tokenize.detect_encoding(f.readline)
        except SyntaxError as se:
            errprint('%s: SyntaxError: %s' % (file, str(se)))
            return
    try:
        with open(file, encoding=encoding) as f:
            r = Reindenter(f)
    except IOError as msg:
        errprint('%s: I/O Error: %s' % (file, str(msg)))
        return
    newline = spec_newline if spec_newline else r.newlines
    if isinstance(newline, tuple):
        errprint('%s: mixed newlines detected; cannot continue without --newline' % file)
        return
    if r.run():
        if verbose:
            print('changed.')
            if dryrun:
                print('But this is a dry run, so leaving it alone.')
        if not dryrun:
            bak = file + '.bak'
            if makebackup:
                shutil.copyfile(file, bak)
                if verbose:
                    print('backed up', file, 'to', bak)
            with open(file, 'w', encoding=encoding, newline=newline) as f:
                r.write(f)
            if verbose:
                print('wrote new', file)
        return True
    else:
        if verbose:
            print('unchanged.')
        return False

def _rstrip(line, JUNK='\n \t'):
    if False:
        for i in range(10):
            print('nop')
    'Return line stripped of trailing spaces, tabs, newlines.\n\n    Note that line.rstrip() instead also strips sundry control characters,\n    but at least one known Emacs user expects to keep junk like that, not\n    mentioning Barry by name or anything <wink>.\n    '
    i = len(line)
    while i > 0 and line[i - 1] in JUNK:
        i -= 1
    return line[:i]

class Reindenter:

    def __init__(self, f):
        if False:
            print('Hello World!')
        self.find_stmt = 1
        self.level = 0
        self.raw = f.readlines()
        self.lines = [_rstrip(line).expandtabs() + '\n' for line in self.raw]
        self.lines.insert(0, None)
        self.index = 1
        self.stats = []
        self.newlines = f.newlines

    def run(self):
        if False:
            print('Hello World!')
        tokens = tokenize.generate_tokens(self.getline)
        for _token in tokens:
            self.tokeneater(*_token)
        lines = self.lines
        while lines and lines[-1] == '\n':
            lines.pop()
        stats = self.stats
        stats.append((len(lines), 0))
        have2want = {}
        after = self.after = []
        i = stats[0][0]
        after.extend(lines[1:i])
        for i in range(len(stats) - 1):
            (thisstmt, thislevel) = stats[i]
            nextstmt = stats[i + 1][0]
            have = getlspace(lines[thisstmt])
            want = thislevel * 4
            if want < 0:
                if have:
                    want = have2want.get(have, -1)
                    if want < 0:
                        for j in range(i + 1, len(stats) - 1):
                            (jline, jlevel) = stats[j]
                            if jlevel >= 0:
                                if have == getlspace(lines[jline]):
                                    want = jlevel * 4
                                break
                    if want < 0:
                        for j in range(i - 1, -1, -1):
                            (jline, jlevel) = stats[j]
                            if jlevel >= 0:
                                want = have + (getlspace(after[jline - 1]) - getlspace(lines[jline]))
                                break
                    if want < 0:
                        want = have
                else:
                    want = 0
            assert want >= 0
            have2want[have] = want
            diff = want - have
            if diff == 0 or have == 0:
                after.extend(lines[thisstmt:nextstmt])
            else:
                for line in lines[thisstmt:nextstmt]:
                    if diff > 0:
                        if line == '\n':
                            after.append(line)
                        else:
                            after.append(' ' * diff + line)
                    else:
                        remove = min(getlspace(line), -diff)
                        after.append(line[remove:])
        return self.raw != self.after

    def write(self, f):
        if False:
            i = 10
            return i + 15
        f.writelines(self.after)

    def getline(self):
        if False:
            return 10
        if self.index >= len(self.lines):
            line = ''
        else:
            line = self.lines[self.index]
            self.index += 1
        return line

    def tokeneater(self, type, token, slinecol, end, line, INDENT=tokenize.INDENT, DEDENT=tokenize.DEDENT, NEWLINE=tokenize.NEWLINE, COMMENT=tokenize.COMMENT, NL=tokenize.NL):
        if False:
            print('Hello World!')
        if type == NEWLINE:
            self.find_stmt = 1
        elif type == INDENT:
            self.find_stmt = 1
            self.level += 1
        elif type == DEDENT:
            self.find_stmt = 1
            self.level -= 1
        elif type == COMMENT:
            if self.find_stmt:
                self.stats.append((slinecol[0], -1))
        elif type == NL:
            pass
        elif self.find_stmt:
            self.find_stmt = 0
            if line:
                self.stats.append((slinecol[0], self.level))

def getlspace(line):
    if False:
        return 10
    (i, n) = (0, len(line))
    while i < n and line[i] == ' ':
        i += 1
    return i
if __name__ == '__main__':
    main()