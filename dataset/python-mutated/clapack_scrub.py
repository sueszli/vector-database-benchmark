import os
import re
import sys
from plex import Scanner, Str, Lexicon, Opt, Bol, State, AnyChar, TEXT, IGNORE
from plex.traditional import re as Re
try:
    from io import BytesIO as UStringIO
except ImportError:
    from io import StringIO as UStringIO

class MyScanner(Scanner):

    def __init__(self, info, name='<default>'):
        if False:
            while True:
                i = 10
        Scanner.__init__(self, self.lexicon, info, name)

    def begin(self, state_name):
        if False:
            for i in range(10):
                print('nop')
        Scanner.begin(self, state_name)

def sep_seq(sequence, sep):
    if False:
        return 10
    pat = Str(sequence[0])
    for s in sequence[1:]:
        pat += sep + Str(s)
    return pat

def runScanner(data, scanner_class, lexicon=None):
    if False:
        print('Hello World!')
    info = UStringIO(data)
    outfo = UStringIO()
    if lexicon is not None:
        scanner = scanner_class(lexicon, info)
    else:
        scanner = scanner_class(info)
    while True:
        (value, text) = scanner.read()
        if value is None:
            break
        elif value is IGNORE:
            pass
        else:
            outfo.write(value)
    return (outfo.getvalue(), scanner)

class LenSubsScanner(MyScanner):
    """Following clapack, we remove ftnlen arguments, which f2c puts after
    a char * argument to hold the length of the passed string. This is just
    a nuisance in C.
    """

    def __init__(self, info, name='<ftnlen>'):
        if False:
            i = 10
            return i + 15
        MyScanner.__init__(self, info, name)
        self.paren_count = 0

    def beginArgs(self, text):
        if False:
            for i in range(10):
                print('nop')
        if self.paren_count == 0:
            self.begin('args')
        self.paren_count += 1
        return text

    def endArgs(self, text):
        if False:
            print('Hello World!')
        self.paren_count -= 1
        if self.paren_count == 0:
            self.begin('')
        return text
    digits = Re('[0-9]+')
    iofun = Re('\\([^;]*;')
    decl = Re('\\([^)]*\\)[,;' + '\n]')
    any = Re('[.]*')
    S = Re('[ \t\n]*')
    cS = Str(',') + S
    len_ = Re('[a-z][a-z0-9]*_len')
    iofunctions = Str('s_cat', 's_copy', 's_stop', 's_cmp', 'i_len', 'do_fio', 'do_lio') + iofun
    keep_ftnlen = (Str('ilaenv_') | Str('iparmq_') | Str('s_rnge')) + Str('(')
    lexicon = Lexicon([(iofunctions, TEXT), (keep_ftnlen, beginArgs), State('args', [(Str(')'), endArgs), (Str('('), beginArgs), (AnyChar, TEXT)]), (cS + Re('[1-9][0-9]*L'), IGNORE), (cS + Str('ftnlen') + Opt(S + len_), IGNORE), (cS + sep_seq(['(', 'ftnlen', ')'], S) + S + digits, IGNORE), (Bol + Str('ftnlen ') + len_ + Str(';\n'), IGNORE), (cS + len_, TEXT), (AnyChar, TEXT)])

def scrubFtnlen(source):
    if False:
        for i in range(10):
            print('nop')
    return runScanner(source, LenSubsScanner)[0]

def cleanSource(source):
    if False:
        print('Hello World!')
    source = re.sub('[\\t ]+\\n', '\n', source)
    source = re.sub('(?m)^[\\t ]*/\\* *\\.\\. .*?\\n', '', source)
    source = re.sub('\\n\\n\\n\\n+', '\\n\\n\\n', source)
    return source

class LineQueue:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        object.__init__(self)
        self._queue = []

    def add(self, line):
        if False:
            print('Hello World!')
        self._queue.append(line)

    def clear(self):
        if False:
            i = 10
            return i + 15
        self._queue = []

    def flushTo(self, other_queue):
        if False:
            print('Hello World!')
        for line in self._queue:
            other_queue.add(line)
        self.clear()

    def getValue(self):
        if False:
            for i in range(10):
                print('nop')
        q = LineQueue()
        self.flushTo(q)
        s = ''.join(q._queue)
        self.clear()
        return s

class CommentQueue(LineQueue):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        LineQueue.__init__(self)

    def add(self, line):
        if False:
            i = 10
            return i + 15
        if line.strip() == '':
            LineQueue.add(self, '\n')
        else:
            line = '  ' + line[2:-3].rstrip() + '\n'
            LineQueue.add(self, line)

    def flushTo(self, other_queue):
        if False:
            print('Hello World!')
        if len(self._queue) == 0:
            pass
        elif len(self._queue) == 1:
            other_queue.add('/*' + self._queue[0][2:].rstrip() + ' */\n')
        else:
            other_queue.add('/*\n')
            LineQueue.flushTo(self, other_queue)
            other_queue.add('*/\n')
        self.clear()

def cleanComments(source):
    if False:
        print('Hello World!')
    lines = LineQueue()
    comments = CommentQueue()

    def isCommentLine(line):
        if False:
            return 10
        return line.startswith('/*') and line.endswith('*/\n')
    blanks = LineQueue()

    def isBlank(line):
        if False:
            i = 10
            return i + 15
        return line.strip() == ''

    def SourceLines(line):
        if False:
            for i in range(10):
                print('nop')
        if isCommentLine(line):
            comments.add(line)
            return HaveCommentLines
        else:
            lines.add(line)
            return SourceLines

    def HaveCommentLines(line):
        if False:
            print('Hello World!')
        if isBlank(line):
            blanks.add('\n')
            return HaveBlankLines
        elif isCommentLine(line):
            comments.add(line)
            return HaveCommentLines
        else:
            comments.flushTo(lines)
            lines.add(line)
            return SourceLines

    def HaveBlankLines(line):
        if False:
            while True:
                i = 10
        if isBlank(line):
            blanks.add('\n')
            return HaveBlankLines
        elif isCommentLine(line):
            blanks.flushTo(comments)
            comments.add(line)
            return HaveCommentLines
        else:
            comments.flushTo(lines)
            blanks.flushTo(lines)
            lines.add(line)
            return SourceLines
    state = SourceLines
    for line in UStringIO(source):
        state = state(line)
    comments.flushTo(lines)
    return lines.getValue()

def removeHeader(source):
    if False:
        while True:
            i = 10
    lines = LineQueue()

    def LookingForHeader(line):
        if False:
            return 10
        m = re.match('/\\*[^\\n]*-- translated', line)
        if m:
            return InHeader
        else:
            lines.add(line)
            return LookingForHeader

    def InHeader(line):
        if False:
            i = 10
            return i + 15
        if line.startswith('*/'):
            return OutOfHeader
        else:
            return InHeader

    def OutOfHeader(line):
        if False:
            while True:
                i = 10
        if line.startswith('#include "f2c.h"'):
            pass
        else:
            lines.add(line)
        return OutOfHeader
    state = LookingForHeader
    for line in UStringIO(source):
        state = state(line)
    return lines.getValue()

def removeSubroutinePrototypes(source):
    if False:
        i = 10
        return i + 15
    return source

def removeBuiltinFunctions(source):
    if False:
        i = 10
        return i + 15
    lines = LineQueue()

    def LookingForBuiltinFunctions(line):
        if False:
            return 10
        if line.strip() == '/* Builtin functions */':
            return InBuiltInFunctions
        else:
            lines.add(line)
            return LookingForBuiltinFunctions

    def InBuiltInFunctions(line):
        if False:
            i = 10
            return i + 15
        if line.strip() == '':
            return LookingForBuiltinFunctions
        else:
            return InBuiltInFunctions
    state = LookingForBuiltinFunctions
    for line in UStringIO(source):
        state = state(line)
    return lines.getValue()

def replaceDlamch(source):
    if False:
        print('Hello World!')
    'Replace dlamch_ calls with appropriate macros'

    def repl(m):
        if False:
            print('Hello World!')
        s = m.group(1)
        return dict(E='EPSILON', P='PRECISION', S='SAFEMINIMUM', B='BASE')[s[0]]
    source = re.sub('dlamch_\\("(.*?)"\\)', repl, source)
    source = re.sub('^\\s+extern.*? dlamch_.*?;$(?m)', '', source)
    return source

def scrubSource(source, nsteps=None, verbose=False):
    if False:
        return 10
    steps = [('scrubbing ftnlen', scrubFtnlen), ('remove header', removeHeader), ('clean source', cleanSource), ('clean comments', cleanComments), ('replace dlamch_() calls', replaceDlamch), ('remove prototypes', removeSubroutinePrototypes), ('remove builtin function prototypes', removeBuiltinFunctions)]
    if nsteps is not None:
        steps = steps[:nsteps]
    for (msg, step) in steps:
        if verbose:
            print(msg)
        source = step(source)
    return source
if __name__ == '__main__':
    filename = sys.argv[1]
    outfilename = os.path.join(sys.argv[2], os.path.basename(filename))
    with open(filename) as fo:
        source = fo.read()
    if len(sys.argv) > 3:
        nsteps = int(sys.argv[3])
    else:
        nsteps = None
    source = scrub_source(source, nsteps, verbose=True)
    with open(outfilename, 'w') as writefo:
        writefo.write(source)