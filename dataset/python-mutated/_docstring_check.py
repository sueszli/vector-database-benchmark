import math

def check(app, what, name, obj, options, lines):
    if False:
        print('Hello World!')
    ctx = DocstringCheckContext(app, what, name, obj, options, lines)
    if what in ('function', 'method'):
        _docstring_check_returns_indent(ctx)

class DocstringCheckContext(object):

    def __init__(self, app, what, name, obj, options, lines):
        if False:
            print('Hello World!')
        self.app = app
        self.what = what
        self.name = name
        self.obj = obj
        self.options = options
        self.lines = lines
        self.iline = 0

    def nextline(self):
        if False:
            return 10
        if self.iline >= len(self.lines):
            raise StopIteration
        line = self.lines[self.iline]
        self.iline += 1
        return line

    def error(self, msg, include_line=True, include_source=True):
        if False:
            return 10
        lines = self.lines
        iline = self.iline - 1
        msg = '{}\n\non {}'.format(msg, self.name)
        if include_line and 0 <= iline < len(lines):
            line = lines[iline]
            msg += '\n' + 'at line {}: "{}"\n'.format(iline, line)
        if include_source:
            msg += '\n'
            msg += 'docstring:\n'
            digits = int(math.floor(math.log10(len(lines)))) + 1
            linum_fmt = '{{:0{}d}} '.format(digits)
            for (i, line) in enumerate(lines):
                msg += linum_fmt.format(i) + line + '\n'
        raise InvalidDocstringError(msg, self, iline)

class InvalidDocstringError(Exception):

    def __init__(self, msg, ctx, iline):
        if False:
            for i in range(10):
                print('nop')
        super(InvalidDocstringError, self).__init__(self, msg)
        self.msg = msg
        self.ctx = ctx
        self.iline = iline

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.msg

def _docstring_check_returns_indent(ctx):
    if False:
        while True:
            i = 10
    try:
        line = ctx.nextline()
        while line != ':returns:':
            line = ctx.nextline()
    except StopIteration:
        return
    try:
        line = ctx.nextline()
        while not line:
            line = ctx.nextline()
    except StopIteration:
        ctx.error('`Returns` section has no content')
    nindent = next((i for (i, c) in enumerate(line) if c != ' '))
    try:
        line = ctx.nextline()
        while line.startswith(' '):
            if not line.startswith(' ' * nindent) or line[nindent:].startswith(' '):
                ctx.error('Invalid indentation of `Returns` section')
            line = ctx.nextline()
    except StopIteration:
        pass