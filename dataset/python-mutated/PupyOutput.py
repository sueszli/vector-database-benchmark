class Hint(object):
    __slots__ = ()

class Text(Hint):
    __slots__ = 'data'

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.data = data

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<Hint({}): {}>'.format(self.__class__.__name__, repr(self.data))

    def __str__(self):
        if False:
            return 10
        raise NotImplementedError('__str__ is not implemented for class {}'.format(self.__class__.__name__))

class Table(Text):
    __slots__ = ('headers', 'caption', 'legend', 'vspace')

    def __init__(self, data, headers=None, caption=None, legend=True, vspace=0):
        if False:
            return 10
        super(Table, self).__init__(data)
        self.headers = headers
        self.caption = caption
        self.legend = legend
        self.vspace = vspace

class List(Text):
    __slots__ = ('caption', 'bullet', 'indent')

    def __init__(self, data, bullet='+', indent=2, caption=None):
        if False:
            for i in range(10):
                print('nop')
        super(List, self).__init__(data)
        self.data = data
        self.bullet = bullet
        self.caption = caption
        self.indent = indent

class Stream(Text):
    __slots__ = ()

class Line(Text):
    __slots__ = 'dm'

    def __init__(self, *data):
        if False:
            while True:
                i = 10
        super(Line, self).__init__(data)
        self.dm = ' '

class TruncateToTerm(Text):
    __slots__ = ()

class Color(Text):
    __slots__ = 'color'

    def __init__(self, data, color):
        if False:
            for i in range(10):
                print('nop')
        super(Color, self).__init__(data)
        self.color = color

class Title(Text):
    __slots__ = ()

class MultiPart(Text):
    __slots__ = ()

class NewLine(Text):
    __slots__ = ()

    def __init__(self, lines=1):
        if False:
            while True:
                i = 10
        super(NewLine, self).__init__(lines)

class Log(Text):
    __slots__ = ()

class Info(Text):
    __slots__ = ()

class ServiceInfo(Text):
    __slots__ = ()

class Warn(Text):
    __slots__ = ()

class Error(Text):
    __slots__ = 'header'

    def __init__(self, error, header=None):
        if False:
            print('Hello World!')
        super(Error, self).__init__(error)
        self.header = header

class Success(Text):
    __slots__ = ()

class Section(Text):
    __slots__ = 'header'

    def __init__(self, header, data):
        if False:
            return 10
        super(Section, self).__init__(data)
        self.header = header

class Usage(Text):
    __slots__ = 'module'

    def __init__(self, module, data):
        if False:
            i = 10
            return i + 15
        super(Usage, self).__init__(data)
        self.module = module

class Pygment(Text):
    __slots__ = 'lexer'

    def __init__(self, lexer, data):
        if False:
            while True:
                i = 10
        super(Pygment, self).__init__(data)
        self.lexer = lexer

class Interact(Hint):
    __slots__ = ()

class Indent(Text):
    __slots__ = 'indent'

    def __init__(self, data, indent=2):
        if False:
            for i in range(10):
                print('nop')
        super(Indent, self).__init__(data)
        self.indent = indent

class Prompt(Interact):
    __slots__ = ('request', 'hide')

    def __init__(self, request, hide=False):
        if False:
            return 10
        self.request = request
        self.hide = hide

class Terminal(Hint):
    __slots__ = ()