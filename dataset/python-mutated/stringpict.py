"""Prettyprinter by Jurjen Bos.
(I hate spammers: mail me at pietjepuk314 at the reverse of ku.oc.oohay).
All objects have a method that create a "stringPict",
that can be used in the str method for pretty printing.

Updates by Jason Gedge (email <my last name> at cs mun ca)
    - terminal_string() method
    - minor fixes and changes (mostly to prettyForm)

TODO:
    - Allow left/center/right alignment options for above/below and
      top/center/bottom alignment options for left/right
"""
import shutil
from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width, center
from sympy.utilities.exceptions import sympy_deprecation_warning

class stringPict:
    """An ASCII picture.
    The pictures are represented as a list of equal length strings.
    """
    LINE = 'line'

    def __init__(self, s, baseline=0):
        if False:
            while True:
                i = 10
        'Initialize from string.\n        Multiline strings are centered.\n        '
        self.s = s
        self.picture = stringPict.equalLengths(s.splitlines())
        self.baseline = baseline
        self.binding = None

    @staticmethod
    def equalLengths(lines):
        if False:
            while True:
                i = 10
        if not lines:
            return ['']
        width = max((line_width(line) for line in lines))
        return [center(line, width) for line in lines]

    def height(self):
        if False:
            return 10
        'The height of the picture in characters.'
        return len(self.picture)

    def width(self):
        if False:
            while True:
                i = 10
        'The width of the picture in characters.'
        return line_width(self.picture[0])

    @staticmethod
    def next(*args):
        if False:
            i = 10
            return i + 15
        'Put a string of stringPicts next to each other.\n        Returns string, baseline arguments for stringPict.\n        '
        objects = []
        for arg in args:
            if isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)
        newBaseline = max((obj.baseline for obj in objects))
        newHeightBelowBaseline = max((obj.height() - obj.baseline for obj in objects))
        newHeight = newBaseline + newHeightBelowBaseline
        pictures = []
        for obj in objects:
            oneEmptyLine = [' ' * obj.width()]
            basePadding = newBaseline - obj.baseline
            totalPadding = newHeight - obj.height()
            pictures.append(oneEmptyLine * basePadding + obj.picture + oneEmptyLine * (totalPadding - basePadding))
        result = [''.join(lines) for lines in zip(*pictures)]
        return ('\n'.join(result), newBaseline)

    def right(self, *args):
        if False:
            i = 10
            return i + 15
        'Put pictures next to this one.\n        Returns string, baseline arguments for stringPict.\n        (Multiline) strings are allowed, and are given a baseline of 0.\n\n        Examples\n        ========\n\n        >>> from sympy.printing.pretty.stringpict import stringPict\n        >>> print(stringPict("10").right(" + ",stringPict("1\\r-\\r2",1))[0])\n             1\n        10 + -\n             2\n\n        '
        return stringPict.next(self, *args)

    def left(self, *args):
        if False:
            i = 10
            return i + 15
        'Put pictures (left to right) at left.\n        Returns string, baseline arguments for stringPict.\n        '
        return stringPict.next(*args + (self,))

    @staticmethod
    def stack(*args):
        if False:
            print('Hello World!')
        "Put pictures on top of each other,\n        from top to bottom.\n        Returns string, baseline arguments for stringPict.\n        The baseline is the baseline of the second picture.\n        Everything is centered.\n        Baseline is the baseline of the second picture.\n        Strings are allowed.\n        The special value stringPict.LINE is a row of '-' extended to the width.\n        "
        objects = []
        for arg in args:
            if arg is not stringPict.LINE and isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)
        newWidth = max((obj.width() for obj in objects if obj is not stringPict.LINE))
        lineObj = stringPict(hobj('-', newWidth))
        for (i, obj) in enumerate(objects):
            if obj is stringPict.LINE:
                objects[i] = lineObj
        newPicture = [center(line, newWidth) for obj in objects for line in obj.picture]
        newBaseline = objects[0].height() + objects[1].baseline
        return ('\n'.join(newPicture), newBaseline)

    def below(self, *args):
        if False:
            return 10
        'Put pictures under this picture.\n        Returns string, baseline arguments for stringPict.\n        Baseline is baseline of top picture\n\n        Examples\n        ========\n\n        >>> from sympy.printing.pretty.stringpict import stringPict\n        >>> print(stringPict("x+3").below(\n        ...       stringPict.LINE, \'3\')[0]) #doctest: +NORMALIZE_WHITESPACE\n        x+3\n        ---\n         3\n\n        '
        (s, baseline) = stringPict.stack(self, *args)
        return (s, self.baseline)

    def above(self, *args):
        if False:
            while True:
                i = 10
        'Put pictures above this picture.\n        Returns string, baseline arguments for stringPict.\n        Baseline is baseline of bottom picture.\n        '
        (string, baseline) = stringPict.stack(*args + (self,))
        baseline = len(string.splitlines()) - self.height() + self.baseline
        return (string, baseline)

    def parens(self, left='(', right=')', ifascii_nougly=False):
        if False:
            return 10
        "Put parentheses around self.\n        Returns string, baseline arguments for stringPict.\n\n        left or right can be None or empty string which means 'no paren from\n        that side'\n        "
        h = self.height()
        b = self.baseline
        if ifascii_nougly and (not pretty_use_unicode()):
            h = 1
            b = 0
        res = self
        if left:
            lparen = stringPict(vobj(left, h), baseline=b)
            res = stringPict(*lparen.right(self))
        if right:
            rparen = stringPict(vobj(right, h), baseline=b)
            res = stringPict(*res.right(rparen))
        return ('\n'.join(res.picture), res.baseline)

    def leftslash(self):
        if False:
            for i in range(10):
                print('nop')
        'Precede object by a slash of the proper size.\n        '
        height = max(self.baseline, self.height() - 1 - self.baseline) * 2 + 1
        slash = '\n'.join((' ' * (height - i - 1) + xobj('/', 1) + ' ' * i for i in range(height)))
        return self.left(stringPict(slash, height // 2))

    def root(self, n=None):
        if False:
            return 10
        'Produce a nice root symbol.\n        Produces ugly results for big n inserts.\n        '
        result = self.above('_' * self.width())
        height = self.height()
        slash = '\n'.join((' ' * (height - i - 1) + '/' + ' ' * i for i in range(height)))
        slash = stringPict(slash, height - 1)
        if height > 2:
            downline = stringPict('\\ \n \\', 1)
        else:
            downline = stringPict('\\')
        if n is not None and n.width() > downline.width():
            downline = downline.left(' ' * (n.width() - downline.width()))
            downline = downline.above(n)
        root = downline.right(slash)
        root.baseline = result.baseline - result.height() + root.height()
        return result.left(root)

    def render(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Return the string form of self.\n\n           Unless the argument line_break is set to False, it will\n           break the expression in a form that can be printed\n           on the terminal without being broken up.\n         '
        if kwargs['wrap_line'] is False:
            return '\n'.join(self.picture)
        if kwargs['num_columns'] is not None:
            ncols = kwargs['num_columns']
        else:
            ncols = self.terminal_width()
        if ncols <= 0:
            ncols = 80
        if self.width() <= ncols:
            return type(self.picture[0])(self)
        '\n        Break long-lines in a visually pleasing format.\n        without overflow indicators | with overflow indicators\n        |   2  2        3     |     |   2  2        3    ↪|\n        |6*x *y  + 4*x*y  +   |     |6*x *y  + 4*x*y  +  ↪|\n        |                     |     |                     |\n        |     3    4    4     |     |↪      3    4    4   |\n        |4*y*x  + x  + y      |     |↪ 4*y*x  + x  + y    |\n        |a*c*e + a*c*f + a*d  |     |a*c*e + a*c*f + a*d ↪|\n        |*e + a*d*f + b*c*e   |     |                     |\n        |+ b*c*f + b*d*e + b  |     |↪ *e + a*d*f + b*c* ↪|\n        |*d*f                 |     |                     |\n        |                     |     |↪ e + b*c*f + b*d*e ↪|\n        |                     |     |                     |\n        |                     |     |↪ + b*d*f            |\n        '
        overflow_first = ''
        if kwargs['use_unicode'] or pretty_use_unicode():
            overflow_start = '↪ '
            overflow_end = ' ↪'
        else:
            overflow_start = '> '
            overflow_end = ' >'

        def chunks(line):
            if False:
                i = 10
                return i + 15
            'Yields consecutive chunks of line_width ncols'
            prefix = overflow_first
            (width, start) = (line_width(prefix + overflow_end), 0)
            for (i, x) in enumerate(line):
                wx = line_width(x)
                if width + wx > ncols:
                    yield (prefix + line[start:i] + overflow_end)
                    prefix = overflow_start
                    (width, start) = (line_width(prefix + overflow_end), i)
                width += wx
            yield (prefix + line[start:])
        pictures = zip(*map(chunks, self.picture))
        pictures = ['\n'.join(picture) for picture in pictures]
        return '\n\n'.join(pictures)

    def terminal_width(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the terminal width if possible, otherwise return 0.\n        '
        size = shutil.get_terminal_size(fallback=(0, 0))
        return size.columns

    def __eq__(self, o):
        if False:
            return 10
        if isinstance(o, str):
            return '\n'.join(self.picture) == o
        elif isinstance(o, stringPict):
            return o.picture == self.picture
        return False

    def __hash__(self):
        if False:
            return 10
        return super().__hash__()

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join(self.picture)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'stringPict(%r,%d)' % ('\n'.join(self.picture), self.baseline)

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        return self.picture[index]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.s)

class prettyForm(stringPict):
    """
    Extension of the stringPict class that knows about basic math applications,
    optimizing double minus signs.

    "Binding" is interpreted as follows::

        ATOM this is an atom: never needs to be parenthesized
        FUNC this is a function application: parenthesize if added (?)
        DIV  this is a division: make wider division if divided
        POW  this is a power: only parenthesize if exponent
        MUL  this is a multiplication: parenthesize if powered
        ADD  this is an addition: parenthesize if multiplied or powered
        NEG  this is a negative number: optimize if added, parenthesize if
             multiplied or powered
        OPEN this is an open object: parenthesize if added, multiplied, or
             powered (example: Piecewise)
    """
    (ATOM, FUNC, DIV, POW, MUL, ADD, NEG, OPEN) = range(8)

    def __init__(self, s, baseline=0, binding=0, unicode=None):
        if False:
            print('Hello World!')
        'Initialize from stringPict and binding power.'
        stringPict.__init__(self, s, baseline)
        self.binding = binding
        if unicode is not None:
            sympy_deprecation_warning('\n                The unicode argument to prettyForm is deprecated. Only the s\n                argument (the first positional argument) should be passed.\n                ', deprecated_since_version='1.7', active_deprecations_target='deprecated-pretty-printing-functions')
        self._unicode = unicode or s

    @property
    def unicode(self):
        if False:
            i = 10
            return i + 15
        sympy_deprecation_warning('\n            The prettyForm.unicode attribute is deprecated. Use the\n            prettyForm.s attribute instead.\n            ', deprecated_since_version='1.7', active_deprecations_target='deprecated-pretty-printing-functions')
        return self._unicode

    def __add__(self, *others):
        if False:
            i = 10
            return i + 15
        'Make a pretty addition.\n        Addition of negative numbers is simplified.\n        '
        arg = self
        if arg.binding > prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            if arg.binding > prettyForm.NEG:
                arg = stringPict(*arg.parens())
            if arg.binding != prettyForm.NEG:
                result.append(' + ')
            result.append(arg)
        return prettyForm(*stringPict.next(*result), binding=prettyForm.ADD)

    def __truediv__(self, den, slashed=False):
        if False:
            for i in range(10):
                print('nop')
        'Make a pretty division; stacked or slashed.\n        '
        if slashed:
            raise NotImplementedError("Can't do slashed fraction yet")
        num = self
        if num.binding == prettyForm.DIV:
            num = stringPict(*num.parens())
        if den.binding == prettyForm.DIV:
            den = stringPict(*den.parens())
        if num.binding == prettyForm.NEG:
            num = num.right(' ')[0]
        return prettyForm(*stringPict.stack(num, stringPict.LINE, den), binding=prettyForm.DIV)

    def __mul__(self, *others):
        if False:
            return 10
        'Make a pretty multiplication.\n        Parentheses are needed around +, - and neg.\n        '
        quantity = {'degree': '°'}
        if len(others) == 0:
            return self
        arg = self
        if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            if arg.picture[0] not in quantity.values():
                result.append(xsym('*'))
            if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
                arg = stringPict(*arg.parens())
            result.append(arg)
        len_res = len(result)
        for i in range(len_res):
            if i < len_res - 1 and result[i] == '-1' and (result[i + 1] == xsym('*')):
                result.pop(i)
                result.pop(i)
                result.insert(i, '-')
        if result[0][0] == '-':
            bin = prettyForm.NEG
            if result[0] == '-':
                right = result[1]
                if right.picture[right.baseline][0] == '-':
                    result[0] = '- '
        else:
            bin = prettyForm.MUL
        return prettyForm(*stringPict.next(*result), binding=bin)

    def __repr__(self):
        if False:
            return 10
        return 'prettyForm(%r,%d,%d)' % ('\n'.join(self.picture), self.baseline, self.binding)

    def __pow__(self, b):
        if False:
            print('Hello World!')
        'Make a pretty power.\n        '
        a = self
        use_inline_func_form = False
        if b.binding == prettyForm.POW:
            b = stringPict(*b.parens())
        if a.binding > prettyForm.FUNC:
            a = stringPict(*a.parens())
        elif a.binding == prettyForm.FUNC:
            if b.height() > 1:
                a = stringPict(*a.parens())
            else:
                use_inline_func_form = True
        if use_inline_func_form:
            b.baseline = a.prettyFunc.baseline + b.height()
            func = stringPict(*a.prettyFunc.right(b))
            return prettyForm(*func.right(a.prettyArgs))
        else:
            top = stringPict(*b.left(' ' * a.width()))
            bot = stringPict(*a.right(' ' * b.width()))
        return prettyForm(*bot.above(top), binding=prettyForm.POW)
    simpleFunctions = ['sin', 'cos', 'tan']

    @staticmethod
    def apply(function, *args):
        if False:
            while True:
                i = 10
        'Functions of one or more variables.\n        '
        if function in prettyForm.simpleFunctions:
            assert len(args) == 1, 'Simple function %s must have 1 argument' % function
            arg = args[0].__pretty__()
            if arg.binding <= prettyForm.DIV:
                return prettyForm(*arg.left(function + ' '), binding=prettyForm.FUNC)
        argumentList = []
        for arg in args:
            argumentList.append(',')
            argumentList.append(arg.__pretty__())
        argumentList = stringPict(*stringPict.next(*argumentList[1:]))
        argumentList = stringPict(*argumentList.parens())
        return prettyForm(*argumentList.left(function), binding=prettyForm.ATOM)