from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from types import FunctionType

class TableForm:
    """
    Create a nice table representation of data.

    Examples
    ========

    >>> from sympy import TableForm
    >>> t = TableForm([[5, 7], [4, 2], [10, 3]])
    >>> print(t)
    5  7
    4  2
    10 3

    You can use the SymPy's printing system to produce tables in any
    format (ascii, latex, html, ...).

    >>> print(t.as_latex())
    \\begin{tabular}{l l}
    $5$ & $7$ \\\\
    $4$ & $2$ \\\\
    $10$ & $3$ \\\\
    \\end{tabular}

    """

    def __init__(self, data, **kwarg):
        if False:
            i = 10
            return i + 15
        '\n        Creates a TableForm.\n\n        Parameters:\n\n            data ...\n                            2D data to be put into the table; data can be\n                            given as a Matrix\n\n            headings ...\n                            gives the labels for rows and columns:\n\n                            Can be a single argument that applies to both\n                            dimensions:\n\n                                - None ... no labels\n                                - "automatic" ... labels are 1, 2, 3, ...\n\n                            Can be a list of labels for rows and columns:\n                            The labels for each dimension can be given\n                            as None, "automatic", or [l1, l2, ...] e.g.\n                            ["automatic", None] will number the rows\n\n                            [default: None]\n\n            alignments ...\n                            alignment of the columns with:\n\n                                - "left" or "<"\n                                - "center" or "^"\n                                - "right" or ">"\n\n                            When given as a single value, the value is used for\n                            all columns. The row headings (if given) will be\n                            right justified unless an explicit alignment is\n                            given for it and all other columns.\n\n                            [default: "left"]\n\n            formats ...\n                            a list of format strings or functions that accept\n                            3 arguments (entry, row number, col number) and\n                            return a string for the table entry. (If a function\n                            returns None then the _print method will be used.)\n\n            wipe_zeros ...\n                            Do not show zeros in the table.\n\n                            [default: True]\n\n            pad ...\n                            the string to use to indicate a missing value (e.g.\n                            elements that are None or those that are missing\n                            from the end of a row (i.e. any row that is shorter\n                            than the rest is assumed to have missing values).\n                            When None, nothing will be shown for values that\n                            are missing from the end of a row; values that are\n                            None, however, will be shown.\n\n                            [default: None]\n\n        Examples\n        ========\n\n        >>> from sympy import TableForm, Symbol\n        >>> TableForm([[5, 7], [4, 2], [10, 3]])\n        5  7\n        4  2\n        10 3\n        >>> TableForm([list(\'.\'*i) for i in range(1, 4)], headings=\'automatic\')\n          | 1 2 3\n        ---------\n        1 | .\n        2 | . .\n        3 | . . .\n        >>> TableForm([[Symbol(\'.\'*(j if not i%2 else 1)) for i in range(3)]\n        ...            for j in range(4)], alignments=\'rcl\')\n            .\n          . . .\n         .. . ..\n        ... . ...\n        '
        from sympy.matrices.dense import Matrix
        if isinstance(data, Matrix):
            data = data.tolist()
        _h = len(data)
        pad = kwarg.get('pad', None)
        ok_None = False
        if pad is None:
            pad = ' '
            ok_None = True
        pad = Symbol(pad)
        _w = max((len(line) for line in data))
        for (i, line) in enumerate(data):
            if len(line) != _w:
                line.extend([pad] * (_w - len(line)))
            for (j, lj) in enumerate(line):
                if lj is None:
                    if not ok_None:
                        lj = pad
                else:
                    try:
                        lj = S(lj)
                    except SympifyError:
                        lj = Symbol(str(lj))
                line[j] = lj
            data[i] = line
        _lines = Tuple(*[Tuple(*d) for d in data])
        headings = kwarg.get('headings', [None, None])
        if headings == 'automatic':
            _headings = [range(1, _h + 1), range(1, _w + 1)]
        else:
            (h1, h2) = headings
            if h1 == 'automatic':
                h1 = range(1, _h + 1)
            if h2 == 'automatic':
                h2 = range(1, _w + 1)
            _headings = [h1, h2]
        allow = ('l', 'r', 'c')
        alignments = kwarg.get('alignments', 'l')

        def _std_align(a):
            if False:
                print('Hello World!')
            a = a.strip().lower()
            if len(a) > 1:
                return {'left': 'l', 'right': 'r', 'center': 'c'}.get(a, a)
            else:
                return {'<': 'l', '>': 'r', '^': 'c'}.get(a, a)
        std_align = _std_align(alignments)
        if std_align in allow:
            _alignments = [std_align] * _w
        else:
            _alignments = []
            for a in alignments:
                std_align = _std_align(a)
                _alignments.append(std_align)
                if std_align not in ('l', 'r', 'c'):
                    raise ValueError('alignment "%s" unrecognized' % alignments)
        if _headings[0] and len(_alignments) == _w + 1:
            _head_align = _alignments[0]
            _alignments = _alignments[1:]
        else:
            _head_align = 'r'
        if len(_alignments) != _w:
            raise ValueError('wrong number of alignments: expected %s but got %s' % (_w, len(_alignments)))
        _column_formats = kwarg.get('formats', [None] * _w)
        _wipe_zeros = kwarg.get('wipe_zeros', True)
        self._w = _w
        self._h = _h
        self._lines = _lines
        self._headings = _headings
        self._head_align = _head_align
        self._alignments = _alignments
        self._column_formats = _column_formats
        self._wipe_zeros = _wipe_zeros

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        from .str import sstr
        return sstr(self, order=None)

    def __str__(self):
        if False:
            while True:
                i = 10
        from .str import sstr
        return sstr(self, order=None)

    def as_matrix(self):
        if False:
            print('Hello World!')
        "Returns the data of the table in Matrix form.\n\n        Examples\n        ========\n\n        >>> from sympy import TableForm\n        >>> t = TableForm([[5, 7], [4, 2], [10, 3]], headings='automatic')\n        >>> t\n          | 1  2\n        --------\n        1 | 5  7\n        2 | 4  2\n        3 | 10 3\n        >>> t.as_matrix()\n        Matrix([\n        [ 5, 7],\n        [ 4, 2],\n        [10, 3]])\n        "
        from sympy.matrices.dense import Matrix
        return Matrix(self._lines)

    def as_str(self):
        if False:
            print('Hello World!')
        return str(self)

    def as_latex(self):
        if False:
            for i in range(10):
                print('nop')
        from .latex import latex
        return latex(self)

    def _sympystr(self, p):
        if False:
            return 10
        "\n        Returns the string representation of 'self'.\n\n        Examples\n        ========\n\n        >>> from sympy import TableForm\n        >>> t = TableForm([[5, 7], [4, 2], [10, 3]])\n        >>> s = t.as_str()\n\n        "
        column_widths = [0] * self._w
        lines = []
        for line in self._lines:
            new_line = []
            for i in range(self._w):
                s = str(line[i])
                if self._wipe_zeros and s == '0':
                    s = ' '
                w = len(s)
                if w > column_widths[i]:
                    column_widths[i] = w
                new_line.append(s)
            lines.append(new_line)
        if self._headings[0]:
            self._headings[0] = [str(x) for x in self._headings[0]]
            _head_width = max([len(x) for x in self._headings[0]])
        if self._headings[1]:
            new_line = []
            for i in range(self._w):
                s = str(self._headings[1][i])
                w = len(s)
                if w > column_widths[i]:
                    column_widths[i] = w
                new_line.append(s)
            self._headings[1] = new_line
        format_str = []

        def _align(align, w):
            if False:
                for i in range(10):
                    print('nop')
            return '%%%s%ss' % ('-' if align == 'l' else '', str(w))
        format_str = [_align(align, w) for (align, w) in zip(self._alignments, column_widths)]
        if self._headings[0]:
            format_str.insert(0, _align(self._head_align, _head_width))
            format_str.insert(1, '|')
        format_str = ' '.join(format_str) + '\n'
        s = []
        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                d = [''] + d
            first_line = format_str % tuple(d)
            s.append(first_line)
            s.append('-' * (len(first_line) - 1) + '\n')
        for (i, line) in enumerate(lines):
            d = [l if self._alignments[j] != 'c' else l.center(column_widths[j]) for (j, l) in enumerate(line)]
            if self._headings[0]:
                l = self._headings[0][i]
                l = l if self._head_align != 'c' else l.center(_head_width)
                d = [l] + d
            s.append(format_str % tuple(d))
        return ''.join(s)[:-1]

    def _latex(self, printer):
        if False:
            print('Hello World!')
        "\n        Returns the string representation of 'self'.\n        "
        if self._headings[1]:
            new_line = []
            for i in range(self._w):
                new_line.append(str(self._headings[1][i]))
            self._headings[1] = new_line
        alignments = []
        if self._headings[0]:
            self._headings[0] = [str(x) for x in self._headings[0]]
            alignments = [self._head_align]
        alignments.extend(self._alignments)
        s = '\\begin{tabular}{' + ' '.join(alignments) + '}\n'
        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                d = [''] + d
            first_line = ' & '.join(d) + ' \\\\' + '\n'
            s += first_line
            s += '\\hline' + '\n'
        for (i, line) in enumerate(self._lines):
            d = []
            for (j, x) in enumerate(line):
                if self._wipe_zeros and x in (0, '0'):
                    d.append(' ')
                    continue
                f = self._column_formats[j]
                if f:
                    if isinstance(f, FunctionType):
                        v = f(x, i, j)
                        if v is None:
                            v = printer._print(x)
                    else:
                        v = f % x
                    d.append(v)
                else:
                    v = printer._print(x)
                    d.append('$%s$' % v)
            if self._headings[0]:
                d = [self._headings[0][i]] + d
            s += ' & '.join(d) + ' \\\\' + '\n'
        s += '\\end{tabular}'
        return s