from sympy.core.singleton import S
from sympy.printing.tableform import TableForm
from sympy.printing.latex import latex
from sympy.abc import x
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import raises
from textwrap import dedent

def test_TableForm():
    if False:
        return 10
    s = str(TableForm([['a', 'b'], ['c', 'd'], ['e', 0]], headings='automatic'))
    assert s == '  | 1 2\n-------\n1 | a b\n2 | c d\n3 | e  '
    s = str(TableForm([['a', 'b'], ['c', 'd'], ['e', 0]], headings='automatic', wipe_zeros=False))
    assert s == dedent('          | 1 2\n        -------\n        1 | a b\n        2 | c d\n        3 | e 0')
    s = str(TableForm([[x ** 2, 'b'], ['c', x ** 2], ['e', 'f']], headings=('automatic', None)))
    assert s == '1 | x**2 b   \n2 | c    x**2\n3 | e    f   '
    s = str(TableForm([['a', 'b'], ['c', 'd'], ['e', 'f']], headings=(None, 'automatic')))
    assert s == dedent('        1 2\n        ---\n        a b\n        c d\n        e f')
    s = str(TableForm([[5, 7], [4, 2], [10, 3]], headings=[['Group A', 'Group B', 'Group C'], ['y1', 'y2']]))
    assert s == '        | y1 y2\n---------------\nGroup A | 5  7 \nGroup B | 4  2 \nGroup C | 10 3 '
    raises(ValueError, lambda : TableForm([[5, 7], [4, 2], [10, 3]], headings=[['Group A', 'Group B', 'Group C'], ['y1', 'y2']], alignments='middle'))
    s = str(TableForm([[5, 7], [4, 2], [10, 3]], headings=[['Group A', 'Group B', 'Group C'], ['y1', 'y2']], alignments='right'))
    assert s == dedent('                | y1 y2\n        ---------------\n        Group A |  5  7\n        Group B |  4  2\n        Group C | 10  3')
    d = [[1, 100], [100, 1]]
    s = TableForm(d, headings=(('xxx', 'x'), None), alignments='l')
    assert str(s) == 'xxx | 1   100\n  x | 100 1  '
    s = TableForm(d, headings=(('xxx', 'x'), None), alignments='lr')
    assert str(s) == dedent('    xxx | 1   100\n      x | 100   1')
    s = TableForm(d, headings=(('xxx', 'x'), None), alignments='clr')
    assert str(s) == dedent('    xxx | 1   100\n     x  | 100   1')
    s = TableForm(d, headings=(('xxx', 'x'), None))
    assert str(s) == 'xxx | 1   100\n  x | 100 1  '
    raises(ValueError, lambda : TableForm(d, alignments='clr'))
    s = str(TableForm([[None, '-', 2], [1]], pad='?'))
    assert s == dedent('        ? - 2\n        1 ? ?')

def test_TableForm_latex():
    if False:
        for i in range(10):
            print('nop')
    s = latex(TableForm([[0, x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], wipe_zeros=True, headings=('automatic', 'automatic')))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 &   & $x^{3}$ \\\\\n2 & $c$ & $\\frac{1}{4}$ \\\\\n3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'
    s = latex(TableForm([[0, x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], wipe_zeros=True, headings=('automatic', 'automatic'), alignments='l'))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 &   & $x^{3}$ \\\\\n2 & $c$ & $\\frac{1}{4}$ \\\\\n3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'
    s = latex(TableForm([[0, x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], wipe_zeros=True, headings=('automatic', 'automatic'), alignments='l' * 3))
    assert s == '\\begin{tabular}{l l l}\n & 1 & 2 \\\\\n\\hline\n1 &   & $x^{3}$ \\\\\n2 & $c$ & $\\frac{1}{4}$ \\\\\n3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'
    s = latex(TableForm([['a', x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], headings=('automatic', 'automatic')))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 & $a$ & $x^{3}$ \\\\\n2 & $c$ & $\\frac{1}{4}$ \\\\\n3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'
    s = latex(TableForm([['a', x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], formats=['(%s)', None], headings=('automatic', 'automatic')))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 & (a) & $x^{3}$ \\\\\n2 & (c) & $\\frac{1}{4}$ \\\\\n3 & (sqrt(x)) & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'

    def neg_in_paren(x, i, j):
        if False:
            i = 10
            return i + 15
        if i % 2:
            return ('(%s)' if x < 0 else '%s') % x
        else:
            pass
    s = latex(TableForm([[-1, 2], [-3, 4]], formats=[neg_in_paren] * 2, headings=('automatic', 'automatic')))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 & -1 & 2 \\\\\n2 & (-3) & 4 \\\\\n\\end{tabular}'
    s = latex(TableForm([['a', x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]]))
    assert s == '\\begin{tabular}{l l}\n$a$ & $x^{3}$ \\\\\n$c$ & $\\frac{1}{4}$ \\\\\n$\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'