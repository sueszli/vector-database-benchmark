import re
from sympy.concrete.products import product
from sympy.concrete.summations import Sum
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin

class MaximaHelpers:

    def maxima_expand(expr):
        if False:
            for i in range(10):
                print('nop')
        return expr.expand()

    def maxima_float(expr):
        if False:
            while True:
                i = 10
        return expr.evalf()

    def maxima_trigexpand(expr):
        if False:
            for i in range(10):
                print('nop')
        return expr.expand(trig=True)

    def maxima_sum(a1, a2, a3, a4):
        if False:
            i = 10
            return i + 15
        return Sum(a1, (a2, a3, a4)).doit()

    def maxima_product(a1, a2, a3, a4):
        if False:
            print('Hello World!')
        return product(a1, (a2, a3, a4))

    def maxima_csc(expr):
        if False:
            print('Hello World!')
        return 1 / sin(expr)

    def maxima_sec(expr):
        if False:
            print('Hello World!')
        return 1 / cos(expr)
sub_dict = {'pi': re.compile('%pi'), 'E': re.compile('%e'), 'I': re.compile('%i'), '**': re.compile('\\^'), 'oo': re.compile('\\binf\\b'), '-oo': re.compile('\\bminf\\b'), "'-'": re.compile('\\bminus\\b'), 'maxima_expand': re.compile('\\bexpand\\b'), 'maxima_float': re.compile('\\bfloat\\b'), 'maxima_trigexpand': re.compile('\\btrigexpand'), 'maxima_sum': re.compile('\\bsum\\b'), 'maxima_product': re.compile('\\bproduct\\b'), 'cancel': re.compile('\\bratsimp\\b'), 'maxima_csc': re.compile('\\bcsc\\b'), 'maxima_sec': re.compile('\\bsec\\b')}
var_name = re.compile('^\\s*(\\w+)\\s*:')

def parse_maxima(str, globals=None, name_dict={}):
    if False:
        for i in range(10):
            print('nop')
    str = str.strip()
    str = str.rstrip('; ')
    for (k, v) in sub_dict.items():
        str = v.sub(k, str)
    assign_var = None
    var_match = var_name.search(str)
    if var_match:
        assign_var = var_match.group(1)
        str = str[var_match.end():].strip()
    dct = MaximaHelpers.__dict__.copy()
    dct.update(name_dict)
    obj = sympify(str, locals=dct)
    if assign_var and globals:
        globals[assign_var] = obj
    return obj