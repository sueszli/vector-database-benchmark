"""
A MathML printer.
"""
from __future__ import annotations
from typing import Any
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence_traditional, PRECEDENCE, PRECEDENCE_TRADITIONAL
from sympy.printing.pretty.pretty_symbology import greek_unicode
from sympy.printing.printer import Printer, print_function
from mpmath.libmp import prec_to_dps, repr_dps, to_str as mlib_to_str

class MathMLPrinterBase(Printer):
    """Contains common code required for MathMLContentPrinter and
    MathMLPresentationPrinter.
    """
    _default_settings: dict[str, Any] = {'order': None, 'encoding': 'utf-8', 'fold_frac_powers': False, 'fold_func_brackets': False, 'fold_short_frac': None, 'inv_trig_style': 'abbreviated', 'ln_notation': False, 'long_frac_ratio': None, 'mat_delim': '[', 'mat_symbol_style': 'plain', 'mul_symbol': None, 'root_notation': True, 'symbol_names': {}, 'mul_symbol_mathml_numbers': '&#xB7;'}

    def __init__(self, settings=None):
        if False:
            print('Hello World!')
        Printer.__init__(self, settings)
        from xml.dom.minidom import Document, Text
        self.dom = Document()

        class RawText(Text):

            def writexml(self, writer, indent='', addindent='', newl=''):
                if False:
                    while True:
                        i = 10
                if self.data:
                    writer.write('{}{}{}'.format(indent, self.data, newl))

        def createRawTextNode(data):
            if False:
                print('Hello World!')
            r = RawText()
            r.data = data
            r.ownerDocument = self.dom
            return r
        self.dom.createTextNode = createRawTextNode

    def doprint(self, expr):
        if False:
            return 10
        '\n        Prints the expression as MathML.\n        '
        mathML = Printer._print(self, expr)
        unistr = mathML.toxml()
        xmlbstr = unistr.encode('ascii', 'xmlcharrefreplace')
        res = xmlbstr.decode()
        return res

class MathMLContentPrinter(MathMLPrinterBase):
    """Prints an expression to the Content MathML markup language.

    References: https://www.w3.org/TR/MathML2/chapter4.html
    """
    printmethod = '_mathml_content'

    def mathml_tag(self, e):
        if False:
            while True:
                i = 10
        'Returns the MathML tag for an expression.'
        translate = {'Add': 'plus', 'Mul': 'times', 'Derivative': 'diff', 'Number': 'cn', 'int': 'cn', 'Pow': 'power', 'Max': 'max', 'Min': 'min', 'Abs': 'abs', 'And': 'and', 'Or': 'or', 'Xor': 'xor', 'Not': 'not', 'Implies': 'implies', 'Symbol': 'ci', 'MatrixSymbol': 'ci', 'RandomSymbol': 'ci', 'Integral': 'int', 'Sum': 'sum', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'cot': 'cot', 'csc': 'csc', 'sec': 'sec', 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'coth': 'coth', 'csch': 'csch', 'sech': 'sech', 'asin': 'arcsin', 'asinh': 'arcsinh', 'acos': 'arccos', 'acosh': 'arccosh', 'atan': 'arctan', 'atanh': 'arctanh', 'atan2': 'arctan', 'acot': 'arccot', 'acoth': 'arccoth', 'asec': 'arcsec', 'asech': 'arcsech', 'acsc': 'arccsc', 'acsch': 'arccsch', 'log': 'ln', 'Equality': 'eq', 'Unequality': 'neq', 'GreaterThan': 'geq', 'LessThan': 'leq', 'StrictGreaterThan': 'gt', 'StrictLessThan': 'lt', 'Union': 'union', 'Intersection': 'intersect'}
        for cls in e.__class__.__mro__:
            n = cls.__name__
            if n in translate:
                return translate[n]
        n = e.__class__.__name__
        return n.lower()

    def _print_Mul(self, expr):
        if False:
            while True:
                i = 10
        if expr.could_extract_minus_sign():
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('minus'))
            x.appendChild(self._print_Mul(-expr))
            return x
        from sympy.simplify import fraction
        (numer, denom) = fraction(expr)
        if denom is not S.One:
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('divide'))
            x.appendChild(self._print(numer))
            x.appendChild(self._print(denom))
            return x
        (coeff, terms) = expr.as_coeff_mul()
        if coeff is S.One and len(terms) == 1:
            return self._print(terms[0])
        if self.order != 'old':
            terms = Mul._from_args(terms).as_ordered_factors()
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('times'))
        if coeff != 1:
            x.appendChild(self._print(coeff))
        for term in terms:
            x.appendChild(self._print(term))
        return x

    def _print_Add(self, expr, order=None):
        if False:
            for i in range(10):
                print('nop')
        args = self._as_ordered_terms(expr, order=order)
        lastProcessed = self._print(args[0])
        plusNodes = []
        for arg in args[1:]:
            if arg.could_extract_minus_sign():
                x = self.dom.createElement('apply')
                x.appendChild(self.dom.createElement('minus'))
                x.appendChild(lastProcessed)
                x.appendChild(self._print(-arg))
                lastProcessed = x
                if arg == args[-1]:
                    plusNodes.append(lastProcessed)
            else:
                plusNodes.append(lastProcessed)
                lastProcessed = self._print(arg)
                if arg == args[-1]:
                    plusNodes.append(self._print(arg))
        if len(plusNodes) == 1:
            return lastProcessed
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('plus'))
        while plusNodes:
            x.appendChild(plusNodes.pop(0))
        return x

    def _print_Piecewise(self, expr):
        if False:
            i = 10
            return i + 15
        if expr.args[-1].cond != True:
            raise ValueError('All Piecewise expressions must contain an (expr, True) statement to be used as a default condition. Without one, the generated expression may not evaluate to anything under some condition.')
        root = self.dom.createElement('piecewise')
        for (i, (e, c)) in enumerate(expr.args):
            if i == len(expr.args) - 1 and c == True:
                piece = self.dom.createElement('otherwise')
                piece.appendChild(self._print(e))
            else:
                piece = self.dom.createElement('piece')
                piece.appendChild(self._print(e))
                piece.appendChild(self._print(c))
            root.appendChild(piece)
        return root

    def _print_MatrixBase(self, m):
        if False:
            return 10
        x = self.dom.createElement('matrix')
        for i in range(m.rows):
            x_r = self.dom.createElement('matrixrow')
            for j in range(m.cols):
                x_r.appendChild(self._print(m[i, j]))
            x.appendChild(x_r)
        return x

    def _print_Rational(self, e):
        if False:
            while True:
                i = 10
        if e.q == 1:
            x = self.dom.createElement('cn')
            x.appendChild(self.dom.createTextNode(str(e.p)))
            return x
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('divide'))
        xnum = self.dom.createElement('cn')
        xnum.appendChild(self.dom.createTextNode(str(e.p)))
        xdenom = self.dom.createElement('cn')
        xdenom.appendChild(self.dom.createTextNode(str(e.q)))
        x.appendChild(xnum)
        x.appendChild(xdenom)
        return x

    def _print_Limit(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        x_1 = self.dom.createElement('bvar')
        x_2 = self.dom.createElement('lowlimit')
        x_1.appendChild(self._print(e.args[1]))
        x_2.appendChild(self._print(e.args[2]))
        x.appendChild(x_1)
        x.appendChild(x_2)
        x.appendChild(self._print(e.args[0]))
        return x

    def _print_ImaginaryUnit(self, e):
        if False:
            while True:
                i = 10
        return self.dom.createElement('imaginaryi')

    def _print_EulerGamma(self, e):
        if False:
            return 10
        return self.dom.createElement('eulergamma')

    def _print_GoldenRatio(self, e):
        if False:
            return 10
        'We use unicode #x3c6 for Greek letter phi as defined here\n        https://www.w3.org/2003/entities/2007doc/isogrk1.html'
        x = self.dom.createElement('cn')
        x.appendChild(self.dom.createTextNode('φ'))
        return x

    def _print_Exp1(self, e):
        if False:
            print('Hello World!')
        return self.dom.createElement('exponentiale')

    def _print_Pi(self, e):
        if False:
            return 10
        return self.dom.createElement('pi')

    def _print_Infinity(self, e):
        if False:
            while True:
                i = 10
        return self.dom.createElement('infinity')

    def _print_NaN(self, e):
        if False:
            i = 10
            return i + 15
        return self.dom.createElement('notanumber')

    def _print_EmptySet(self, e):
        if False:
            i = 10
            return i + 15
        return self.dom.createElement('emptyset')

    def _print_BooleanTrue(self, e):
        if False:
            for i in range(10):
                print('nop')
        return self.dom.createElement('true')

    def _print_BooleanFalse(self, e):
        if False:
            while True:
                i = 10
        return self.dom.createElement('false')

    def _print_NegativeInfinity(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('minus'))
        x.appendChild(self.dom.createElement('infinity'))
        return x

    def _print_Integral(self, e):
        if False:
            print('Hello World!')

        def lime_recur(limits):
            if False:
                for i in range(10):
                    print('nop')
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement(self.mathml_tag(e)))
            bvar_elem = self.dom.createElement('bvar')
            bvar_elem.appendChild(self._print(limits[0][0]))
            x.appendChild(bvar_elem)
            if len(limits[0]) == 3:
                low_elem = self.dom.createElement('lowlimit')
                low_elem.appendChild(self._print(limits[0][1]))
                x.appendChild(low_elem)
                up_elem = self.dom.createElement('uplimit')
                up_elem.appendChild(self._print(limits[0][2]))
                x.appendChild(up_elem)
            if len(limits[0]) == 2:
                up_elem = self.dom.createElement('uplimit')
                up_elem.appendChild(self._print(limits[0][1]))
                x.appendChild(up_elem)
            if len(limits) == 1:
                x.appendChild(self._print(e.function))
            else:
                x.appendChild(lime_recur(limits[1:]))
            return x
        limits = list(e.limits)
        limits.reverse()
        return lime_recur(limits)

    def _print_Sum(self, e):
        if False:
            print('Hello World!')
        return self._print_Integral(e)

    def _print_Symbol(self, sym):
        if False:
            i = 10
            return i + 15
        ci = self.dom.createElement(self.mathml_tag(sym))

        def join(items):
            if False:
                print('Hello World!')
            if len(items) > 1:
                mrow = self.dom.createElement('mml:mrow')
                for (i, item) in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mml:mo')
                        mo.appendChild(self.dom.createTextNode(' '))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mml:mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            else:
                mi = self.dom.createElement('mml:mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi

        def translate(s):
            if False:
                while True:
                    i = 10
            if s in greek_unicode:
                return greek_unicode.get(s)
            else:
                return s
        (name, supers, subs) = split_super_sub(sym.name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]
        mname = self.dom.createElement('mml:mi')
        mname.appendChild(self.dom.createTextNode(name))
        if not supers:
            if not subs:
                ci.appendChild(self.dom.createTextNode(name))
            else:
                msub = self.dom.createElement('mml:msub')
                msub.appendChild(mname)
                msub.appendChild(join(subs))
                ci.appendChild(msub)
        elif not subs:
            msup = self.dom.createElement('mml:msup')
            msup.appendChild(mname)
            msup.appendChild(join(supers))
            ci.appendChild(msup)
        else:
            msubsup = self.dom.createElement('mml:msubsup')
            msubsup.appendChild(mname)
            msubsup.appendChild(join(subs))
            msubsup.appendChild(join(supers))
            ci.appendChild(msubsup)
        return ci
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print_Pow(self, e):
        if False:
            while True:
                i = 10
        if self._settings['root_notation'] and e.exp.is_Rational and (e.exp.p == 1):
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('root'))
            if e.exp.q != 2:
                xmldeg = self.dom.createElement('degree')
                xmlcn = self.dom.createElement('cn')
                xmlcn.appendChild(self.dom.createTextNode(str(e.exp.q)))
                xmldeg.appendChild(xmlcn)
                x.appendChild(xmldeg)
            x.appendChild(self._print(e.base))
            return x
        x = self.dom.createElement('apply')
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        x.appendChild(self._print(e.base))
        x.appendChild(self._print(e.exp))
        return x

    def _print_Number(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(self.dom.createTextNode(str(e)))
        return x

    def _print_Float(self, e):
        if False:
            return 10
        x = self.dom.createElement(self.mathml_tag(e))
        repr_e = mlib_to_str(e._mpf_, repr_dps(e._prec))
        x.appendChild(self.dom.createTextNode(repr_e))
        return x

    def _print_Derivative(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('apply')
        diff_symbol = self.mathml_tag(e)
        if requires_partial(e.expr):
            diff_symbol = 'partialdiff'
        x.appendChild(self.dom.createElement(diff_symbol))
        x_1 = self.dom.createElement('bvar')
        for (sym, times) in reversed(e.variable_count):
            x_1.appendChild(self._print(sym))
            if times > 1:
                degree = self.dom.createElement('degree')
                degree.appendChild(self._print(sympify(times)))
                x_1.appendChild(degree)
        x.appendChild(x_1)
        x.appendChild(self._print(e.expr))
        return x

    def _print_Function(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Basic(self, e):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement(self.mathml_tag(e))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_AssocOp(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('apply')
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Relational(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        x.appendChild(self._print(e.lhs))
        x.appendChild(self._print(e.rhs))
        return x

    def _print_list(self, seq):
        if False:
            i = 10
            return i + 15
        'MathML reference for the <list> element:\n        https://www.w3.org/TR/MathML2/chapter4.html#contm.list'
        dom_element = self.dom.createElement('list')
        for item in seq:
            dom_element.appendChild(self._print(item))
        return dom_element

    def _print_int(self, p):
        if False:
            return 10
        dom_element = self.dom.createElement(self.mathml_tag(p))
        dom_element.appendChild(self.dom.createTextNode(str(p)))
        return dom_element
    _print_Implies = _print_AssocOp
    _print_Not = _print_AssocOp
    _print_Xor = _print_AssocOp

    def _print_FiniteSet(self, e):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement('set')
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Complement(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('setdiff'))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_ProductSet(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('cartesianproduct'))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Lambda(self, e):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement(self.mathml_tag(e))
        for arg in e.signature:
            x_1 = self.dom.createElement('bvar')
            x_1.appendChild(self._print(arg))
            x.appendChild(x_1)
        x.appendChild(self._print(e.expr))
        return x

class MathMLPresentationPrinter(MathMLPrinterBase):
    """Prints an expression to the Presentation MathML markup language.

    References: https://www.w3.org/TR/MathML2/chapter3.html
    """
    printmethod = '_mathml_presentation'

    def mathml_tag(self, e):
        if False:
            for i in range(10):
                print('nop')
        'Returns the MathML tag for an expression.'
        translate = {'Number': 'mn', 'Limit': '&#x2192;', 'Derivative': '&dd;', 'int': 'mn', 'Symbol': 'mi', 'Integral': '&int;', 'Sum': '&#x2211;', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'cot': 'cot', 'asin': 'arcsin', 'asinh': 'arcsinh', 'acos': 'arccos', 'acosh': 'arccosh', 'atan': 'arctan', 'atanh': 'arctanh', 'acot': 'arccot', 'atan2': 'arctan', 'Equality': '=', 'Unequality': '&#x2260;', 'GreaterThan': '&#x2265;', 'LessThan': '&#x2264;', 'StrictGreaterThan': '>', 'StrictLessThan': '<', 'lerchphi': '&#x3A6;', 'zeta': '&#x3B6;', 'dirichlet_eta': '&#x3B7;', 'elliptic_k': '&#x39A;', 'lowergamma': '&#x3B3;', 'uppergamma': '&#x393;', 'gamma': '&#x393;', 'totient': '&#x3D5;', 'reduced_totient': '&#x3BB;', 'primenu': '&#x3BD;', 'primeomega': '&#x3A9;', 'fresnels': 'S', 'fresnelc': 'C', 'LambertW': 'W', 'Heaviside': '&#x398;', 'BooleanTrue': 'True', 'BooleanFalse': 'False', 'NoneType': 'None', 'mathieus': 'S', 'mathieuc': 'C', 'mathieusprime': 'S&#x2032;', 'mathieucprime': 'C&#x2032;', 'Lambda': 'lambda'}

        def mul_symbol_selection():
            if False:
                return 10
            if self._settings['mul_symbol'] is None or self._settings['mul_symbol'] == 'None':
                return '&InvisibleTimes;'
            elif self._settings['mul_symbol'] == 'times':
                return '&#xD7;'
            elif self._settings['mul_symbol'] == 'dot':
                return '&#xB7;'
            elif self._settings['mul_symbol'] == 'ldot':
                return '&#x2024;'
            elif not isinstance(self._settings['mul_symbol'], str):
                raise TypeError
            else:
                return self._settings['mul_symbol']
        for cls in e.__class__.__mro__:
            n = cls.__name__
            if n in translate:
                return translate[n]
        if e.__class__.__name__ == 'Mul':
            return mul_symbol_selection()
        n = e.__class__.__name__
        return n.lower()

    def parenthesize(self, item, level, strict=False):
        if False:
            print('Hello World!')
        prec_val = precedence_traditional(item)
        if prec_val < level or (not strict and prec_val <= level):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(item))
            return brac
        else:
            return self._print(item)

    def _print_Mul(self, expr):
        if False:
            while True:
                i = 10

        def multiply(expr, mrow):
            if False:
                print('Hello World!')
            from sympy.simplify import fraction
            (numer, denom) = fraction(expr)
            if denom is not S.One:
                frac = self.dom.createElement('mfrac')
                if self._settings['fold_short_frac'] and len(str(expr)) < 7:
                    frac.setAttribute('bevelled', 'true')
                xnum = self._print(numer)
                xden = self._print(denom)
                frac.appendChild(xnum)
                frac.appendChild(xden)
                mrow.appendChild(frac)
                return mrow
            (coeff, terms) = expr.as_coeff_mul()
            if coeff is S.One and len(terms) == 1:
                mrow.appendChild(self._print(terms[0]))
                return mrow
            if self.order != 'old':
                terms = Mul._from_args(terms).as_ordered_factors()
            if coeff != 1:
                x = self._print(coeff)
                y = self.dom.createElement('mo')
                y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                mrow.appendChild(x)
                mrow.appendChild(y)
            for term in terms:
                mrow.appendChild(self.parenthesize(term, PRECEDENCE['Mul']))
                if not term == terms[-1]:
                    y = self.dom.createElement('mo')
                    y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                    mrow.appendChild(y)
            return mrow
        mrow = self.dom.createElement('mrow')
        if expr.could_extract_minus_sign():
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode('-'))
            mrow.appendChild(x)
            mrow = multiply(-expr, mrow)
        else:
            mrow = multiply(expr, mrow)
        return mrow

    def _print_Add(self, expr, order=None):
        if False:
            for i in range(10):
                print('nop')
        mrow = self.dom.createElement('mrow')
        args = self._as_ordered_terms(expr, order=order)
        mrow.appendChild(self._print(args[0]))
        for arg in args[1:]:
            if arg.could_extract_minus_sign():
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('-'))
                y = self._print(-arg)
            else:
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('+'))
                y = self._print(arg)
            mrow.appendChild(x)
            mrow.appendChild(y)
        return mrow

    def _print_MatrixBase(self, m):
        if False:
            i = 10
            return i + 15
        table = self.dom.createElement('mtable')
        for i in range(m.rows):
            x = self.dom.createElement('mtr')
            for j in range(m.cols):
                y = self.dom.createElement('mtd')
                y.appendChild(self._print(m[i, j]))
                x.appendChild(y)
            table.appendChild(x)
        if self._settings['mat_delim'] == '':
            return table
        brac = self.dom.createElement('mfenced')
        if self._settings['mat_delim'] == '[':
            brac.setAttribute('close', ']')
            brac.setAttribute('open', '[')
        brac.appendChild(table)
        return brac

    def _get_printed_Rational(self, e, folded=None):
        if False:
            print('Hello World!')
        if e.p < 0:
            p = -e.p
        else:
            p = e.p
        x = self.dom.createElement('mfrac')
        if folded or self._settings['fold_short_frac']:
            x.setAttribute('bevelled', 'true')
        x.appendChild(self._print(p))
        x.appendChild(self._print(e.q))
        if e.p < 0:
            mrow = self.dom.createElement('mrow')
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('-'))
            mrow.appendChild(mo)
            mrow.appendChild(x)
            return mrow
        else:
            return x

    def _print_Rational(self, e):
        if False:
            return 10
        if e.q == 1:
            return self._print(e.p)
        return self._get_printed_Rational(e, self._settings['fold_short_frac'])

    def _print_Limit(self, e):
        if False:
            for i in range(10):
                print('nop')
        mrow = self.dom.createElement('mrow')
        munder = self.dom.createElement('munder')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('lim'))
        x = self.dom.createElement('mrow')
        x_1 = self._print(e.args[1])
        arrow = self.dom.createElement('mo')
        arrow.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        x_2 = self._print(e.args[2])
        x.appendChild(x_1)
        x.appendChild(arrow)
        x.appendChild(x_2)
        munder.appendChild(mi)
        munder.appendChild(x)
        mrow.appendChild(munder)
        mrow.appendChild(self._print(e.args[0]))
        return mrow

    def _print_ImaginaryUnit(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&ImaginaryI;'))
        return x

    def _print_GoldenRatio(self, e):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x3A6;'))
        return x

    def _print_Exp1(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&ExponentialE;'))
        return x

    def _print_Pi(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&pi;'))
        return x

    def _print_Infinity(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x221E;'))
        return x

    def _print_NegativeInfinity(self, e):
        if False:
            return 10
        mrow = self.dom.createElement('mrow')
        y = self.dom.createElement('mo')
        y.appendChild(self.dom.createTextNode('-'))
        x = self._print_Infinity(e)
        mrow.appendChild(y)
        mrow.appendChild(x)
        return mrow

    def _print_HBar(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x210F;'))
        return x

    def _print_EulerGamma(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x3B3;'))
        return x

    def _print_TribonacciConstant(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('TribonacciConstant'))
        return x

    def _print_Dagger(self, e):
        if False:
            for i in range(10):
                print('nop')
        msup = self.dom.createElement('msup')
        msup.appendChild(self._print(e.args[0]))
        msup.appendChild(self.dom.createTextNode('&#x2020;'))
        return msup

    def _print_Contains(self, e):
        if False:
            i = 10
            return i + 15
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._print(e.args[0]))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2208;'))
        mrow.appendChild(mo)
        mrow.appendChild(self._print(e.args[1]))
        return mrow

    def _print_HilbertSpace(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x210B;'))
        return x

    def _print_ComplexSpace(self, e):
        if False:
            for i in range(10):
                print('nop')
        msup = self.dom.createElement('msup')
        msup.appendChild(self.dom.createTextNode('&#x1D49E;'))
        msup.appendChild(self._print(e.args[0]))
        return msup

    def _print_FockSpace(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x2131;'))
        return x

    def _print_Integral(self, expr):
        if False:
            for i in range(10):
                print('nop')
        intsymbols = {1: '&#x222B;', 2: '&#x222C;', 3: '&#x222D;'}
        mrow = self.dom.createElement('mrow')
        if len(expr.limits) <= 3 and all((len(lim) == 1 for lim in expr.limits)):
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode(intsymbols[len(expr.limits)]))
            mrow.appendChild(mo)
        else:
            for lim in reversed(expr.limits):
                mo = self.dom.createElement('mo')
                mo.appendChild(self.dom.createTextNode(intsymbols[1]))
                if len(lim) == 1:
                    mrow.appendChild(mo)
                if len(lim) == 2:
                    msup = self.dom.createElement('msup')
                    msup.appendChild(mo)
                    msup.appendChild(self._print(lim[1]))
                    mrow.appendChild(msup)
                if len(lim) == 3:
                    msubsup = self.dom.createElement('msubsup')
                    msubsup.appendChild(mo)
                    msubsup.appendChild(self._print(lim[1]))
                    msubsup.appendChild(self._print(lim[2]))
                    mrow.appendChild(msubsup)
        mrow.appendChild(self.parenthesize(expr.function, PRECEDENCE['Mul'], strict=True))
        for lim in reversed(expr.limits):
            d = self.dom.createElement('mo')
            d.appendChild(self.dom.createTextNode('&dd;'))
            mrow.appendChild(d)
            mrow.appendChild(self._print(lim[0]))
        return mrow

    def _print_Sum(self, e):
        if False:
            i = 10
            return i + 15
        limits = list(e.limits)
        subsup = self.dom.createElement('munderover')
        low_elem = self._print(limits[0][1])
        up_elem = self._print(limits[0][2])
        summand = self.dom.createElement('mo')
        summand.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        low = self.dom.createElement('mrow')
        var = self._print(limits[0][0])
        equal = self.dom.createElement('mo')
        equal.appendChild(self.dom.createTextNode('='))
        low.appendChild(var)
        low.appendChild(equal)
        low.appendChild(low_elem)
        subsup.appendChild(summand)
        subsup.appendChild(low)
        subsup.appendChild(up_elem)
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(subsup)
        if len(str(e.function)) == 1:
            mrow.appendChild(self._print(e.function))
        else:
            fence = self.dom.createElement('mfenced')
            fence.appendChild(self._print(e.function))
            mrow.appendChild(fence)
        return mrow

    def _print_Symbol(self, sym, style='plain'):
        if False:
            for i in range(10):
                print('nop')

        def join(items):
            if False:
                i = 10
                return i + 15
            if len(items) > 1:
                mrow = self.dom.createElement('mrow')
                for (i, item) in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode(' '))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            else:
                mi = self.dom.createElement('mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi

        def translate(s):
            if False:
                i = 10
                return i + 15
            if s in greek_unicode:
                return greek_unicode.get(s)
            else:
                return s
        (name, supers, subs) = split_super_sub(sym.name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]
        mname = self.dom.createElement('mi')
        mname.appendChild(self.dom.createTextNode(name))
        if len(supers) == 0:
            if len(subs) == 0:
                x = mname
            else:
                x = self.dom.createElement('msub')
                x.appendChild(mname)
                x.appendChild(join(subs))
        elif len(subs) == 0:
            x = self.dom.createElement('msup')
            x.appendChild(mname)
            x.appendChild(join(supers))
        else:
            x = self.dom.createElement('msubsup')
            x.appendChild(mname)
            x.appendChild(join(subs))
            x.appendChild(join(supers))
        if style == 'bold':
            x.setAttribute('mathvariant', 'bold')
        return x

    def _print_MatrixSymbol(self, sym):
        if False:
            i = 10
            return i + 15
        return self._print_Symbol(sym, style=self._settings['mat_symbol_style'])
    _print_RandomSymbol = _print_Symbol

    def _print_conjugate(self, expr):
        if False:
            while True:
                i = 10
        enc = self.dom.createElement('menclose')
        enc.setAttribute('notation', 'top')
        enc.appendChild(self._print(expr.args[0]))
        return enc

    def _print_operator_after(self, op, expr):
        if False:
            print('Hello World!')
        row = self.dom.createElement('mrow')
        row.appendChild(self.parenthesize(expr, PRECEDENCE['Func']))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(op))
        row.appendChild(mo)
        return row

    def _print_factorial(self, expr):
        if False:
            while True:
                i = 10
        return self._print_operator_after('!', expr.args[0])

    def _print_factorial2(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return self._print_operator_after('!!', expr.args[0])

    def _print_binomial(self, expr):
        if False:
            while True:
                i = 10
        brac = self.dom.createElement('mfenced')
        frac = self.dom.createElement('mfrac')
        frac.setAttribute('linethickness', '0')
        frac.appendChild(self._print(expr.args[0]))
        frac.appendChild(self._print(expr.args[1]))
        brac.appendChild(frac)
        return brac

    def _print_Pow(self, e):
        if False:
            return 10
        if e.exp.is_Rational and abs(e.exp.p) == 1 and (e.exp.q != 1) and self._settings['root_notation']:
            if e.exp.q == 2:
                x = self.dom.createElement('msqrt')
                x.appendChild(self._print(e.base))
            if e.exp.q != 2:
                x = self.dom.createElement('mroot')
                x.appendChild(self._print(e.base))
                x.appendChild(self._print(e.exp.q))
            if e.exp.p == -1:
                frac = self.dom.createElement('mfrac')
                frac.appendChild(self._print(1))
                frac.appendChild(x)
                return frac
            else:
                return x
        if e.exp.is_Rational and e.exp.q != 1:
            if e.exp.is_negative:
                top = self.dom.createElement('mfrac')
                top.appendChild(self._print(1))
                x = self.dom.createElement('msup')
                x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                x.appendChild(self._get_printed_Rational(-e.exp, self._settings['fold_frac_powers']))
                top.appendChild(x)
                return top
            else:
                x = self.dom.createElement('msup')
                x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                x.appendChild(self._get_printed_Rational(e.exp, self._settings['fold_frac_powers']))
                return x
        if e.exp.is_negative:
            top = self.dom.createElement('mfrac')
            top.appendChild(self._print(1))
            if e.exp == -1:
                top.appendChild(self._print(e.base))
            else:
                x = self.dom.createElement('msup')
                x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                x.appendChild(self._print(-e.exp))
                top.appendChild(x)
            return top
        x = self.dom.createElement('msup')
        x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
        x.appendChild(self._print(e.exp))
        return x

    def _print_Number(self, e):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(self.dom.createTextNode(str(e)))
        return x

    def _print_AccumulationBounds(self, i):
        if False:
            return 10
        brac = self.dom.createElement('mfenced')
        brac.setAttribute('close', '⟩')
        brac.setAttribute('open', '⟨')
        brac.appendChild(self._print(i.min))
        brac.appendChild(self._print(i.max))
        return brac

    def _print_Derivative(self, e):
        if False:
            print('Hello World!')
        if requires_partial(e.expr):
            d = '&#x2202;'
        else:
            d = self.mathml_tag(e)
        m = self.dom.createElement('mrow')
        dim = 0
        for (sym, num) in reversed(e.variable_count):
            dim += num
            if num >= 2:
                x = self.dom.createElement('msup')
                xx = self.dom.createElement('mo')
                xx.appendChild(self.dom.createTextNode(d))
                x.appendChild(xx)
                x.appendChild(self._print(num))
            else:
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode(d))
            m.appendChild(x)
            y = self._print(sym)
            m.appendChild(y)
        mnum = self.dom.createElement('mrow')
        if dim >= 2:
            x = self.dom.createElement('msup')
            xx = self.dom.createElement('mo')
            xx.appendChild(self.dom.createTextNode(d))
            x.appendChild(xx)
            x.appendChild(self._print(dim))
        else:
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(d))
        mnum.appendChild(x)
        mrow = self.dom.createElement('mrow')
        frac = self.dom.createElement('mfrac')
        frac.appendChild(mnum)
        frac.appendChild(m)
        mrow.appendChild(frac)
        mrow.appendChild(self._print(e.expr))
        return mrow

    def _print_Function(self, e):
        if False:
            return 10
        mrow = self.dom.createElement('mrow')
        x = self.dom.createElement('mi')
        if self.mathml_tag(e) == 'log' and self._settings['ln_notation']:
            x.appendChild(self.dom.createTextNode('ln'))
        else:
            x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        y = self.dom.createElement('mfenced')
        for arg in e.args:
            y.appendChild(self._print(arg))
        mrow.appendChild(x)
        mrow.appendChild(y)
        return mrow

    def _print_Float(self, expr):
        if False:
            for i in range(10):
                print('nop')
        dps = prec_to_dps(expr._prec)
        str_real = mlib_to_str(expr._mpf_, dps, strip_zeros=True)
        separator = self._settings['mul_symbol_mathml_numbers']
        mrow = self.dom.createElement('mrow')
        if 'e' in str_real:
            (mant, exp) = str_real.split('e')
            if exp[0] == '+':
                exp = exp[1:]
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(mant))
            mrow.appendChild(mn)
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode(separator))
            mrow.appendChild(mo)
            msup = self.dom.createElement('msup')
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode('10'))
            msup.appendChild(mn)
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(exp))
            msup.appendChild(mn)
            mrow.appendChild(msup)
            return mrow
        elif str_real == '+inf':
            return self._print_Infinity(None)
        elif str_real == '-inf':
            return self._print_NegativeInfinity(None)
        else:
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(str_real))
            return mn

    def _print_polylog(self, expr):
        if False:
            while True:
                i = 10
        mrow = self.dom.createElement('mrow')
        m = self.dom.createElement('msub')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('Li'))
        m.appendChild(mi)
        m.appendChild(self._print(expr.args[0]))
        mrow.appendChild(m)
        brac = self.dom.createElement('mfenced')
        brac.appendChild(self._print(expr.args[1]))
        mrow.appendChild(brac)
        return mrow

    def _print_Basic(self, e):
        if False:
            i = 10
            return i + 15
        mrow = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(mi)
        brac = self.dom.createElement('mfenced')
        for arg in e.args:
            brac.appendChild(self._print(arg))
        mrow.appendChild(brac)
        return mrow

    def _print_Tuple(self, e):
        if False:
            i = 10
            return i + 15
        mrow = self.dom.createElement('mrow')
        x = self.dom.createElement('mfenced')
        for arg in e.args:
            x.appendChild(self._print(arg))
        mrow.appendChild(x)
        return mrow

    def _print_Interval(self, i):
        if False:
            while True:
                i = 10
        mrow = self.dom.createElement('mrow')
        brac = self.dom.createElement('mfenced')
        if i.start == i.end:
            brac.setAttribute('close', '}')
            brac.setAttribute('open', '{')
            brac.appendChild(self._print(i.start))
        else:
            if i.right_open:
                brac.setAttribute('close', ')')
            else:
                brac.setAttribute('close', ']')
            if i.left_open:
                brac.setAttribute('open', '(')
            else:
                brac.setAttribute('open', '[')
            brac.appendChild(self._print(i.start))
            brac.appendChild(self._print(i.end))
        mrow.appendChild(brac)
        return mrow

    def _print_Abs(self, expr, exp=None):
        if False:
            for i in range(10):
                print('nop')
        mrow = self.dom.createElement('mrow')
        x = self.dom.createElement('mfenced')
        x.setAttribute('close', '|')
        x.setAttribute('open', '|')
        x.appendChild(self._print(expr.args[0]))
        mrow.appendChild(x)
        return mrow
    _print_Determinant = _print_Abs

    def _print_re_im(self, c, expr):
        if False:
            print('Hello World!')
        mrow = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'fraktur')
        mi.appendChild(self.dom.createTextNode(c))
        mrow.appendChild(mi)
        brac = self.dom.createElement('mfenced')
        brac.appendChild(self._print(expr))
        mrow.appendChild(brac)
        return mrow

    def _print_re(self, expr, exp=None):
        if False:
            return 10
        return self._print_re_im('R', expr.args[0])

    def _print_im(self, expr, exp=None):
        if False:
            i = 10
            return i + 15
        return self._print_re_im('I', expr.args[0])

    def _print_AssocOp(self, e):
        if False:
            while True:
                i = 10
        mrow = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(mi)
        for arg in e.args:
            mrow.appendChild(self._print(arg))
        return mrow

    def _print_SetOp(self, expr, symbol, prec):
        if False:
            print('Hello World!')
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self.parenthesize(expr.args[0], prec))
        for arg in expr.args[1:]:
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(symbol))
            y = self.parenthesize(arg, prec)
            mrow.appendChild(x)
            mrow.appendChild(y)
        return mrow

    def _print_Union(self, expr):
        if False:
            return 10
        prec = PRECEDENCE_TRADITIONAL['Union']
        return self._print_SetOp(expr, '&#x222A;', prec)

    def _print_Intersection(self, expr):
        if False:
            for i in range(10):
                print('nop')
        prec = PRECEDENCE_TRADITIONAL['Intersection']
        return self._print_SetOp(expr, '&#x2229;', prec)

    def _print_Complement(self, expr):
        if False:
            for i in range(10):
                print('nop')
        prec = PRECEDENCE_TRADITIONAL['Complement']
        return self._print_SetOp(expr, '&#x2216;', prec)

    def _print_SymmetricDifference(self, expr):
        if False:
            return 10
        prec = PRECEDENCE_TRADITIONAL['SymmetricDifference']
        return self._print_SetOp(expr, '&#x2206;', prec)

    def _print_ProductSet(self, expr):
        if False:
            while True:
                i = 10
        prec = PRECEDENCE_TRADITIONAL['ProductSet']
        return self._print_SetOp(expr, '&#x00d7;', prec)

    def _print_FiniteSet(self, s):
        if False:
            while True:
                i = 10
        return self._print_set(s.args)

    def _print_set(self, s):
        if False:
            for i in range(10):
                print('nop')
        items = sorted(s, key=default_sort_key)
        brac = self.dom.createElement('mfenced')
        brac.setAttribute('close', '}')
        brac.setAttribute('open', '{')
        for item in items:
            brac.appendChild(self._print(item))
        return brac
    _print_frozenset = _print_set

    def _print_LogOp(self, args, symbol):
        if False:
            while True:
                i = 10
        mrow = self.dom.createElement('mrow')
        if args[0].is_Boolean and (not args[0].is_Not):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(args[0]))
            mrow.appendChild(brac)
        else:
            mrow.appendChild(self._print(args[0]))
        for arg in args[1:]:
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(symbol))
            if arg.is_Boolean and (not arg.is_Not):
                y = self.dom.createElement('mfenced')
                y.appendChild(self._print(arg))
            else:
                y = self._print(arg)
            mrow.appendChild(x)
            mrow.appendChild(y)
        return mrow

    def _print_BasisDependent(self, expr):
        if False:
            return 10
        from sympy.vector import Vector
        if expr == expr.zero:
            return self._print(expr.zero)
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]
        mrow = self.dom.createElement('mrow')
        for (system, vect) in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key=lambda x: x[0].__str__())
            for (i, (k, v)) in enumerate(inneritems):
                if v == 1:
                    if i:
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode('+'))
                        mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))
                elif v == -1:
                    mo = self.dom.createElement('mo')
                    mo.appendChild(self.dom.createTextNode('-'))
                    mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))
                else:
                    if i:
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode('+'))
                        mrow.appendChild(mo)
                    mbrac = self.dom.createElement('mfenced')
                    mbrac.appendChild(self._print(v))
                    mrow.appendChild(mbrac)
                    mo = self.dom.createElement('mo')
                    mo.appendChild(self.dom.createTextNode('&InvisibleTimes;'))
                    mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))
        return mrow

    def _print_And(self, expr):
        if False:
            for i in range(10):
                print('nop')
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x2227;')

    def _print_Or(self, expr):
        if False:
            print('Hello World!')
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x2228;')

    def _print_Xor(self, expr):
        if False:
            for i in range(10):
                print('nop')
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x22BB;')

    def _print_Implies(self, expr):
        if False:
            i = 10
            return i + 15
        return self._print_LogOp(expr.args, '&#x21D2;')

    def _print_Equivalent(self, expr):
        if False:
            print('Hello World!')
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x21D4;')

    def _print_Not(self, e):
        if False:
            while True:
                i = 10
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xAC;'))
        mrow.appendChild(mo)
        if e.args[0].is_Boolean:
            x = self.dom.createElement('mfenced')
            x.appendChild(self._print(e.args[0]))
        else:
            x = self._print(e.args[0])
        mrow.appendChild(x)
        return mrow

    def _print_bool(self, e):
        if False:
            for i in range(10):
                print('nop')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        return mi
    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        if False:
            return 10
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        return mi

    def _print_Range(self, s):
        if False:
            for i in range(10):
                print('nop')
        dots = '…'
        brac = self.dom.createElement('mfenced')
        brac.setAttribute('close', '}')
        brac.setAttribute('open', '{')
        if s.start.is_infinite and s.stop.is_infinite:
            if s.step.is_positive:
                printset = (dots, -1, 0, 1, dots)
            else:
                printset = (dots, 1, 0, -1, dots)
        elif s.start.is_infinite:
            printset = (dots, s[-1] - s.step, s[-1])
        elif s.stop.is_infinite:
            it = iter(s)
            printset = (next(it), next(it), dots)
        elif len(s) > 4:
            it = iter(s)
            printset = (next(it), next(it), dots, s[-1])
        else:
            printset = tuple(s)
        for el in printset:
            if el == dots:
                mi = self.dom.createElement('mi')
                mi.appendChild(self.dom.createTextNode(dots))
                brac.appendChild(mi)
            else:
                brac.appendChild(self._print(el))
        return brac

    def _hprint_variadic_function(self, expr):
        if False:
            i = 10
            return i + 15
        args = sorted(expr.args, key=default_sort_key)
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(str(expr.func).lower()))
        mrow.appendChild(mo)
        brac = self.dom.createElement('mfenced')
        for symbol in args:
            brac.appendChild(self._print(symbol))
        mrow.appendChild(brac)
        return mrow
    _print_Min = _print_Max = _hprint_variadic_function

    def _print_exp(self, expr):
        if False:
            while True:
                i = 10
        msup = self.dom.createElement('msup')
        msup.appendChild(self._print_Exp1(None))
        msup.appendChild(self._print(expr.args[0]))
        return msup

    def _print_Relational(self, e):
        if False:
            for i in range(10):
                print('nop')
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._print(e.lhs))
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(x)
        mrow.appendChild(self._print(e.rhs))
        return mrow

    def _print_int(self, p):
        if False:
            for i in range(10):
                print('nop')
        dom_element = self.dom.createElement(self.mathml_tag(p))
        dom_element.appendChild(self.dom.createTextNode(str(p)))
        return dom_element

    def _print_BaseScalar(self, e):
        if False:
            return 10
        msub = self.dom.createElement('msub')
        (index, system) = e._id
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._variable_names[index]))
        msub.appendChild(mi)
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._name))
        msub.appendChild(mi)
        return msub

    def _print_BaseVector(self, e):
        if False:
            i = 10
            return i + 15
        msub = self.dom.createElement('msub')
        (index, system) = e._id
        mover = self.dom.createElement('mover')
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._vector_names[index]))
        mover.appendChild(mi)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('^'))
        mover.appendChild(mo)
        msub.appendChild(mover)
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._name))
        msub.appendChild(mi)
        return msub

    def _print_VectorZero(self, e):
        if False:
            while True:
                i = 10
        mover = self.dom.createElement('mover')
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode('0'))
        mover.appendChild(mi)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('^'))
        mover.appendChild(mo)
        return mover

    def _print_Cross(self, expr):
        if False:
            while True:
                i = 10
        mrow = self.dom.createElement('mrow')
        vec1 = expr._expr1
        vec2 = expr._expr2
        mrow.appendChild(self.parenthesize(vec1, PRECEDENCE['Mul']))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xD7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(vec2, PRECEDENCE['Mul']))
        return mrow

    def _print_Curl(self, expr):
        if False:
            while True:
                i = 10
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xD7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Divergence(self, expr):
        if False:
            for i in range(10):
                print('nop')
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xB7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Dot(self, expr):
        if False:
            for i in range(10):
                print('nop')
        mrow = self.dom.createElement('mrow')
        vec1 = expr._expr1
        vec2 = expr._expr2
        mrow.appendChild(self.parenthesize(vec1, PRECEDENCE['Mul']))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xB7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(vec2, PRECEDENCE['Mul']))
        return mrow

    def _print_Gradient(self, expr):
        if False:
            i = 10
            return i + 15
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Laplacian(self, expr):
        if False:
            for i in range(10):
                print('nop')
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2206;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Integers(self, e):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2124;'))
        return x

    def _print_Complexes(self, e):
        if False:
            return 10
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2102;'))
        return x

    def _print_Reals(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x211D;'))
        return x

    def _print_Naturals(self, e):
        if False:
            return 10
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2115;'))
        return x

    def _print_Naturals0(self, e):
        if False:
            for i in range(10):
                print('nop')
        sub = self.dom.createElement('msub')
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2115;'))
        sub.appendChild(x)
        sub.appendChild(self._print(S.Zero))
        return sub

    def _print_SingularityFunction(self, expr):
        if False:
            while True:
                i = 10
        shift = expr.args[0] - expr.args[1]
        power = expr.args[2]
        sup = self.dom.createElement('msup')
        brac = self.dom.createElement('mfenced')
        brac.setAttribute('close', '⟩')
        brac.setAttribute('open', '⟨')
        brac.appendChild(self._print(shift))
        sup.appendChild(brac)
        sup.appendChild(self._print(power))
        return sup

    def _print_NaN(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('NaN'))
        return x

    def _print_number_function(self, e, name):
        if False:
            while True:
                i = 10
        sub = self.dom.createElement('msub')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(name))
        sub.appendChild(mi)
        sub.appendChild(self._print(e.args[0]))
        if len(e.args) == 1:
            return sub
        mrow = self.dom.createElement('mrow')
        y = self.dom.createElement('mfenced')
        for arg in e.args[1:]:
            y.appendChild(self._print(arg))
        mrow.appendChild(sub)
        mrow.appendChild(y)
        return mrow

    def _print_bernoulli(self, e):
        if False:
            while True:
                i = 10
        return self._print_number_function(e, 'B')
    _print_bell = _print_bernoulli

    def _print_catalan(self, e):
        if False:
            while True:
                i = 10
        return self._print_number_function(e, 'C')

    def _print_euler(self, e):
        if False:
            i = 10
            return i + 15
        return self._print_number_function(e, 'E')

    def _print_fibonacci(self, e):
        if False:
            print('Hello World!')
        return self._print_number_function(e, 'F')

    def _print_lucas(self, e):
        if False:
            for i in range(10):
                print('nop')
        return self._print_number_function(e, 'L')

    def _print_stieltjes(self, e):
        if False:
            return 10
        return self._print_number_function(e, '&#x03B3;')

    def _print_tribonacci(self, e):
        if False:
            while True:
                i = 10
        return self._print_number_function(e, 'T')

    def _print_ComplexInfinity(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('mover')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x221E;'))
        x.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('~'))
        x.appendChild(mo)
        return x

    def _print_EmptySet(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode('&#x2205;'))
        return x

    def _print_UniversalSet(self, e):
        if False:
            return 10
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode('&#x1D54C;'))
        return x

    def _print_Adjoint(self, expr):
        if False:
            i = 10
            return i + 15
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(mat))
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2020;'))
        sup.appendChild(mo)
        return sup

    def _print_Transpose(self, expr):
        if False:
            while True:
                i = 10
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(mat))
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('T'))
        sup.appendChild(mo)
        return sup

    def _print_Inverse(self, expr):
        if False:
            return 10
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(mat))
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        sup.appendChild(self._print(-1))
        return sup

    def _print_MatMul(self, expr):
        if False:
            for i in range(10):
                print('nop')
        from sympy.matrices.expressions.matmul import MatMul
        x = self.dom.createElement('mrow')
        args = expr.args
        if isinstance(args[0], Mul):
            args = args[0].as_ordered_factors() + list(args[1:])
        else:
            args = list(args)
        if isinstance(expr, MatMul) and expr.could_extract_minus_sign():
            if args[0] == -1:
                args = args[1:]
            else:
                args[0] = -args[0]
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('-'))
            x.appendChild(mo)
        for arg in args[:-1]:
            x.appendChild(self.parenthesize(arg, precedence_traditional(expr), False))
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('&InvisibleTimes;'))
            x.appendChild(mo)
        x.appendChild(self.parenthesize(args[-1], precedence_traditional(expr), False))
        return x

    def _print_MatPow(self, expr):
        if False:
            while True:
                i = 10
        from sympy.matrices import MatrixSymbol
        (base, exp) = (expr.base, expr.exp)
        sup = self.dom.createElement('msup')
        if not isinstance(base, MatrixSymbol):
            brac = self.dom.createElement('mfenced')
            brac.appendChild(self._print(base))
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(base))
        sup.appendChild(self._print(exp))
        return sup

    def _print_HadamardProduct(self, expr):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement('mrow')
        args = expr.args
        for arg in args[:-1]:
            x.appendChild(self.parenthesize(arg, precedence_traditional(expr), False))
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('&#x2218;'))
            x.appendChild(mo)
        x.appendChild(self.parenthesize(args[-1], precedence_traditional(expr), False))
        return x

    def _print_ZeroMatrix(self, Z):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('mn')
        x.appendChild(self.dom.createTextNode('&#x1D7D8'))
        return x

    def _print_OneMatrix(self, Z):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement('mn')
        x.appendChild(self.dom.createTextNode('&#x1D7D9'))
        return x

    def _print_Identity(self, I):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x1D540;'))
        return x

    def _print_floor(self, e):
        if False:
            return 10
        mrow = self.dom.createElement('mrow')
        x = self.dom.createElement('mfenced')
        x.setAttribute('close', '⌋')
        x.setAttribute('open', '⌊')
        x.appendChild(self._print(e.args[0]))
        mrow.appendChild(x)
        return mrow

    def _print_ceiling(self, e):
        if False:
            return 10
        mrow = self.dom.createElement('mrow')
        x = self.dom.createElement('mfenced')
        x.setAttribute('close', '⌉')
        x.setAttribute('open', '⌈')
        x.appendChild(self._print(e.args[0]))
        mrow.appendChild(x)
        return mrow

    def _print_Lambda(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('mfenced')
        mrow = self.dom.createElement('mrow')
        symbols = e.args[0]
        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(symbols)
        mrow.appendChild(symbols)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x21A6;'))
        mrow.appendChild(mo)
        mrow.appendChild(self._print(e.args[1]))
        x.appendChild(mrow)
        return x

    def _print_tuple(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('mfenced')
        for i in e:
            x.appendChild(self._print(i))
        return x

    def _print_IndexedBase(self, e):
        if False:
            return 10
        return self._print(e.label)

    def _print_Indexed(self, e):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement('msub')
        x.appendChild(self._print(e.base))
        if len(e.indices) == 1:
            x.appendChild(self._print(e.indices[0]))
            return x
        x.appendChild(self._print(e.indices))
        return x

    def _print_MatrixElement(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('msub')
        x.appendChild(self.parenthesize(e.parent, PRECEDENCE['Atom'], strict=True))
        brac = self.dom.createElement('mfenced')
        brac.setAttribute('close', '')
        brac.setAttribute('open', '')
        for i in e.indices:
            brac.appendChild(self._print(i))
        x.appendChild(brac)
        return x

    def _print_elliptic_f(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d5a5;'))
        x.appendChild(mi)
        y = self.dom.createElement('mfenced')
        y.setAttribute('separators', '|')
        for i in e.args:
            y.appendChild(self._print(i))
        x.appendChild(y)
        return x

    def _print_elliptic_e(self, e):
        if False:
            return 10
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d5a4;'))
        x.appendChild(mi)
        y = self.dom.createElement('mfenced')
        y.setAttribute('separators', '|')
        for i in e.args:
            y.appendChild(self._print(i))
        x.appendChild(y)
        return x

    def _print_elliptic_pi(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d6f1;'))
        x.appendChild(mi)
        y = self.dom.createElement('mfenced')
        if len(e.args) == 2:
            y.setAttribute('separators', '|')
        else:
            y.setAttribute('separators', ';|')
        for i in e.args:
            y.appendChild(self._print(i))
        x.appendChild(y)
        return x

    def _print_Ei(self, e):
        if False:
            return 10
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('Ei'))
        x.appendChild(mi)
        x.appendChild(self._print(e.args))
        return x

    def _print_expint(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('E'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_jacobi(self, e):
        if False:
            return 10
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:3]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[3:]))
        return x

    def _print_gegenbauer(self, e):
        if False:
            return 10
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('C'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:2]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[2:]))
        return x

    def _print_chebyshevt(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('T'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_chebyshevu(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('U'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_legendre(self, e):
        if False:
            i = 10
            return i + 15
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_assoc_legendre(self, e):
        if False:
            print('Hello World!')
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:2]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[2:]))
        return x

    def _print_laguerre(self, e):
        if False:
            for i in range(10):
                print('nop')
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('L'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_assoc_laguerre(self, e):
        if False:
            return 10
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('L'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:2]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[2:]))
        return x

    def _print_hermite(self, e):
        if False:
            while True:
                i = 10
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('H'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

@print_function(MathMLPrinterBase)
def mathml(expr, printer='content', **settings):
    if False:
        i = 10
        return i + 15
    'Returns the MathML representation of expr. If printer is presentation\n    then prints Presentation MathML else prints content MathML.\n    '
    if printer == 'presentation':
        return MathMLPresentationPrinter(settings).doprint(expr)
    else:
        return MathMLContentPrinter(settings).doprint(expr)

def print_mathml(expr, printer='content', **settings):
    if False:
        print('Hello World!')
    "\n    Prints a pretty representation of the MathML code for expr. If printer is\n    presentation then prints Presentation MathML else prints content MathML.\n\n    Examples\n    ========\n\n    >>> ##\n    >>> from sympy import print_mathml\n    >>> from sympy.abc import x\n    >>> print_mathml(x+1) #doctest: +NORMALIZE_WHITESPACE\n    <apply>\n        <plus/>\n        <ci>x</ci>\n        <cn>1</cn>\n    </apply>\n    >>> print_mathml(x+1, printer='presentation')\n    <mrow>\n        <mi>x</mi>\n        <mo>+</mo>\n        <mn>1</mn>\n    </mrow>\n\n    "
    if printer == 'presentation':
        s = MathMLPresentationPrinter(settings)
    else:
        s = MathMLContentPrinter(settings)
    xml = s._print(sympify(expr))
    pretty_xml = xml.toprettyxml()
    print(pretty_xml)
MathMLPrinter = MathMLContentPrinter