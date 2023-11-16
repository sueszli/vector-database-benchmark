"""
The language that shall be used in a type 4 function contains expressions involving integers, real numbers, and
boolean values only. There shall be no composite data structures such as strings or arrays, no procedures, and
no variables or names. Table 42 lists the operators that can be used in this type of function. (For more
information on these operators, see Appendix B of the PostScript Language Reference, Third Edition.)
Although the semantics are those of the corresponding PostScript operators, a full PostScript interpreter is not
required.
"""
import typing
from decimal import Decimal
from math import atan
from math import ceil
from math import cos
from math import degrees
from math import exp
from math import floor
from math import log
from math import sin
from math import sqrt

class PostScriptEval:
    """
    The language that shall be used in a type 4 function contains expressions involving integers, real numbers, and
    boolean values only. There shall be no composite data structures such as strings or arrays, no procedures, and
    no variables or names. Table 42 lists the operators that can be used in this type of function. (For more
    information on these operators, see Appendix B of the PostScript Language Reference, Third Edition.)
    Although the semantics are those of the corresponding PostScript operators, a full PostScript interpreter is not
    required.
    """

    @staticmethod
    def evaluate(s: str, args: typing.List[Decimal]) -> typing.List[Decimal]:
        if False:
            while True:
                i = 10
        '\n        This function evaluates a postscript str, using args as the (initial) stack.\n        This function returns a typing.List[Decimal], or throws an assertion error\n        '
        stk: typing.List[typing.Union[Decimal, bool]] = []
        stk += args
        known_operators: typing.List[str] = ['abs', 'add', 'and', 'atan', 'bitshift', 'ceiling', 'copy', 'cos', 'cvi', 'cvr', 'div', 'dup', 'eq', 'exch', 'exp', 'false', 'floor', 'ge', 'gt', 'idiv', 'index', 'le', 'ln', 'log', 'lt', 'mod', 'mul', 'ne', 'neg', 'not', 'or', 'pop', 'roll', 'round', 'sin', 'sqrt', 'sub', 'true', 'truncate', 'xor']
        i: int = 0
        while i < len(s):
            if s[i] in ' \n\t':
                i += 1
                continue
            if s[i] == '{' or s[i] == '}':
                i += 1
                continue
            if s[i] in '0123456789.-':
                operand: str = ''
                while i < len(s) and s[i] in '0123456789.-':
                    operand += s[i]
                    i += 1
                stk.append(Decimal(operand))
                continue
            if any([x.startswith(s[i]) for x in known_operators]):
                operator: str = ''
                while i < len(s) and s[i] in 'abcdefghijklmnopqrstuvwxyz':
                    operator += s[i]
                    i += 1
                if operator not in known_operators:
                    assert False, 'Unknown operator %s in postscript str' % operator
                arg0: typing.Optional[typing.Union[Decimal, bool]] = None
                arg1: typing.Optional[typing.Union[Decimal, bool]] = None
                if operator == 'abs':
                    assert len(stk) >= 1, 'Unable to apply operator abs, stack underflow'
                    assert isinstance(stk[-1], Decimal), 'Unable to apply operator abs, arg 1 must be of type Decimal'
                    arg0 = stk[-1]
                    stk.pop(len(stk) - 1)
                    stk.append(abs(arg0))
                    continue
                if operator == 'add':
                    assert len(stk) >= 2, 'Unable to apply operator add, stack underflow'
                    assert isinstance(stk[-1], Decimal), 'Unable to apply operator add, arg 1 must be of type Decimal'
                    assert isinstance(stk[-2], Decimal), 'Unable to apply operator add, arg 2 must be of type Decimal'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg0 + arg1)
                    continue
                if operator == 'and':
                    assert len(stk) >= 2, 'Unable to apply operator and, stack underflow'
                    assert isinstance(stk[-1], bool), 'Unable to apply operator and, arg 1 must be of type bool'
                    assert isinstance(stk[-2], bool), 'Unable to apply operator and, arg 2 must be of type bool'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg0 and arg1)
                    continue
                if operator == 'atan':
                    assert len(stk) >= 1, 'Unable to apply operator atan, stack underflow'
                    assert isinstance(stk[-1], Decimal), 'Unable to apply operator atan, arg 1 must be of type Decimal'
                    arg0 = stk[-1]
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(atan(arg0)))
                    continue
                if operator == 'ceiling':
                    assert len(stk) >= 1, 'Unable to apply operator ceiling, stack underflow'
                    assert isinstance(stk[-1], Decimal), 'Unable to apply operator ceiling, arg 1 must be of type Decimal'
                    arg0 = stk[-1]
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(ceil(arg0)))
                    continue
                if operator == 'cos':
                    assert len(stk) >= 1, 'Unable to apply operator cos, stack underflow'
                    assert isinstance(stk[-1], Decimal), 'Unable to apply operator cos, arg 1 must be of type Decimal'
                    arg0 = stk[-1]
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(cos(degrees(arg0))))
                    continue
                if operator == 'cvi':
                    assert len(stk) >= 1, 'Unable to apply operator cvi, stack underflow'
                    assert isinstance(stk[-1], Decimal), 'Unable to apply operator cvi, arg 1 must be of type Decimal'
                    arg0 = stk[-1]
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(int(arg0)))
                    continue
                if operator == 'cvr':
                    assert len(stk) >= 1, 'Unable to apply operator cvr, stack underflow'
                if operator == 'div':
                    assert len(stk) >= 2, 'Unable to apply operator div, stack underflow'
                    assert isinstance(stk[-1], Decimal), 'Unable to apply operator div, arg 1 must be of type Decimal'
                    assert isinstance(stk[-2], Decimal), 'Unable to apply operator div, arg 2 must be of type Decimal'
                    assert stk[-1] != Decimal(0), 'Unable to apply operator div, arg1 must not be 0'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg1 / arg0)
                    continue
                if operator == 'dup':
                    assert len(stk) >= 1, 'Unable to apply operator dup, stack underflow'
                    stk.append(stk[-1])
                    continue
                if operator == 'eq':
                    assert len(stk) >= 2, 'Unable to apply operator eq, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg0 == arg1)
                    continue
                if operator == 'exch':
                    assert len(stk) >= 2, 'Unable to apply operator exch, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg0)
                    stk.append(arg1)
                    continue
                if operator == 'exp':
                    assert len(stk) >= 1, 'Unable to apply operator exp, stack underflow'
                    arg0 = stk[-1]
                    assert isinstance(arg0, Decimal), 'Unable to apply operator exp, unexpected type'
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(exp(arg0)))
                    continue
                if operator == 'false':
                    stk.append(False)
                    continue
                if operator == 'floor':
                    assert len(stk) >= 1, 'Unable to apply operator floor, stack underflow'
                    arg0 = stk[-1]
                    assert isinstance(arg0, Decimal), 'Unable to apply operator floor, unexpected type'
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(floor(arg0)))
                    continue
                if operator == 'ge':
                    assert len(stk) >= 2, 'Unable to apply operator ge, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, Decimal), 'Unable to apply operator ge, unexpected type'
                    assert isinstance(arg1, Decimal), 'Unable to apply operator ge, unexpected type'
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg1 >= arg0)
                    continue
                if operator == 'gt':
                    assert len(stk) >= 2, 'Unable to apply operator gt, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, Decimal)
                    assert isinstance(arg1, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg1 > arg0)
                    continue
                if operator == 'idiv':
                    assert len(stk) >= 2, 'Unable to apply operator idiv, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, Decimal)
                    assert isinstance(arg1, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    assert arg0 != Decimal(0), 'Unable to apply operator idiv, division by zero'
                    stk.append(Decimal(int(arg1 / arg0)))
                    continue
                if operator == 'le':
                    assert len(stk) >= 2, 'Unable to apply operator le, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, Decimal)
                    assert isinstance(arg1, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg1 <= arg0)
                    continue
                if operator == 'ln':
                    assert len(stk) >= 1, 'Unable to apply operator ln, stack underflow'
                    arg0 = stk[-1]
                    assert isinstance(arg0, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(log(arg0)))
                    continue
                if operator == 'log':
                    assert len(stk) >= 1, 'Unable to apply operator log, stack underflow'
                    arg0 = stk[-1]
                    assert isinstance(arg0, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(log(arg0, Decimal(10))))
                    continue
                if operator == 'lt':
                    assert len(stk) >= 2, 'Unable to apply operator lt, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, Decimal)
                    assert isinstance(arg1, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg1 < arg0)
                    continue
                if operator == 'mod':
                    assert len(stk) >= 2, 'Unable to apply operator mod, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, Decimal)
                    assert isinstance(arg1, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    assert arg1 != Decimal(0), 'Unable to apply operator mod, division by zero'
                    stk.append(Decimal(int(arg1) % int(arg0)))
                    continue
                if operator == 'mul':
                    assert len(stk) >= 2, 'Unable to apply operator mul, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, Decimal)
                    assert isinstance(arg1, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg1 * arg0)
                    continue
                if operator == 'ne':
                    assert len(stk) >= 2, 'Unable to apply operator ne, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg1 != arg0)
                    continue
                if operator == 'neg':
                    assert len(stk) >= 1, 'Unable to apply operator neg, stack underflow'
                    arg0 = stk[-1]
                    assert isinstance(arg0, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.append(-arg0)
                    continue
                if operator == 'not':
                    assert len(stk) >= 1, 'Unable to apply operator not, stack underflow'
                    arg0 = stk[-1]
                    assert isinstance(arg0, bool)
                    stk.pop(len(stk) - 1)
                    stk.append(not arg0)
                    continue
                if operator == 'or':
                    assert len(stk) >= 2, 'Unable to apply operator or, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, bool)
                    assert isinstance(arg1, bool)
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg1 or arg0)
                    continue
                if operator == 'pop':
                    assert len(stk) >= 1, 'Unable to apply operator pop, stack underflow'
                    stk.pop(-1)
                    continue
                if operator == 'round':
                    assert len(stk) >= 1, 'Unable to apply operator round, stack underflow'
                    arg0 = stk[-1]
                    assert isinstance(arg0, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(round(arg0)))
                    continue
                if operator == 'sin':
                    assert len(stk) >= 1, 'Unable to apply operator sin, stack underflow'
                    arg0 = stk[-1]
                    assert isinstance(arg0, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(sin(degrees(arg0))))
                    continue
                if operator == 'sqrt':
                    assert len(stk) >= 1, 'Unable to apply operator sqrt, stack underflow'
                    arg0 = stk[-1]
                    assert isinstance(arg0, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.append(Decimal(sqrt(arg0)))
                    continue
                if operator == 'sub':
                    assert len(stk) >= 2, 'Unable to apply operator sub, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, Decimal)
                    assert isinstance(arg1, Decimal)
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg1 - arg0)
                    continue
                if operator == 'true':
                    stk.append(True)
                    continue
                if operator == 'xor':
                    assert len(stk) >= 2, 'Unable to apply operator xor, stack underflow'
                    arg0 = stk[-1]
                    arg1 = stk[-2]
                    assert isinstance(arg0, bool)
                    assert isinstance(arg1, bool)
                    stk.pop(len(stk) - 1)
                    stk.pop(len(stk) - 1)
                    stk.append(arg0 or (arg1 and (not (arg0 and arg1))))
                    continue
            i += 1
        out: typing.List[Decimal] = []
        for arg0 in stk:
            assert isinstance(arg0, Decimal)
            out.append(arg0)
        return out