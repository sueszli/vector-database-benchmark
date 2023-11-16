"""Transform a string with Python-like source code into SymPy expression. """
from tokenize import generate_tokens, untokenize, TokenError, NUMBER, STRING, NAME, OP, ENDMARKER, ERRORTOKEN, NEWLINE
from keyword import iskeyword
import ast
import unicodedata
from io import StringIO
import builtins
import types
from typing import Tuple as tTuple, Dict as tDict, Any, Callable, List, Optional, Union as tUnion
from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min
null = ''
TOKEN = tTuple[int, str]
DICT = tDict[str, Any]
TRANS = Callable[[List[TOKEN], DICT, DICT], List[TOKEN]]

def _token_splittable(token_name: str) -> bool:
    if False:
        while True:
            i = 10
    "\n    Predicate for whether a token name can be split into multiple tokens.\n\n    A token is splittable if it does not contain an underscore character and\n    it is not the name of a Greek letter. This is used to implicitly convert\n    expressions like 'xyz' into 'x*y*z'.\n    "
    if '_' in token_name:
        return False
    try:
        return not unicodedata.lookup('GREEK SMALL LETTER ' + token_name)
    except KeyError:
        return len(token_name) > 1

def _token_callable(token: TOKEN, local_dict: DICT, global_dict: DICT, nextToken=None):
    if False:
        i = 10
        return i + 15
    '\n    Predicate for whether a token name represents a callable function.\n\n    Essentially wraps ``callable``, but looks up the token name in the\n    locals and globals.\n    '
    func = local_dict.get(token[1])
    if not func:
        func = global_dict.get(token[1])
    return callable(func) and (not isinstance(func, Symbol))

def _add_factorial_tokens(name: str, result: List[TOKEN]) -> List[TOKEN]:
    if False:
        for i in range(10):
            print('nop')
    if result == [] or result[-1][1] == '(':
        raise TokenError()
    beginning = [(NAME, name), (OP, '(')]
    end = [(OP, ')')]
    diff = 0
    length = len(result)
    for (index, token) in enumerate(result[::-1]):
        (toknum, tokval) = token
        i = length - index - 1
        if tokval == ')':
            diff += 1
        elif tokval == '(':
            diff -= 1
        if diff == 0:
            if i - 1 >= 0 and result[i - 1][0] == NAME:
                return result[:i - 1] + beginning + result[i - 1:] + end
            else:
                return result[:i] + beginning + result[i:] + end
    return result

class ParenthesisGroup(List[TOKEN]):
    """List of tokens representing an expression in parentheses."""
    pass

class AppliedFunction:
    """
    A group of tokens representing a function and its arguments.

    `exponent` is for handling the shorthand sin^2, ln^2, etc.
    """

    def __init__(self, function: TOKEN, args: ParenthesisGroup, exponent=None):
        if False:
            while True:
                i = 10
        if exponent is None:
            exponent = []
        self.function = function
        self.args = args
        self.exponent = exponent
        self.items = ['function', 'args', 'exponent']

    def expand(self) -> List[TOKEN]:
        if False:
            print('Hello World!')
        'Return a list of tokens representing the function'
        return [self.function, *self.args]

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return getattr(self, self.items[index])

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'AppliedFunction(%s, %s, %s)' % (self.function, self.args, self.exponent)

def _flatten(result: List[tUnion[TOKEN, AppliedFunction]]):
    if False:
        i = 10
        return i + 15
    result2: List[TOKEN] = []
    for tok in result:
        if isinstance(tok, AppliedFunction):
            result2.extend(tok.expand())
        else:
            result2.append(tok)
    return result2

def _group_parentheses(recursor: TRANS):
    if False:
        for i in range(10):
            print('nop')

    def _inner(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
        if False:
            for i in range(10):
                print('nop')
        'Group tokens between parentheses with ParenthesisGroup.\n\n        Also processes those tokens recursively.\n\n        '
        result: List[tUnion[TOKEN, ParenthesisGroup]] = []
        stacks: List[ParenthesisGroup] = []
        stacklevel = 0
        for token in tokens:
            if token[0] == OP:
                if token[1] == '(':
                    stacks.append(ParenthesisGroup([]))
                    stacklevel += 1
                elif token[1] == ')':
                    stacks[-1].append(token)
                    stack = stacks.pop()
                    if len(stacks) > 0:
                        stacks[-1].extend(stack)
                    else:
                        inner = stack[1:-1]
                        inner = recursor(inner, local_dict, global_dict)
                        parenGroup = [stack[0]] + inner + [stack[-1]]
                        result.append(ParenthesisGroup(parenGroup))
                    stacklevel -= 1
                    continue
            if stacklevel:
                stacks[-1].append(token)
            else:
                result.append(token)
        if stacklevel:
            raise TokenError('Mismatched parentheses')
        return result
    return _inner

def _apply_functions(tokens: List[tUnion[TOKEN, ParenthesisGroup]], local_dict: DICT, global_dict: DICT):
    if False:
        print('Hello World!')
    'Convert a NAME token + ParenthesisGroup into an AppliedFunction.\n\n    Note that ParenthesisGroups, if not applied to any function, are\n    converted back into lists of tokens.\n\n    '
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    symbol = None
    for tok in tokens:
        if isinstance(tok, ParenthesisGroup):
            if symbol and _token_callable(symbol, local_dict, global_dict):
                result[-1] = AppliedFunction(symbol, tok)
                symbol = None
            else:
                result.extend(tok)
        elif tok[0] == NAME:
            symbol = tok
            result.append(tok)
        else:
            symbol = None
            result.append(tok)
    return result

def _implicit_multiplication(tokens: List[tUnion[TOKEN, AppliedFunction]], local_dict: DICT, global_dict: DICT):
    if False:
        while True:
            i = 10
    'Implicitly adds \'*\' tokens.\n\n    Cases:\n\n    - Two AppliedFunctions next to each other ("sin(x)cos(x)")\n\n    - AppliedFunction next to an open parenthesis ("sin x (cos x + 1)")\n\n    - A close parenthesis next to an AppliedFunction ("(x+2)sin x")\n    - A close parenthesis next to an open parenthesis ("(x+2)(x+3)")\n\n    - AppliedFunction next to an implicitly applied function ("sin(x)cos x")\n\n    '
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    skip = False
    for (tok, nextTok) in zip(tokens, tokens[1:]):
        result.append(tok)
        if skip:
            skip = False
            continue
        if tok[0] == OP and tok[1] == '.' and (nextTok[0] == NAME):
            skip = True
            continue
        if isinstance(tok, AppliedFunction):
            if isinstance(nextTok, AppliedFunction):
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                if tok.function[1] == 'Function':
                    tok.function = (tok.function[0], 'Symbol')
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                result.append((OP, '*'))
        elif tok == (OP, ')'):
            if isinstance(nextTok, AppliedFunction):
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                result.append((OP, '*'))
        elif tok[0] == NAME and (not _token_callable(tok, local_dict, global_dict)):
            if isinstance(nextTok, AppliedFunction) or (nextTok[0] == NAME and _token_callable(nextTok, local_dict, global_dict)):
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                result.append((OP, '*'))
    if tokens:
        result.append(tokens[-1])
    return result

def _implicit_application(tokens: List[tUnion[TOKEN, AppliedFunction]], local_dict: DICT, global_dict: DICT):
    if False:
        i = 10
        return i + 15
    'Adds parentheses as needed after functions.'
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    appendParen = 0
    skip = 0
    exponentSkip = False
    for (tok, nextTok) in zip(tokens, tokens[1:]):
        result.append(tok)
        if tok[0] == NAME and nextTok[0] not in [OP, ENDMARKER, NEWLINE]:
            if _token_callable(tok, local_dict, global_dict, nextTok):
                result.append((OP, '('))
                appendParen += 1
        elif tok[0] == NAME and nextTok[0] == OP and (nextTok[1] == '**'):
            if _token_callable(tok, local_dict, global_dict):
                exponentSkip = True
        elif exponentSkip:
            if isinstance(tok, AppliedFunction) or (tok[0] == OP and tok[1] == '*'):
                if not (nextTok[0] == OP and nextTok[1] == '*'):
                    if not (nextTok[0] == OP and nextTok[1] == '('):
                        result.append((OP, '('))
                        appendParen += 1
                    exponentSkip = False
        elif appendParen:
            if nextTok[0] == OP and nextTok[1] in ('^', '**', '*'):
                skip = 1
                continue
            if skip:
                skip -= 1
                continue
            result.append((OP, ')'))
            appendParen -= 1
    if tokens:
        result.append(tokens[-1])
    if appendParen:
        result.extend([(OP, ')')] * appendParen)
    return result

def function_exponentiation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    if False:
        for i in range(10):
            print('nop')
    "Allows functions to be exponentiated, e.g. ``cos**2(x)``.\n\n    Examples\n    ========\n\n    >>> from sympy.parsing.sympy_parser import (parse_expr,\n    ... standard_transformations, function_exponentiation)\n    >>> transformations = standard_transformations + (function_exponentiation,)\n    >>> parse_expr('sin**4(x)', transformations=transformations)\n    sin(x)**4\n    "
    result: List[TOKEN] = []
    exponent: List[TOKEN] = []
    consuming_exponent = False
    level = 0
    for (tok, nextTok) in zip(tokens, tokens[1:]):
        if tok[0] == NAME and nextTok[0] == OP and (nextTok[1] == '**'):
            if _token_callable(tok, local_dict, global_dict):
                consuming_exponent = True
        elif consuming_exponent:
            if tok[0] == NAME and tok[1] == 'Function':
                tok = (NAME, 'Symbol')
            exponent.append(tok)
            if tok[0] == nextTok[0] == OP and tok[1] == ')' and (nextTok[1] == '('):
                consuming_exponent = False
            if tok[0] == nextTok[0] == OP and tok[1] == '*' and (nextTok[1] == '('):
                consuming_exponent = False
                del exponent[-1]
            continue
        elif exponent and (not consuming_exponent):
            if tok[0] == OP:
                if tok[1] == '(':
                    level += 1
                elif tok[1] == ')':
                    level -= 1
            if level == 0:
                result.append(tok)
                result.extend(exponent)
                exponent = []
                continue
        result.append(tok)
    if tokens:
        result.append(tokens[-1])
    if exponent:
        result.extend(exponent)
    return result

def split_symbols_custom(predicate: Callable[[str], bool]):
    if False:
        while True:
            i = 10
    "Creates a transformation that splits symbol names.\n\n    ``predicate`` should return True if the symbol name is to be split.\n\n    For instance, to retain the default behavior but avoid splitting certain\n    symbol names, a predicate like this would work:\n\n\n    >>> from sympy.parsing.sympy_parser import (parse_expr, _token_splittable,\n    ... standard_transformations, implicit_multiplication,\n    ... split_symbols_custom)\n    >>> def can_split(symbol):\n    ...     if symbol not in ('list', 'of', 'unsplittable', 'names'):\n    ...             return _token_splittable(symbol)\n    ...     return False\n    ...\n    >>> transformation = split_symbols_custom(can_split)\n    >>> parse_expr('unsplittable', transformations=standard_transformations +\n    ... (transformation, implicit_multiplication))\n    unsplittable\n    "

    def _split_symbols(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
        if False:
            i = 10
            return i + 15
        result: List[TOKEN] = []
        split = False
        split_previous = False
        for tok in tokens:
            if split_previous:
                split_previous = False
                continue
            split_previous = False
            if tok[0] == NAME and tok[1] in ['Symbol', 'Function']:
                split = True
            elif split and tok[0] == NAME:
                symbol = tok[1][1:-1]
                if predicate(symbol):
                    tok_type = result[-2][1]
                    del result[-2:]
                    i = 0
                    while i < len(symbol):
                        char = symbol[i]
                        if char in local_dict or char in global_dict:
                            result.append((NAME, '%s' % char))
                        elif char.isdigit():
                            chars = [char]
                            for i in range(i + 1, len(symbol)):
                                if not symbol[i].isdigit():
                                    i -= 1
                                    break
                                chars.append(symbol[i])
                            char = ''.join(chars)
                            result.extend([(NAME, 'Number'), (OP, '('), (NAME, "'%s'" % char), (OP, ')')])
                        else:
                            use = tok_type if i == len(symbol) else 'Symbol'
                            result.extend([(NAME, use), (OP, '('), (NAME, "'%s'" % char), (OP, ')')])
                        i += 1
                    split = False
                    split_previous = True
                    continue
                else:
                    split = False
            result.append(tok)
        return result
    return _split_symbols
split_symbols = split_symbols_custom(_token_splittable)

def implicit_multiplication(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT) -> List[TOKEN]:
    if False:
        i = 10
        return i + 15
    "Makes the multiplication operator optional in most cases.\n\n    Use this before :func:`implicit_application`, otherwise expressions like\n    ``sin 2x`` will be parsed as ``x * sin(2)`` rather than ``sin(2*x)``.\n\n    Examples\n    ========\n\n    >>> from sympy.parsing.sympy_parser import (parse_expr,\n    ... standard_transformations, implicit_multiplication)\n    >>> transformations = standard_transformations + (implicit_multiplication,)\n    >>> parse_expr('3 x y', transformations=transformations)\n    3*x*y\n    "
    res1 = _group_parentheses(implicit_multiplication)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_multiplication(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result

def implicit_application(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT) -> List[TOKEN]:
    if False:
        while True:
            i = 10
    "Makes parentheses optional in some cases for function calls.\n\n    Use this after :func:`implicit_multiplication`, otherwise expressions\n    like ``sin 2x`` will be parsed as ``x * sin(2)`` rather than\n    ``sin(2*x)``.\n\n    Examples\n    ========\n\n    >>> from sympy.parsing.sympy_parser import (parse_expr,\n    ... standard_transformations, implicit_application)\n    >>> transformations = standard_transformations + (implicit_application,)\n    >>> parse_expr('cot z + csc z', transformations=transformations)\n    cot(z) + csc(z)\n    "
    res1 = _group_parentheses(implicit_application)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_application(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result

def implicit_multiplication_application(result: List[TOKEN], local_dict: DICT, global_dict: DICT) -> List[TOKEN]:
    if False:
        print('Hello World!')
    'Allows a slightly relaxed syntax.\n\n    - Parentheses for single-argument method calls are optional.\n\n    - Multiplication is implicit.\n\n    - Symbol names can be split (i.e. spaces are not needed between\n      symbols).\n\n    - Functions can be exponentiated.\n\n    Examples\n    ========\n\n    >>> from sympy.parsing.sympy_parser import (parse_expr,\n    ... standard_transformations, implicit_multiplication_application)\n    >>> parse_expr("10sin**2 x**2 + 3xyz + tan theta",\n    ... transformations=(standard_transformations +\n    ... (implicit_multiplication_application,)))\n    3*x*y*z + 10*sin(x**2)**2 + tan(theta)\n\n    '
    for step in (split_symbols, implicit_multiplication, implicit_application, function_exponentiation):
        result = step(result, local_dict, global_dict)
    return result

def auto_symbol(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    if False:
        for i in range(10):
            print('nop')
    'Inserts calls to ``Symbol``/``Function`` for undefined variables.'
    result: List[TOKEN] = []
    prevTok = (-1, '')
    tokens.append((-1, ''))
    for (tok, nextTok) in zip(tokens, tokens[1:]):
        (tokNum, tokVal) = tok
        (nextTokNum, nextTokVal) = nextTok
        if tokNum == NAME:
            name = tokVal
            if name in ['True', 'False', 'None'] or iskeyword(name) or (prevTok[0] == OP and prevTok[1] == '.') or (prevTok[0] == OP and prevTok[1] in ('(', ',') and (nextTokNum == OP) and (nextTokVal == '=')) or (name in local_dict and local_dict[name] is not null):
                result.append((NAME, name))
                continue
            elif name in local_dict:
                local_dict.setdefault(null, set()).add(name)
                if nextTokVal == '(':
                    local_dict[name] = Function(name)
                else:
                    local_dict[name] = Symbol(name)
                result.append((NAME, name))
                continue
            elif name in global_dict:
                obj = global_dict[name]
                if isinstance(obj, (AssumptionKeys, Basic, type)) or callable(obj):
                    result.append((NAME, name))
                    continue
            result.extend([(NAME, 'Symbol' if nextTokVal != '(' else 'Function'), (OP, '('), (NAME, repr(str(name))), (OP, ')')])
        else:
            result.append((tokNum, tokVal))
        prevTok = (tokNum, tokVal)
    return result

def lambda_notation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    if False:
        while True:
            i = 10
    'Substitutes "lambda" with its SymPy equivalent Lambda().\n    However, the conversion does not take place if only "lambda"\n    is passed because that is a syntax error.\n\n    '
    result: List[TOKEN] = []
    flag = False
    (toknum, tokval) = tokens[0]
    tokLen = len(tokens)
    if toknum == NAME and tokval == 'lambda':
        if tokLen == 2 or (tokLen == 3 and tokens[1][0] == NEWLINE):
            result.extend(tokens)
        elif tokLen > 2:
            result.extend([(NAME, 'Lambda'), (OP, '('), (OP, '('), (OP, ')'), (OP, ')')])
            for (tokNum, tokVal) in tokens[1:]:
                if tokNum == OP and tokVal == ':':
                    tokVal = ','
                    flag = True
                if not flag and tokNum == OP and (tokVal in ('*', '**')):
                    raise TokenError('Starred arguments in lambda not supported')
                if flag:
                    result.insert(-1, (tokNum, tokVal))
                else:
                    result.insert(-2, (tokNum, tokVal))
    else:
        result.extend(tokens)
    return result

def factorial_notation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    if False:
        print('Hello World!')
    'Allows standard notation for factorial.'
    result: List[TOKEN] = []
    nfactorial = 0
    for (toknum, tokval) in tokens:
        if toknum == OP and tokval == '!':
            nfactorial += 1
        elif toknum == ERRORTOKEN:
            op = tokval
            if op == '!':
                nfactorial += 1
            else:
                nfactorial = 0
                result.append((OP, op))
        else:
            if nfactorial == 1:
                result = _add_factorial_tokens('factorial', result)
            elif nfactorial == 2:
                result = _add_factorial_tokens('factorial2', result)
            elif nfactorial > 2:
                raise TokenError
            nfactorial = 0
            result.append((toknum, tokval))
    return result

def convert_xor(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    if False:
        print('Hello World!')
    'Treats XOR, ``^``, as exponentiation, ``**``.'
    result: List[TOKEN] = []
    for (toknum, tokval) in tokens:
        if toknum == OP:
            if tokval == '^':
                result.append((OP, '**'))
            else:
                result.append((toknum, tokval))
        else:
            result.append((toknum, tokval))
    return result

def repeated_decimals(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    if False:
        return 10
    '\n    Allows 0.2[1] notation to represent the repeated decimal 0.2111... (19/90)\n\n    Run this before auto_number.\n\n    '
    result: List[TOKEN] = []

    def is_digit(s):
        if False:
            for i in range(10):
                print('nop')
        return all((i in '0123456789_' for i in s))
    num: List[TOKEN] = []
    for (toknum, tokval) in tokens:
        if toknum == NUMBER:
            if not num and '.' in tokval and ('e' not in tokval.lower()) and ('j' not in tokval.lower()):
                num.append((toknum, tokval))
            elif is_digit(tokval) and len(num) == 2:
                num.append((toknum, tokval))
            elif is_digit(tokval) and len(num) == 3 and is_digit(num[-1][1]):
                num.append((toknum, tokval))
            else:
                num = []
        elif toknum == OP:
            if tokval == '[' and len(num) == 1:
                num.append((OP, tokval))
            elif tokval == ']' and len(num) >= 3:
                num.append((OP, tokval))
            elif tokval == '.' and (not num):
                num.append((NUMBER, '0.'))
            else:
                num = []
        else:
            num = []
        result.append((toknum, tokval))
        if num and num[-1][1] == ']':
            result = result[:-len(num)]
            (pre, post) = num[0][1].split('.')
            repetend = num[2][1]
            if len(num) == 5:
                repetend += num[3][1]
            pre = pre.replace('_', '')
            post = post.replace('_', '')
            repetend = repetend.replace('_', '')
            zeros = '0' * len(post)
            (post, repetends) = [w.lstrip('0') for w in [post, repetend]]
            a = pre or '0'
            (b, c) = (post or '0', '1' + zeros)
            (d, e) = (repetends, '9' * len(repetend) + zeros)
            seq = [(OP, '('), (NAME, 'Integer'), (OP, '('), (NUMBER, a), (OP, ')'), (OP, '+'), (NAME, 'Rational'), (OP, '('), (NUMBER, b), (OP, ','), (NUMBER, c), (OP, ')'), (OP, '+'), (NAME, 'Rational'), (OP, '('), (NUMBER, d), (OP, ','), (NUMBER, e), (OP, ')'), (OP, ')')]
            result.extend(seq)
            num = []
    return result

def auto_number(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    if False:
        i = 10
        return i + 15
    '\n    Converts numeric literals to use SymPy equivalents.\n\n    Complex numbers use ``I``, integer literals use ``Integer``, and float\n    literals use ``Float``.\n\n    '
    result: List[TOKEN] = []
    for (toknum, tokval) in tokens:
        if toknum == NUMBER:
            number = tokval
            postfix = []
            if number.endswith(('j', 'J')):
                number = number[:-1]
                postfix = [(OP, '*'), (NAME, 'I')]
            if '.' in number or (('e' in number or 'E' in number) and (not number.startswith(('0x', '0X')))):
                seq = [(NAME, 'Float'), (OP, '('), (NUMBER, repr(str(number))), (OP, ')')]
            else:
                seq = [(NAME, 'Integer'), (OP, '('), (NUMBER, number), (OP, ')')]
            result.extend(seq + postfix)
        else:
            result.append((toknum, tokval))
    return result

def rationalize(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    if False:
        for i in range(10):
            print('nop')
    'Converts floats into ``Rational``. Run AFTER ``auto_number``.'
    result: List[TOKEN] = []
    passed_float = False
    for (toknum, tokval) in tokens:
        if toknum == NAME:
            if tokval == 'Float':
                passed_float = True
                tokval = 'Rational'
            result.append((toknum, tokval))
        elif passed_float == True and toknum == NUMBER:
            passed_float = False
            result.append((STRING, tokval))
        else:
            result.append((toknum, tokval))
    return result

def _transform_equals_sign(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    if False:
        i = 10
        return i + 15
    'Transforms the equals sign ``=`` to instances of Eq.\n\n    This is a helper function for ``convert_equals_signs``.\n    Works with expressions containing one equals sign and no\n    nesting. Expressions like ``(1=2)=False`` will not work with this\n    and should be used with ``convert_equals_signs``.\n\n    Examples: 1=2     to Eq(1,2)\n              1*2=x   to Eq(1*2, x)\n\n    This does not deal with function arguments yet.\n\n    '
    result: List[TOKEN] = []
    if (OP, '=') in tokens:
        result.append((NAME, 'Eq'))
        result.append((OP, '('))
        for token in tokens:
            if token == (OP, '='):
                result.append((OP, ','))
                continue
            result.append(token)
        result.append((OP, ')'))
    else:
        result = tokens
    return result

def convert_equals_signs(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT) -> List[TOKEN]:
    if False:
        print('Hello World!')
    ' Transforms all the equals signs ``=`` to instances of Eq.\n\n    Parses the equals signs in the expression and replaces them with\n    appropriate Eq instances. Also works with nested equals signs.\n\n    Does not yet play well with function arguments.\n    For example, the expression ``(x=y)`` is ambiguous and can be interpreted\n    as x being an argument to a function and ``convert_equals_signs`` will not\n    work for this.\n\n    See also\n    ========\n    convert_equality_operators\n\n    Examples\n    ========\n\n    >>> from sympy.parsing.sympy_parser import (parse_expr,\n    ... standard_transformations, convert_equals_signs)\n    >>> parse_expr("1*2=x", transformations=(\n    ... standard_transformations + (convert_equals_signs,)))\n    Eq(2, x)\n    >>> parse_expr("(1*2=x)=False", transformations=(\n    ... standard_transformations + (convert_equals_signs,)))\n    Eq(Eq(2, x), False)\n\n    '
    res1 = _group_parentheses(convert_equals_signs)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _transform_equals_sign(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result
standard_transformations: tTuple[TRANS, ...] = (lambda_notation, auto_symbol, repeated_decimals, auto_number, factorial_notation)

def stringify_expr(s: str, local_dict: DICT, global_dict: DICT, transformations: tTuple[TRANS, ...]) -> str:
    if False:
        return 10
    '\n    Converts the string ``s`` to Python code, in ``local_dict``\n\n    Generally, ``parse_expr`` should be used.\n    '
    tokens = []
    input_code = StringIO(s.strip())
    for (toknum, tokval, _, _, _) in generate_tokens(input_code.readline):
        tokens.append((toknum, tokval))
    for transform in transformations:
        tokens = transform(tokens, local_dict, global_dict)
    return untokenize(tokens)

def eval_expr(code, local_dict: DICT, global_dict: DICT):
    if False:
        return 10
    '\n    Evaluate Python code generated by ``stringify_expr``.\n\n    Generally, ``parse_expr`` should be used.\n    '
    expr = eval(code, global_dict, local_dict)
    return expr

def parse_expr(s: str, local_dict: Optional[DICT]=None, transformations: tUnion[tTuple[TRANS, ...], str]=standard_transformations, global_dict: Optional[DICT]=None, evaluate=True):
    if False:
        return 10
    'Converts the string ``s`` to a SymPy expression, in ``local_dict``.\n\n    Parameters\n    ==========\n\n    s : str\n        The string to parse.\n\n    local_dict : dict, optional\n        A dictionary of local variables to use when parsing.\n\n    global_dict : dict, optional\n        A dictionary of global variables. By default, this is initialized\n        with ``from sympy import *``; provide this parameter to override\n        this behavior (for instance, to parse ``"Q & S"``).\n\n    transformations : tuple or str\n        A tuple of transformation functions used to modify the tokens of the\n        parsed expression before evaluation. The default transformations\n        convert numeric literals into their SymPy equivalents, convert\n        undefined variables into SymPy symbols, and allow the use of standard\n        mathematical factorial notation (e.g. ``x!``). Selection via\n        string is available (see below).\n\n    evaluate : bool, optional\n        When False, the order of the arguments will remain as they were in the\n        string and automatic simplification that would normally occur is\n        suppressed. (see examples)\n\n    Examples\n    ========\n\n    >>> from sympy.parsing.sympy_parser import parse_expr\n    >>> parse_expr("1/2")\n    1/2\n    >>> type(_)\n    <class \'sympy.core.numbers.Half\'>\n    >>> from sympy.parsing.sympy_parser import standard_transformations,\\\n    ... implicit_multiplication_application\n    >>> transformations = (standard_transformations +\n    ...     (implicit_multiplication_application,))\n    >>> parse_expr("2x", transformations=transformations)\n    2*x\n\n    When evaluate=False, some automatic simplifications will not occur:\n\n    >>> parse_expr("2**3"), parse_expr("2**3", evaluate=False)\n    (8, 2**3)\n\n    In addition the order of the arguments will not be made canonical.\n    This feature allows one to tell exactly how the expression was entered:\n\n    >>> a = parse_expr(\'1 + x\', evaluate=False)\n    >>> b = parse_expr(\'x + 1\', evaluate=0)\n    >>> a == b\n    False\n    >>> a.args\n    (1, x)\n    >>> b.args\n    (x, 1)\n\n    Note, however, that when these expressions are printed they will\n    appear the same:\n\n    >>> assert str(a) == str(b)\n\n    As a convenience, transformations can be seen by printing ``transformations``:\n\n    >>> from sympy.parsing.sympy_parser import transformations\n\n    >>> print(transformations)\n    0: lambda_notation\n    1: auto_symbol\n    2: repeated_decimals\n    3: auto_number\n    4: factorial_notation\n    5: implicit_multiplication_application\n    6: convert_xor\n    7: implicit_application\n    8: implicit_multiplication\n    9: convert_equals_signs\n    10: function_exponentiation\n    11: rationalize\n\n    The ``T`` object provides a way to select these transformations:\n\n    >>> from sympy.parsing.sympy_parser import T\n\n    If you print it, you will see the same list as shown above.\n\n    >>> str(T) == str(transformations)\n    True\n\n    Standard slicing will return a tuple of transformations:\n\n    >>> T[:5] == standard_transformations\n    True\n\n    So ``T`` can be used to specify the parsing transformations:\n\n    >>> parse_expr("2x", transformations=T[:5])\n    Traceback (most recent call last):\n    ...\n    SyntaxError: invalid syntax\n    >>> parse_expr("2x", transformations=T[:6])\n    2*x\n    >>> parse_expr(\'.3\', transformations=T[3, 11])\n    3/10\n    >>> parse_expr(\'.3x\', transformations=T[:])\n    3*x/10\n\n    As a further convenience, strings \'implicit\' and \'all\' can be used\n    to select 0-5 and all the transformations, respectively.\n\n    >>> parse_expr(\'.3x\', transformations=\'all\')\n    3*x/10\n\n    See Also\n    ========\n\n    stringify_expr, eval_expr, standard_transformations,\n    implicit_multiplication_application\n\n    '
    if local_dict is None:
        local_dict = {}
    elif not isinstance(local_dict, dict):
        raise TypeError('expecting local_dict to be a dict')
    elif null in local_dict:
        raise ValueError('cannot use "" in local_dict')
    if global_dict is None:
        global_dict = {}
        exec('from sympy import *', global_dict)
        builtins_dict = vars(builtins)
        for (name, obj) in builtins_dict.items():
            if isinstance(obj, types.BuiltinFunctionType):
                global_dict[name] = obj
        global_dict['max'] = Max
        global_dict['min'] = Min
    elif not isinstance(global_dict, dict):
        raise TypeError('expecting global_dict to be a dict')
    transformations = transformations or ()
    if isinstance(transformations, str):
        if transformations == 'all':
            _transformations = T[:]
        elif transformations == 'implicit':
            _transformations = T[:6]
        else:
            raise ValueError('unknown transformation group name')
    else:
        _transformations = transformations
    code = stringify_expr(s, local_dict, global_dict, _transformations)
    if not evaluate:
        code = compile(evaluateFalse(code), '<string>', 'eval')
    try:
        rv = eval_expr(code, local_dict, global_dict)
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        return rv
    except Exception as e:
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        raise e from ValueError(f'Error from parse_expr with transformed code: {code!r}')

def evaluateFalse(s: str):
    if False:
        return 10
    '\n    Replaces operators with the SymPy equivalent and sets evaluate=False.\n    '
    node = ast.parse(s)
    transformed_node = EvaluateFalseTransformer().visit(node)
    transformed_node = ast.Expression(transformed_node.body[0].value)
    return ast.fix_missing_locations(transformed_node)

class EvaluateFalseTransformer(ast.NodeTransformer):
    operators = {ast.Add: 'Add', ast.Mult: 'Mul', ast.Pow: 'Pow', ast.Sub: 'Add', ast.Div: 'Mul', ast.BitOr: 'Or', ast.BitAnd: 'And', ast.BitXor: 'Not'}
    functions = ('Abs', 'im', 're', 'sign', 'arg', 'conjugate', 'acos', 'acot', 'acsc', 'asec', 'asin', 'atan', 'acosh', 'acoth', 'acsch', 'asech', 'asinh', 'atanh', 'cos', 'cot', 'csc', 'sec', 'sin', 'tan', 'cosh', 'coth', 'csch', 'sech', 'sinh', 'tanh', 'exp', 'ln', 'log', 'sqrt', 'cbrt')
    relational_operators = {ast.NotEq: 'Ne', ast.Lt: 'Lt', ast.LtE: 'Le', ast.Gt: 'Gt', ast.GtE: 'Ge', ast.Eq: 'Eq'}

    def visit_Compare(self, node):
        if False:
            return 10
        if node.ops[0].__class__ in self.relational_operators:
            sympy_class = self.relational_operators[node.ops[0].__class__]
            right = self.visit(node.comparators[0])
            left = self.visit(node.left)
            new_node = ast.Call(func=ast.Name(id=sympy_class, ctx=ast.Load()), args=[left, right], keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
            return new_node
        return node

    def flatten(self, args, func):
        if False:
            return 10
        result = []
        for arg in args:
            if isinstance(arg, ast.Call):
                arg_func = arg.func
                if isinstance(arg_func, ast.Call):
                    arg_func = arg_func.func
                if arg_func.id == func:
                    result.extend(self.flatten(arg.args, func))
                else:
                    result.append(arg)
            else:
                result.append(arg)
        return result

    def visit_BinOp(self, node):
        if False:
            return 10
        if node.op.__class__ in self.operators:
            sympy_class = self.operators[node.op.__class__]
            right = self.visit(node.right)
            left = self.visit(node.left)
            rev = False
            if isinstance(node.op, ast.Sub):
                right = ast.Call(func=ast.Name(id='Mul', ctx=ast.Load()), args=[ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1)), right], keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
            elif isinstance(node.op, ast.Div):
                if isinstance(node.left, ast.UnaryOp):
                    (left, right) = (right, left)
                    rev = True
                    left = ast.Call(func=ast.Name(id='Pow', ctx=ast.Load()), args=[left, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))], keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
                else:
                    right = ast.Call(func=ast.Name(id='Pow', ctx=ast.Load()), args=[right, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))], keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
            if rev:
                (left, right) = (right, left)
            new_node = ast.Call(func=ast.Name(id=sympy_class, ctx=ast.Load()), args=[left, right], keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False, ctx=ast.Load()))], starargs=None, kwargs=None)
            if sympy_class in ('Add', 'Mul'):
                new_node.args = self.flatten(new_node.args, sympy_class)
            return new_node
        return node

    def visit_Call(self, node):
        if False:
            return 10
        new_node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            new_node.keywords.append(ast.keyword(arg='evaluate', value=ast.Constant(value=False, ctx=ast.Load())))
        return new_node
_transformation = {0: lambda_notation, 1: auto_symbol, 2: repeated_decimals, 3: auto_number, 4: factorial_notation, 5: implicit_multiplication_application, 6: convert_xor, 7: implicit_application, 8: implicit_multiplication, 9: convert_equals_signs, 10: function_exponentiation, 11: rationalize}
transformations = '\n'.join(('%s: %s' % (i, func_name(f)) for (i, f) in _transformation.items()))

class _T:
    """class to retrieve transformations from a given slice

    EXAMPLES
    ========

    >>> from sympy.parsing.sympy_parser import T, standard_transformations
    >>> assert T[:5] == standard_transformations
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.N = len(_transformation)

    def __str__(self):
        if False:
            return 10
        return transformations

    def __getitem__(self, t):
        if False:
            print('Hello World!')
        if not type(t) is tuple:
            t = (t,)
        i = []
        for ti in t:
            if type(ti) is int:
                i.append(range(self.N)[ti])
            elif type(ti) is slice:
                i.extend(range(*ti.indices(self.N)))
            else:
                raise TypeError('unexpected slice arg')
        return tuple([_transformation[_] for _ in i])
T = _T()