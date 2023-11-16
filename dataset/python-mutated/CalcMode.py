import ast
import logging
import math
import operator as op
import re
from decimal import Decimal
from functools import lru_cache
from ulauncher.modes.BaseMode import BaseMode
from ulauncher.modes.calc.CalcResult import CalcResult
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor, ast.USub: op.neg, ast.Mod: op.mod}
functions = {'sqrt': Decimal.sqrt, 'exp': Decimal.exp, 'ln': Decimal.ln, 'log10': Decimal.log10, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh, 'asinh': math.asinh, 'acosh': math.acosh, 'atanh': math.atanh, 'erf': math.erf, 'erfc': math.erfc, 'gamma': math.gamma, 'lgamma': math.lgamma}
constants = {'pi': Decimal(math.pi), 'e': Decimal(math.e)}
logger = logging.getLogger()

def normalize_expr(expr: str) -> str:
    if False:
        i = 10
        return i + 15
    expr = expr.replace(',', '.')
    expr = expr.replace('^', '**')
    expr = re.sub('\\s*[\\.\\+\\-\\*/%\\(]\\*?\\s*$', '', expr)
    expr = expr + ')' * (expr.count('(') - expr.count(')'))
    return expr

@lru_cache(maxsize=1000)
def eval_expr(expr: str):
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> eval_expr('2^6')\n    64\n    >>> eval_expr('2**6')\n    64\n    >>> eval_expr('2*6+')\n    12\n    >>> eval_expr('1 + 2*3**(4^5) / (6 + -7)')\n    -5.0\n    "
    expr = normalize_expr(expr)
    tree = ast.parse(expr, mode='eval').body
    result = _eval(tree).quantize(Decimal('1e-15'))
    int_result = int(result)
    if result == int_result:
        return int_result
    return result.normalize()

@lru_cache(maxsize=1000)
def _is_enabled(query: str):
    if False:
        i = 10
        return i + 15
    query = normalize_expr(query)
    try:
        node = ast.parse(query, mode='eval').body
        if isinstance(node, ast.Num):
            return True
        if isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Name) and node.left.id not in constants:
                return False
            if isinstance(node.right, ast.Name) and node.right.id not in constants:
                return False
            return True
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, ast.USub)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id in functions
    except SyntaxError:
        pass
    except Exception:
        logger.warning("Calc mode parse error for query: '%s'", query)
    return False

def _eval(node):
    if False:
        print('Hello World!')
    if isinstance(node, ast.Num):
        return Decimal(str(node.n))
    if isinstance(node, ast.BinOp):
        return operators[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.Call) and node.func.id in functions:
        value = functions[node.func.id](_eval(node.args[0]))
        if isinstance(value, float):
            value = Decimal(value)
        return value
    raise TypeError(node)

class CalcMode(BaseMode):

    def is_enabled(self, query):
        if False:
            i = 10
            return i + 15
        return _is_enabled(query)

    def handle_query(self, query):
        if False:
            print('Hello World!')
        try:
            result = CalcResult(result=eval_expr(query))
        except Exception:
            result = CalcResult(error='Invalid expression')
        return [result]