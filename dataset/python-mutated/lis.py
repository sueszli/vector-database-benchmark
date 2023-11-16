import math
import operator as op
from collections import ChainMap as Environment
Symbol = str
List = list
Number = (int, float)

class Procedure(object):
    """A user-defined Scheme procedure."""

    def __init__(self, parms, body, env):
        if False:
            while True:
                i = 10
        (self.parms, self.body, self.env) = (parms, body, env)

    def __call__(self, *args):
        if False:
            while True:
                i = 10
        env = Environment(dict(zip(self.parms, args)), self.env)
        return eval(self.body, env)

def standard_env():
    if False:
        return 10
    'An environment with some Scheme standard procedures.'
    env = {}
    env.update(vars(math))
    env.update({'+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le, '=': op.eq, 'abs': abs, 'append': op.add, 'apply': lambda proc, args: proc(*args), 'begin': lambda *x: x[-1], 'car': lambda x: x[0], 'cdr': lambda x: x[1:], 'cons': lambda x, y: [x] + y, 'eq?': op.is_, 'equal?': op.eq, 'length': len, 'list': lambda *x: list(x), 'list?': lambda x: isinstance(x, list), 'map': lambda *args: list(map(*args)), 'max': max, 'min': min, 'not': op.not_, 'null?': lambda x: x == [], 'number?': lambda x: isinstance(x, Number), 'procedure?': callable, 'round': round, 'symbol?': lambda x: isinstance(x, Symbol)})
    return env
global_env = standard_env()

def parse(program):
    if False:
        return 10
    'Read a Scheme expression from a string.'
    return read_from_tokens(tokenize(program))

def tokenize(s):
    if False:
        for i in range(10):
            print('nop')
    'Convert a string into a list of tokens.'
    return s.replace('(', ' ( ').replace(')', ' ) ').split()

def read_from_tokens(tokens):
    if False:
        while True:
            i = 10
    'Read an expression from a sequence of tokens.'
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)
        return L
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return atom(token)

def atom(token):
    if False:
        for i in range(10):
            print('nop')
    'Numbers become numbers; every other token is a symbol.'
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)

def repl(prompt='lis.py> '):
    if False:
        for i in range(10):
            print('nop')
    'A prompt-read-eval-print loop.'
    while True:
        val = eval(parse(input(prompt)))
        if val is not None:
            print(lispstr(val))

def lispstr(exp):
    if False:
        i = 10
        return i + 15
    'Convert a Python object back into a Lisp-readable string.'
    if isinstance(exp, List):
        return '(' + ' '.join(map(lispstr, exp)) + ')'
    else:
        return str(exp)

def eval(x, env=global_env):
    if False:
        return 10
    'Evaluate an expression in an environment.'
    if isinstance(x, Symbol):
        return env[x]
    elif not isinstance(x, List):
        return x
    elif x[0] == 'quote':
        (_, exp) = x
        return exp
    elif x[0] == 'if':
        (_, test, conseq, alt) = x
        exp = conseq if eval(test, env) else alt
        return eval(exp, env)
    elif x[0] == 'define':
        (_, var, exp) = x
        env[var] = eval(exp, env)
    elif x[0] == 'lambda':
        (_, parms, body) = x
        return Procedure(parms, body, env)
    else:
        proc = eval(x[0], env)
        args = [eval(exp, env) for exp in x[1:]]
        return proc(*args)