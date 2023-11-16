"""Utilities for generating program synthesis and evaluation data."""
import contextlib
import sys
import random
import os
try:
    import StringIO
except ImportError:
    from io import StringIO

class ListType(object):

    def __init__(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.arg = arg

    def __str__(self):
        if False:
            print('Hello World!')
        return '[' + str(self.arg) + ']'

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, ListType):
            return False
        return self.arg == other.arg

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.arg)

class VarType(object):

    def __init__(self, arg):
        if False:
            while True:
                i = 10
        self.arg = arg

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.arg)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, VarType):
            return False
        return self.arg == other.arg

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.arg)

class FunctionType(object):

    def __init__(self, args):
        if False:
            print('Hello World!')
        self.args = args

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.args[0]) + ' -> ' + str(self.args[1])

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, FunctionType):
            return False
        return self.args == other.args

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(tuple(self.args))

class Function(object):

    def __init__(self, name, arg_types, output_type, fn_arg_types=None):
        if False:
            return 10
        self.name = name
        self.arg_types = arg_types
        self.fn_arg_types = fn_arg_types or []
        self.output_type = output_type
Null = 100
f_head = Function('c_head', [ListType('Int')], 'Int')

def c_head(xs):
    if False:
        while True:
            i = 10
    return xs[0] if len(xs) > 0 else Null
f_last = Function('c_last', [ListType('Int')], 'Int')

def c_last(xs):
    if False:
        print('Hello World!')
    return xs[-1] if len(xs) > 0 else Null
f_take = Function('c_take', ['Int', ListType('Int')], ListType('Int'))

def c_take(n, xs):
    if False:
        while True:
            i = 10
    return xs[:n]
f_drop = Function('c_drop', ['Int', ListType('Int')], ListType('Int'))

def c_drop(n, xs):
    if False:
        return 10
    return xs[n:]
f_access = Function('c_access', ['Int', ListType('Int')], 'Int')

def c_access(n, xs):
    if False:
        for i in range(10):
            print('nop')
    return xs[n] if n >= 0 and len(xs) > n else Null
f_max = Function('c_max', [ListType('Int')], 'Int')

def c_max(xs):
    if False:
        print('Hello World!')
    return max(xs) if len(xs) > 0 else Null
f_min = Function('c_min', [ListType('Int')], 'Int')

def c_min(xs):
    if False:
        return 10
    return min(xs) if len(xs) > 0 else Null
f_reverse = Function('c_reverse', [ListType('Int')], ListType('Int'))

def c_reverse(xs):
    if False:
        while True:
            i = 10
    return list(reversed(xs))
f_sort = Function('sorted', [ListType('Int')], ListType('Int'))
f_sum = Function('sum', [ListType('Int')], 'Int')

def plus_one(x):
    if False:
        return 10
    return x + 1

def minus_one(x):
    if False:
        i = 10
        return i + 15
    return x - 1

def times_two(x):
    if False:
        i = 10
        return i + 15
    return x * 2

def neg(x):
    if False:
        for i in range(10):
            print('nop')
    return x * -1

def div_two(x):
    if False:
        print('Hello World!')
    return int(x / 2)

def sq(x):
    if False:
        print('Hello World!')
    return x ** 2

def times_three(x):
    if False:
        while True:
            i = 10
    return x * 3

def div_three(x):
    if False:
        print('Hello World!')
    return int(x / 3)

def times_four(x):
    if False:
        print('Hello World!')
    return x * 4

def div_four(x):
    if False:
        while True:
            i = 10
    return int(x / 4)

def pos(x):
    if False:
        return 10
    return x > 0

def neg(x):
    if False:
        while True:
            i = 10
    return x < 0

def even(x):
    if False:
        while True:
            i = 10
    return x % 2 == 0

def odd(x):
    if False:
        while True:
            i = 10
    return x % 2 == 1

def add(x, y):
    if False:
        i = 10
        return i + 15
    return x + y

def sub(x, y):
    if False:
        while True:
            i = 10
    return x - y

def mul(x, y):
    if False:
        i = 10
        return i + 15
    return x * y
f_map = Function('map', [ListType('Int')], ListType('Int'), [FunctionType(['Int', 'Int'])])
f_filter = Function('filter', [ListType('Int')], ListType('Int'), [FunctionType(['Int', 'Bool'])])
f_count = Function('c_count', [ListType('Int')], 'Int', [FunctionType(['Int', 'Bool'])])

def c_count(f, xs):
    if False:
        while True:
            i = 10
    return len([x for x in xs if f(x)])
f_zipwith = Function('c_zipwith', [ListType('Int'), ListType('Int')], ListType('Int'), [FunctionType(['Int', 'Int', 'Int'])])

def c_zipwith(f, xs, ys):
    if False:
        i = 10
        return i + 15
    return [f(x, y) for (x, y) in zip(xs, ys)]
f_scan = Function('c_scan', [ListType('Int')], ListType('Int'), [FunctionType(['Int', 'Int', 'Int'])])

def c_scan(f, xs):
    if False:
        for i in range(10):
            print('nop')
    out = xs
    for i in range(1, len(xs)):
        out[i] = f(xs[i], xs[i - 1])
    return out

@contextlib.contextmanager
def stdoutIO(stdout=None):
    if False:
        return 10
    old = sys.stdout
    if stdout is None:
        stdout = StringIO.StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

def evaluate(program_str, input_names_to_vals, default='ERROR'):
    if False:
        print('Hello World!')
    exec_str = []
    for (name, val) in input_names_to_vals.iteritems():
        exec_str += name + ' = ' + str(val) + '; '
    exec_str += program_str
    if type(exec_str) is list:
        exec_str = ''.join(exec_str)
    with stdoutIO() as s:
        try:
            exec(exec_str + ' print(out)')
            return s.getvalue()[:-1]
        except:
            return default

class Statement(object):
    """Statement class."""

    def __init__(self, fn, output_var, arg_vars, fn_args=None):
        if False:
            for i in range(10):
                print('nop')
        self.fn = fn
        self.output_var = output_var
        self.arg_vars = arg_vars
        self.fn_args = fn_args or []

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s = %s(%s%s%s)' % (self.output_var, self.fn.name, ', '.join(self.fn_args), ', ' if self.fn_args else '', ', '.join(self.arg_vars))

    def substitute(self, env):
        if False:
            return 10
        self.output_var = env.get(self.output_var, self.output_var)
        self.arg_vars = [env.get(v, v) for v in self.arg_vars]

class ProgramGrower(object):
    """Grow programs."""

    def __init__(self, functions, types_to_lambdas):
        if False:
            for i in range(10):
                print('nop')
        self.functions = functions
        self.types_to_lambdas = types_to_lambdas

    def grow_body(self, new_var_name, dependencies, types_to_vars):
        if False:
            while True:
                i = 10
        'Grow the program body.'
        choices = []
        for f in self.functions:
            if all([a in types_to_vars.keys() for a in f.arg_types]):
                choices.append(f)
        f = random.choice(choices)
        args = []
        for t in f.arg_types:
            possible_vars = random.choice(types_to_vars[t])
            var = random.choice(possible_vars)
            args.append(var)
            dependencies.setdefault(new_var_name, []).extend([var] + dependencies[var])
        fn_args = [random.choice(self.types_to_lambdas[t]) for t in f.fn_arg_types]
        types_to_vars.setdefault(f.output_type, []).append(new_var_name)
        return Statement(f, new_var_name, args, fn_args)

    def grow(self, program_len, input_types):
        if False:
            print('Hello World!')
        'Grow the program.'
        var_names = list(reversed(map(chr, range(97, 123))))
        dependencies = dict()
        types_to_vars = dict()
        input_names = []
        for t in input_types:
            var = var_names.pop()
            dependencies[var] = []
            types_to_vars.setdefault(t, []).append(var)
            input_names.append(var)
        statements = []
        for _ in range(program_len - 1):
            var = var_names.pop()
            statements.append(self.grow_body(var, dependencies, types_to_vars))
        statements.append(self.grow_body('out', dependencies, types_to_vars))
        new_var_names = [c for c in map(chr, range(97, 123)) if c not in input_names]
        new_var_names.reverse()
        keep_statements = []
        env = dict()
        for s in statements:
            if s.output_var in dependencies['out']:
                keep_statements.append(s)
                env[s.output_var] = new_var_names.pop()
            if s.output_var == 'out':
                keep_statements.append(s)
        for k in keep_statements:
            k.substitute(env)
        return Program(input_names, input_types, ';'.join([str(k) for k in keep_statements]))

class Program(object):
    """The program class."""

    def __init__(self, input_names, input_types, body):
        if False:
            for i in range(10):
                print('nop')
        self.input_names = input_names
        self.input_types = input_types
        self.body = body

    def evaluate(self, inputs):
        if False:
            while True:
                i = 10
        'Evaluate this program.'
        if len(inputs) != len(self.input_names):
            raise AssertionError('inputs and input_names have tohave the same len. inp: %s , names: %s' % (str(inputs), str(self.input_names)))
        inp_str = ''
        for (name, inp) in zip(self.input_names, inputs):
            inp_str += name + ' = ' + str(inp) + '; '
        with stdoutIO() as s:
            exec(inp_str + self.body + '; print(out)')
        return s.getvalue()[:-1]

    def flat_str(self):
        if False:
            return 10
        out = ''
        for s in self.body.split(';'):
            out += s + ';'
        return out

    def __str__(self):
        if False:
            while True:
                i = 10
        out = ''
        for (n, t) in zip(self.input_names, self.input_types):
            out += n + ' = ' + str(t) + '\n'
        for s in self.body.split(';'):
            out += s + '\n'
        return out
prog_vocab = []
prog_rev_vocab = {}

def tokenize(string, tokens=None):
    if False:
        i = 10
        return i + 15
    'Tokenize the program string.'
    if tokens is None:
        tokens = prog_vocab
    tokens = sorted(tokens, key=len, reverse=True)
    out = []
    string = string.strip()
    while string:
        found = False
        for t in tokens:
            if string.startswith(t):
                out.append(t)
                string = string[len(t):]
                found = True
                break
        if not found:
            raise ValueError("Couldn't tokenize this: " + string)
        string = string.strip()
    return out

def clean_up(output, max_val=100):
    if False:
        i = 10
        return i + 15
    o = eval(str(output))
    if isinstance(o, bool):
        return o
    if isinstance(o, int):
        if o >= 0:
            return min(o, max_val)
        else:
            return max(o, -1 * max_val)
    if isinstance(o, list):
        return [clean_up(l) for l in o]

def make_vocab():
    if False:
        print('Hello World!')
    gen(2, 0)

def gen(max_len, how_many):
    if False:
        i = 10
        return i + 15
    'Generate some programs.'
    functions = [f_head, f_last, f_take, f_drop, f_access, f_max, f_min, f_reverse, f_sort, f_sum, f_map, f_filter, f_count, f_zipwith, f_scan]
    types_to_lambdas = {FunctionType(['Int', 'Int']): ['plus_one', 'minus_one', 'times_two', 'div_two', 'sq', 'times_three', 'div_three', 'times_four', 'div_four'], FunctionType(['Int', 'Bool']): ['pos', 'neg', 'even', 'odd'], FunctionType(['Int', 'Int', 'Int']): ['add', 'sub', 'mul']}
    tokens = []
    for f in functions:
        tokens.append(f.name)
    for v in types_to_lambdas.values():
        tokens.extend(v)
    tokens.extend(['=', ';', ',', '(', ')', '[', ']', 'Int', 'out'])
    tokens.extend(map(chr, range(97, 123)))
    io_tokens = map(str, range(-220, 220))
    if not prog_vocab:
        prog_vocab.extend(['_PAD', '_EOS'] + tokens + io_tokens)
        for (i, t) in enumerate(prog_vocab):
            prog_rev_vocab[t] = i
    io_tokens += [',', '[', ']', ')', '(', 'None']
    grower = ProgramGrower(functions=functions, types_to_lambdas=types_to_lambdas)

    def mk_inp(l):
        if False:
            print('Hello World!')
        return [random.choice(range(-5, 5)) for _ in range(l)]
    tar = [ListType('Int')]
    inps = [[mk_inp(3)], [mk_inp(5)], [mk_inp(7)], [mk_inp(15)]]
    save_prefix = None
    outcomes_to_programs = dict()
    tried = set()
    counter = 0
    choices = [0] if max_len == 0 else range(max_len)
    while counter < 100 * how_many and len(outcomes_to_programs) < how_many:
        counter += 1
        length = random.choice(choices)
        t = grower.grow(length, tar)
        while t in tried:
            length = random.choice(choices)
            t = grower.grow(length, tar)
        tried.add(t)
        outcomes = [clean_up(t.evaluate(i)) for i in inps]
        outcome_str = str(zip(inps, outcomes))
        if outcome_str in outcomes_to_programs:
            outcomes_to_programs[outcome_str] = min([t.flat_str(), outcomes_to_programs[outcome_str]], key=lambda x: len(tokenize(x, tokens)))
        else:
            outcomes_to_programs[outcome_str] = t.flat_str()
        if counter % 5000 == 0:
            print('== proggen: tried: ' + str(counter))
            print('== proggen: kept:  ' + str(len(outcomes_to_programs)))
        if counter % 250000 == 0 and save_prefix is not None:
            print('saving...')
            save_counter = 0
            progfilename = os.path.join(save_prefix, 'prog_' + str(counter) + '.txt')
            iofilename = os.path.join(save_prefix, 'io_' + str(counter) + '.txt')
            prog_token_filename = os.path.join(save_prefix, 'prog_tokens_' + str(counter) + '.txt')
            io_token_filename = os.path.join(save_prefix, 'io_tokens_' + str(counter) + '.txt')
            with open(progfilename, 'a+') as fp, open(iofilename, 'a+') as fi, open(prog_token_filename, 'a+') as ftp, open(io_token_filename, 'a+') as fti:
                for (o, p) in outcomes_to_programs.iteritems():
                    save_counter += 1
                    if save_counter % 500 == 0:
                        print('saving %d of %d' % (save_counter, len(outcomes_to_programs)))
                    fp.write(p + '\n')
                    fi.write(o + '\n')
                    ftp.write(str(tokenize(p, tokens)) + '\n')
                    fti.write(str(tokenize(o, io_tokens)) + '\n')
    return list(outcomes_to_programs.values())