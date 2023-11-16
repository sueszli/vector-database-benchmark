import re
import types
import sys
import os.path
import inspect
import warnings
__version__ = '3.11'
__tabversion__ = '3.10'
yaccdebug = True
debug_file = 'parser.out'
tab_module = 'parsetab'
default_lr = 'LALR'
error_count = 3
yaccdevel = False
resultlimit = 40
pickle_protocol = 0
if sys.version_info[0] < 3:
    string_types = basestring
else:
    string_types = str
MAXINT = sys.maxsize

class PlyLogger(object):

    def __init__(self, f):
        if False:
            return 10
        self.f = f

    def debug(self, msg, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.f.write(msg % args + '\n')
    info = debug

    def warning(self, msg, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.f.write('WARNING: ' + msg % args + '\n')

    def error(self, msg, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.f.write('ERROR: ' + msg % args + '\n')
    critical = debug

class NullLogger(object):

    def __getattribute__(self, name):
        if False:
            print('Hello World!')
        return self

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self

class YaccError(Exception):
    pass

def format_result(r):
    if False:
        print('Hello World!')
    repr_str = repr(r)
    if '\n' in repr_str:
        repr_str = repr(repr_str)
    if len(repr_str) > resultlimit:
        repr_str = repr_str[:resultlimit] + ' ...'
    result = '<%s @ 0x%x> (%s)' % (type(r).__name__, id(r), repr_str)
    return result

def format_stack_entry(r):
    if False:
        i = 10
        return i + 15
    repr_str = repr(r)
    if '\n' in repr_str:
        repr_str = repr(repr_str)
    if len(repr_str) < 16:
        return repr_str
    else:
        return '<%s @ 0x%x>' % (type(r).__name__, id(r))
_errok = None
_token = None
_restart = None
_warnmsg = "PLY: Don't use global functions errok(), token(), and restart() in p_error().\nInstead, invoke the methods on the associated parser instance:\n\n    def p_error(p):\n        ...\n        # Use parser.errok(), parser.token(), parser.restart()\n        ...\n\n    parser = yacc.yacc()\n"

def errok():
    if False:
        for i in range(10):
            print('nop')
    warnings.warn(_warnmsg)
    return _errok()

def restart():
    if False:
        i = 10
        return i + 15
    warnings.warn(_warnmsg)
    return _restart()

def token():
    if False:
        i = 10
        return i + 15
    warnings.warn(_warnmsg)
    return _token()

def call_errorfunc(errorfunc, token, parser):
    if False:
        print('Hello World!')
    global _errok, _token, _restart
    _errok = parser.errok
    _token = parser.token
    _restart = parser.restart
    r = errorfunc(token)
    try:
        del _errok, _token, _restart
    except NameError:
        pass
    return r

class YaccSymbol:

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.type

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self)

class YaccProduction:

    def __init__(self, s, stack=None):
        if False:
            return 10
        self.slice = s
        self.stack = stack
        self.lexer = None
        self.parser = None

    def __getitem__(self, n):
        if False:
            return 10
        if isinstance(n, slice):
            return [s.value for s in self.slice[n]]
        elif n >= 0:
            return self.slice[n].value
        else:
            return self.stack[n].value

    def __setitem__(self, n, v):
        if False:
            print('Hello World!')
        self.slice[n].value = v

    def __getslice__(self, i, j):
        if False:
            while True:
                i = 10
        return [s.value for s in self.slice[i:j]]

    def __len__(self):
        if False:
            return 10
        return len(self.slice)

    def lineno(self, n):
        if False:
            return 10
        return getattr(self.slice[n], 'lineno', 0)

    def set_lineno(self, n, lineno):
        if False:
            print('Hello World!')
        self.slice[n].lineno = lineno

    def linespan(self, n):
        if False:
            return 10
        startline = getattr(self.slice[n], 'lineno', 0)
        endline = getattr(self.slice[n], 'endlineno', startline)
        return (startline, endline)

    def lexpos(self, n):
        if False:
            print('Hello World!')
        return getattr(self.slice[n], 'lexpos', 0)

    def set_lexpos(self, n, lexpos):
        if False:
            print('Hello World!')
        self.slice[n].lexpos = lexpos

    def lexspan(self, n):
        if False:
            print('Hello World!')
        startpos = getattr(self.slice[n], 'lexpos', 0)
        endpos = getattr(self.slice[n], 'endlexpos', startpos)
        return (startpos, endpos)

    def error(self):
        if False:
            return 10
        raise SyntaxError

class LRParser:

    def __init__(self, lrtab, errorf):
        if False:
            return 10
        self.productions = lrtab.lr_productions
        self.action = lrtab.lr_action
        self.goto = lrtab.lr_goto
        self.errorfunc = errorf
        self.set_defaulted_states()
        self.errorok = True

    def errok(self):
        if False:
            while True:
                i = 10
        self.errorok = True

    def restart(self):
        if False:
            print('Hello World!')
        del self.statestack[:]
        del self.symstack[:]
        sym = YaccSymbol()
        sym.type = '$end'
        self.symstack.append(sym)
        self.statestack.append(0)

    def set_defaulted_states(self):
        if False:
            return 10
        self.defaulted_states = {}
        for (state, actions) in self.action.items():
            rules = list(actions.values())
            if len(rules) == 1 and rules[0] < 0:
                self.defaulted_states[state] = rules[0]

    def disable_defaulted_states(self):
        if False:
            while True:
                i = 10
        self.defaulted_states = {}

    def parse(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        if False:
            for i in range(10):
                print('nop')
        if debug or yaccdevel:
            if isinstance(debug, int):
                debug = PlyLogger(sys.stderr)
            return self.parsedebug(input, lexer, debug, tracking, tokenfunc)
        elif tracking:
            return self.parseopt(input, lexer, debug, tracking, tokenfunc)
        else:
            return self.parseopt_notrack(input, lexer, debug, tracking, tokenfunc)

    def parsedebug(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        if False:
            for i in range(10):
                print('nop')
        lookahead = None
        lookaheadstack = []
        actions = self.action
        goto = self.goto
        prod = self.productions
        defaulted_states = self.defaulted_states
        pslice = YaccProduction(None)
        errorcount = 0
        debug.info('PLY: PARSE DEBUG START')
        if not lexer:
            from . import lex
            lexer = lex.lexer
        pslice.lexer = lexer
        pslice.parser = self
        if input is not None:
            lexer.input(input)
        if tokenfunc is None:
            get_token = lexer.token
        else:
            get_token = tokenfunc
        self.token = get_token
        statestack = []
        self.statestack = statestack
        symstack = []
        self.symstack = symstack
        pslice.stack = symstack
        errtoken = None
        statestack.append(0)
        sym = YaccSymbol()
        sym.type = '$end'
        symstack.append(sym)
        state = 0
        while True:
            debug.debug('')
            debug.debug('State  : %s', state)
            if state not in defaulted_states:
                if not lookahead:
                    if not lookaheadstack:
                        lookahead = get_token()
                    else:
                        lookahead = lookaheadstack.pop()
                    if not lookahead:
                        lookahead = YaccSymbol()
                        lookahead.type = '$end'
                ltype = lookahead.type
                t = actions[state].get(ltype)
            else:
                t = defaulted_states[state]
                debug.debug('Defaulted state %s: Reduce using %d', state, -t)
            debug.debug('Stack  : %s', ('%s . %s' % (' '.join([xx.type for xx in symstack][1:]), str(lookahead))).lstrip())
            if t is not None:
                if t > 0:
                    statestack.append(t)
                    state = t
                    debug.debug('Action : Shift and goto state %s', t)
                    symstack.append(lookahead)
                    lookahead = None
                    if errorcount:
                        errorcount -= 1
                    continue
                if t < 0:
                    p = prod[-t]
                    pname = p.name
                    plen = p.len
                    sym = YaccSymbol()
                    sym.type = pname
                    sym.value = None
                    if plen:
                        debug.info('Action : Reduce rule [%s] with %s and goto state %d', p.str, '[' + ','.join([format_stack_entry(_v.value) for _v in symstack[-plen:]]) + ']', goto[statestack[-1 - plen]][pname])
                    else:
                        debug.info('Action : Reduce rule [%s] with %s and goto state %d', p.str, [], goto[statestack[-1]][pname])
                    if plen:
                        targ = symstack[-plen - 1:]
                        targ[0] = sym
                        if tracking:
                            t1 = targ[1]
                            sym.lineno = t1.lineno
                            sym.lexpos = t1.lexpos
                            t1 = targ[-1]
                            sym.endlineno = getattr(t1, 'endlineno', t1.lineno)
                            sym.endlexpos = getattr(t1, 'endlexpos', t1.lexpos)
                        pslice.slice = targ
                        try:
                            del symstack[-plen:]
                            self.state = state
                            p.callable(pslice)
                            del statestack[-plen:]
                            debug.info('Result : %s', format_result(pslice[0]))
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            symstack.extend(targ[1:-1])
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                    else:
                        if tracking:
                            sym.lineno = lexer.lineno
                            sym.lexpos = lexer.lexpos
                        targ = [sym]
                        pslice.slice = targ
                        try:
                            self.state = state
                            p.callable(pslice)
                            debug.info('Result : %s', format_result(pslice[0]))
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, 'value', None)
                    debug.info('Done   : Returning %s', format_result(result))
                    debug.info('PLY: PARSE DEBUG END')
                    return result
            if t is None:
                debug.error('Error  : %s', ('%s . %s' % (' '.join([xx.type for xx in symstack][1:]), str(lookahead))).lstrip())
                if errorcount == 0 or self.errorok:
                    errorcount = error_count
                    self.errorok = False
                    errtoken = lookahead
                    if errtoken.type == '$end':
                        errtoken = None
                    if self.errorfunc:
                        if errtoken and (not hasattr(errtoken, 'lexer')):
                            errtoken.lexer = lexer
                        self.state = state
                        tok = call_errorfunc(self.errorfunc, errtoken, self)
                        if self.errorok:
                            lookahead = tok
                            errtoken = None
                            continue
                    elif errtoken:
                        if hasattr(errtoken, 'lineno'):
                            lineno = lookahead.lineno
                        else:
                            lineno = 0
                        if lineno:
                            sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                        else:
                            sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                    else:
                        sys.stderr.write('yacc: Parse error in input. EOF\n')
                        return
                else:
                    errorcount = error_count
                if len(statestack) <= 1 and lookahead.type != '$end':
                    lookahead = None
                    errtoken = None
                    state = 0
                    del lookaheadstack[:]
                    continue
                if lookahead.type == '$end':
                    return
                if lookahead.type != 'error':
                    sym = symstack[-1]
                    if sym.type == 'error':
                        if tracking:
                            sym.endlineno = getattr(lookahead, 'lineno', sym.lineno)
                            sym.endlexpos = getattr(lookahead, 'lexpos', sym.lexpos)
                        lookahead = None
                        continue
                    t = YaccSymbol()
                    t.type = 'error'
                    if hasattr(lookahead, 'lineno'):
                        t.lineno = t.endlineno = lookahead.lineno
                    if hasattr(lookahead, 'lexpos'):
                        t.lexpos = t.endlexpos = lookahead.lexpos
                    t.value = lookahead
                    lookaheadstack.append(lookahead)
                    lookahead = t
                else:
                    sym = symstack.pop()
                    if tracking:
                        lookahead.lineno = sym.lineno
                        lookahead.lexpos = sym.lexpos
                    statestack.pop()
                    state = statestack[-1]
                continue
            raise RuntimeError('yacc: internal parser error!!!\n')

    def parseopt(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        if False:
            return 10
        lookahead = None
        lookaheadstack = []
        actions = self.action
        goto = self.goto
        prod = self.productions
        defaulted_states = self.defaulted_states
        pslice = YaccProduction(None)
        errorcount = 0
        if not lexer:
            from . import lex
            lexer = lex.lexer
        pslice.lexer = lexer
        pslice.parser = self
        if input is not None:
            lexer.input(input)
        if tokenfunc is None:
            get_token = lexer.token
        else:
            get_token = tokenfunc
        self.token = get_token
        statestack = []
        self.statestack = statestack
        symstack = []
        self.symstack = symstack
        pslice.stack = symstack
        errtoken = None
        statestack.append(0)
        sym = YaccSymbol()
        sym.type = '$end'
        symstack.append(sym)
        state = 0
        while True:
            if state not in defaulted_states:
                if not lookahead:
                    if not lookaheadstack:
                        lookahead = get_token()
                    else:
                        lookahead = lookaheadstack.pop()
                    if not lookahead:
                        lookahead = YaccSymbol()
                        lookahead.type = '$end'
                ltype = lookahead.type
                t = actions[state].get(ltype)
            else:
                t = defaulted_states[state]
            if t is not None:
                if t > 0:
                    statestack.append(t)
                    state = t
                    symstack.append(lookahead)
                    lookahead = None
                    if errorcount:
                        errorcount -= 1
                    continue
                if t < 0:
                    p = prod[-t]
                    pname = p.name
                    plen = p.len
                    sym = YaccSymbol()
                    sym.type = pname
                    sym.value = None
                    if plen:
                        targ = symstack[-plen - 1:]
                        targ[0] = sym
                        if tracking:
                            t1 = targ[1]
                            sym.lineno = t1.lineno
                            sym.lexpos = t1.lexpos
                            t1 = targ[-1]
                            sym.endlineno = getattr(t1, 'endlineno', t1.lineno)
                            sym.endlexpos = getattr(t1, 'endlexpos', t1.lexpos)
                        pslice.slice = targ
                        try:
                            del symstack[-plen:]
                            self.state = state
                            p.callable(pslice)
                            del statestack[-plen:]
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            symstack.extend(targ[1:-1])
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                    else:
                        if tracking:
                            sym.lineno = lexer.lineno
                            sym.lexpos = lexer.lexpos
                        targ = [sym]
                        pslice.slice = targ
                        try:
                            self.state = state
                            p.callable(pslice)
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, 'value', None)
                    return result
            if t is None:
                if errorcount == 0 or self.errorok:
                    errorcount = error_count
                    self.errorok = False
                    errtoken = lookahead
                    if errtoken.type == '$end':
                        errtoken = None
                    if self.errorfunc:
                        if errtoken and (not hasattr(errtoken, 'lexer')):
                            errtoken.lexer = lexer
                        self.state = state
                        tok = call_errorfunc(self.errorfunc, errtoken, self)
                        if self.errorok:
                            lookahead = tok
                            errtoken = None
                            continue
                    elif errtoken:
                        if hasattr(errtoken, 'lineno'):
                            lineno = lookahead.lineno
                        else:
                            lineno = 0
                        if lineno:
                            sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                        else:
                            sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                    else:
                        sys.stderr.write('yacc: Parse error in input. EOF\n')
                        return
                else:
                    errorcount = error_count
                if len(statestack) <= 1 and lookahead.type != '$end':
                    lookahead = None
                    errtoken = None
                    state = 0
                    del lookaheadstack[:]
                    continue
                if lookahead.type == '$end':
                    return
                if lookahead.type != 'error':
                    sym = symstack[-1]
                    if sym.type == 'error':
                        if tracking:
                            sym.endlineno = getattr(lookahead, 'lineno', sym.lineno)
                            sym.endlexpos = getattr(lookahead, 'lexpos', sym.lexpos)
                        lookahead = None
                        continue
                    t = YaccSymbol()
                    t.type = 'error'
                    if hasattr(lookahead, 'lineno'):
                        t.lineno = t.endlineno = lookahead.lineno
                    if hasattr(lookahead, 'lexpos'):
                        t.lexpos = t.endlexpos = lookahead.lexpos
                    t.value = lookahead
                    lookaheadstack.append(lookahead)
                    lookahead = t
                else:
                    sym = symstack.pop()
                    if tracking:
                        lookahead.lineno = sym.lineno
                        lookahead.lexpos = sym.lexpos
                    statestack.pop()
                    state = statestack[-1]
                continue
            raise RuntimeError('yacc: internal parser error!!!\n')

    def parseopt_notrack(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        if False:
            print('Hello World!')
        lookahead = None
        lookaheadstack = []
        actions = self.action
        goto = self.goto
        prod = self.productions
        defaulted_states = self.defaulted_states
        pslice = YaccProduction(None)
        errorcount = 0
        if not lexer:
            from . import lex
            lexer = lex.lexer
        pslice.lexer = lexer
        pslice.parser = self
        if input is not None:
            lexer.input(input)
        if tokenfunc is None:
            get_token = lexer.token
        else:
            get_token = tokenfunc
        self.token = get_token
        statestack = []
        self.statestack = statestack
        symstack = []
        self.symstack = symstack
        pslice.stack = symstack
        errtoken = None
        statestack.append(0)
        sym = YaccSymbol()
        sym.type = '$end'
        symstack.append(sym)
        state = 0
        while True:
            if state not in defaulted_states:
                if not lookahead:
                    if not lookaheadstack:
                        lookahead = get_token()
                    else:
                        lookahead = lookaheadstack.pop()
                    if not lookahead:
                        lookahead = YaccSymbol()
                        lookahead.type = '$end'
                ltype = lookahead.type
                t = actions[state].get(ltype)
            else:
                t = defaulted_states[state]
            if t is not None:
                if t > 0:
                    statestack.append(t)
                    state = t
                    symstack.append(lookahead)
                    lookahead = None
                    if errorcount:
                        errorcount -= 1
                    continue
                if t < 0:
                    p = prod[-t]
                    pname = p.name
                    plen = p.len
                    sym = YaccSymbol()
                    sym.type = pname
                    sym.value = None
                    if plen:
                        targ = symstack[-plen - 1:]
                        targ[0] = sym
                        pslice.slice = targ
                        try:
                            del symstack[-plen:]
                            self.state = state
                            p.callable(pslice)
                            del statestack[-plen:]
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            symstack.extend(targ[1:-1])
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                    else:
                        targ = [sym]
                        pslice.slice = targ
                        try:
                            self.state = state
                            p.callable(pslice)
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, 'value', None)
                    return result
            if t is None:
                if errorcount == 0 or self.errorok:
                    errorcount = error_count
                    self.errorok = False
                    errtoken = lookahead
                    if errtoken.type == '$end':
                        errtoken = None
                    if self.errorfunc:
                        if errtoken and (not hasattr(errtoken, 'lexer')):
                            errtoken.lexer = lexer
                        self.state = state
                        tok = call_errorfunc(self.errorfunc, errtoken, self)
                        if self.errorok:
                            lookahead = tok
                            errtoken = None
                            continue
                    elif errtoken:
                        if hasattr(errtoken, 'lineno'):
                            lineno = lookahead.lineno
                        else:
                            lineno = 0
                        if lineno:
                            sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                        else:
                            sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                    else:
                        sys.stderr.write('yacc: Parse error in input. EOF\n')
                        return
                else:
                    errorcount = error_count
                if len(statestack) <= 1 and lookahead.type != '$end':
                    lookahead = None
                    errtoken = None
                    state = 0
                    del lookaheadstack[:]
                    continue
                if lookahead.type == '$end':
                    return
                if lookahead.type != 'error':
                    sym = symstack[-1]
                    if sym.type == 'error':
                        lookahead = None
                        continue
                    t = YaccSymbol()
                    t.type = 'error'
                    if hasattr(lookahead, 'lineno'):
                        t.lineno = t.endlineno = lookahead.lineno
                    if hasattr(lookahead, 'lexpos'):
                        t.lexpos = t.endlexpos = lookahead.lexpos
                    t.value = lookahead
                    lookaheadstack.append(lookahead)
                    lookahead = t
                else:
                    sym = symstack.pop()
                    statestack.pop()
                    state = statestack[-1]
                continue
            raise RuntimeError('yacc: internal parser error!!!\n')
_is_identifier = re.compile('^[a-zA-Z0-9_-]+$')

class Production(object):
    reduced = 0

    def __init__(self, number, name, prod, precedence=('right', 0), func=None, file='', line=0):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.prod = tuple(prod)
        self.number = number
        self.func = func
        self.callable = None
        self.file = file
        self.line = line
        self.prec = precedence
        self.len = len(self.prod)
        self.usyms = []
        for s in self.prod:
            if s not in self.usyms:
                self.usyms.append(s)
        self.lr_items = []
        self.lr_next = None
        if self.prod:
            self.str = '%s -> %s' % (self.name, ' '.join(self.prod))
        else:
            self.str = '%s -> <empty>' % self.name

    def __str__(self):
        if False:
            print('Hello World!')
        return self.str

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'Production(' + str(self) + ')'

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.prod)

    def __nonzero__(self):
        if False:
            while True:
                i = 10
        return 1

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        return self.prod[index]

    def lr_item(self, n):
        if False:
            print('Hello World!')
        if n > len(self.prod):
            return None
        p = LRItem(self, n)
        try:
            p.lr_after = self.Prodnames[p.prod[n + 1]]
        except (IndexError, KeyError):
            p.lr_after = []
        try:
            p.lr_before = p.prod[n - 1]
        except IndexError:
            p.lr_before = None
        return p

    def bind(self, pdict):
        if False:
            return 10
        if self.func:
            self.callable = pdict[self.func]

class MiniProduction(object):

    def __init__(self, str, name, len, func, file, line):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.len = len
        self.func = func
        self.callable = None
        self.file = file
        self.line = line
        self.str = str

    def __str__(self):
        if False:
            print('Hello World!')
        return self.str

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'MiniProduction(%s)' % self.str

    def bind(self, pdict):
        if False:
            return 10
        if self.func:
            self.callable = pdict[self.func]

class LRItem(object):

    def __init__(self, p, n):
        if False:
            i = 10
            return i + 15
        self.name = p.name
        self.prod = list(p.prod)
        self.number = p.number
        self.lr_index = n
        self.lookaheads = {}
        self.prod.insert(n, '.')
        self.prod = tuple(self.prod)
        self.len = len(self.prod)
        self.usyms = p.usyms

    def __str__(self):
        if False:
            while True:
                i = 10
        if self.prod:
            s = '%s -> %s' % (self.name, ' '.join(self.prod))
        else:
            s = '%s -> <empty>' % self.name
        return s

    def __repr__(self):
        if False:
            return 10
        return 'LRItem(' + str(self) + ')'

def rightmost_terminal(symbols, terminals):
    if False:
        i = 10
        return i + 15
    i = len(symbols) - 1
    while i >= 0:
        if symbols[i] in terminals:
            return symbols[i]
        i -= 1
    return None

class GrammarError(YaccError):
    pass

class Grammar(object):

    def __init__(self, terminals):
        if False:
            return 10
        self.Productions = [None]
        self.Prodnames = {}
        self.Prodmap = {}
        self.Terminals = {}
        for term in terminals:
            self.Terminals[term] = []
        self.Terminals['error'] = []
        self.Nonterminals = {}
        self.First = {}
        self.Follow = {}
        self.Precedence = {}
        self.UsedPrecedence = set()
        self.Start = None

    def __len__(self):
        if False:
            return 10
        return len(self.Productions)

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        return self.Productions[index]

    def set_precedence(self, term, assoc, level):
        if False:
            return 10
        assert self.Productions == [None], 'Must call set_precedence() before add_production()'
        if term in self.Precedence:
            raise GrammarError('Precedence already specified for terminal %r' % term)
        if assoc not in ['left', 'right', 'nonassoc']:
            raise GrammarError("Associativity must be one of 'left','right', or 'nonassoc'")
        self.Precedence[term] = (assoc, level)

    def add_production(self, prodname, syms, func=None, file='', line=0):
        if False:
            print('Hello World!')
        if prodname in self.Terminals:
            raise GrammarError('%s:%d: Illegal rule name %r. Already defined as a token' % (file, line, prodname))
        if prodname == 'error':
            raise GrammarError('%s:%d: Illegal rule name %r. error is a reserved word' % (file, line, prodname))
        if not _is_identifier.match(prodname):
            raise GrammarError('%s:%d: Illegal rule name %r' % (file, line, prodname))
        for (n, s) in enumerate(syms):
            if s[0] in '\'"':
                try:
                    c = eval(s)
                    if len(c) > 1:
                        raise GrammarError('%s:%d: Literal token %s in rule %r may only be a single character' % (file, line, s, prodname))
                    if c not in self.Terminals:
                        self.Terminals[c] = []
                    syms[n] = c
                    continue
                except SyntaxError:
                    pass
            if not _is_identifier.match(s) and s != '%prec':
                raise GrammarError('%s:%d: Illegal name %r in rule %r' % (file, line, s, prodname))
        if '%prec' in syms:
            if syms[-1] == '%prec':
                raise GrammarError('%s:%d: Syntax error. Nothing follows %%prec' % (file, line))
            if syms[-2] != '%prec':
                raise GrammarError('%s:%d: Syntax error. %%prec can only appear at the end of a grammar rule' % (file, line))
            precname = syms[-1]
            prodprec = self.Precedence.get(precname)
            if not prodprec:
                raise GrammarError('%s:%d: Nothing known about the precedence of %r' % (file, line, precname))
            else:
                self.UsedPrecedence.add(precname)
            del syms[-2:]
        else:
            precname = rightmost_terminal(syms, self.Terminals)
            prodprec = self.Precedence.get(precname, ('right', 0))
        map = '%s -> %s' % (prodname, syms)
        if map in self.Prodmap:
            m = self.Prodmap[map]
            raise GrammarError('%s:%d: Duplicate rule %s. ' % (file, line, m) + 'Previous definition at %s:%d' % (m.file, m.line))
        pnumber = len(self.Productions)
        if prodname not in self.Nonterminals:
            self.Nonterminals[prodname] = []
        for t in syms:
            if t in self.Terminals:
                self.Terminals[t].append(pnumber)
            else:
                if t not in self.Nonterminals:
                    self.Nonterminals[t] = []
                self.Nonterminals[t].append(pnumber)
        p = Production(pnumber, prodname, syms, prodprec, func, file, line)
        self.Productions.append(p)
        self.Prodmap[map] = p
        try:
            self.Prodnames[prodname].append(p)
        except KeyError:
            self.Prodnames[prodname] = [p]

    def set_start(self, start=None):
        if False:
            while True:
                i = 10
        if not start:
            start = self.Productions[1].name
        if start not in self.Nonterminals:
            raise GrammarError('start symbol %s undefined' % start)
        self.Productions[0] = Production(0, "S'", [start])
        self.Nonterminals[start].append(0)
        self.Start = start

    def find_unreachable(self):
        if False:
            return 10

        def mark_reachable_from(s):
            if False:
                print('Hello World!')
            if s in reachable:
                return
            reachable.add(s)
            for p in self.Prodnames.get(s, []):
                for r in p.prod:
                    mark_reachable_from(r)
        reachable = set()
        mark_reachable_from(self.Productions[0].prod[0])
        return [s for s in self.Nonterminals if s not in reachable]

    def infinite_cycles(self):
        if False:
            for i in range(10):
                print('nop')
        terminates = {}
        for t in self.Terminals:
            terminates[t] = True
        terminates['$end'] = True
        for n in self.Nonterminals:
            terminates[n] = False
        while True:
            some_change = False
            for (n, pl) in self.Prodnames.items():
                for p in pl:
                    for s in p.prod:
                        if not terminates[s]:
                            p_terminates = False
                            break
                    else:
                        p_terminates = True
                    if p_terminates:
                        if not terminates[n]:
                            terminates[n] = True
                            some_change = True
                        break
            if not some_change:
                break
        infinite = []
        for (s, term) in terminates.items():
            if not term:
                if s not in self.Prodnames and s not in self.Terminals and (s != 'error'):
                    pass
                else:
                    infinite.append(s)
        return infinite

    def undefined_symbols(self):
        if False:
            print('Hello World!')
        result = []
        for p in self.Productions:
            if not p:
                continue
            for s in p.prod:
                if s not in self.Prodnames and s not in self.Terminals and (s != 'error'):
                    result.append((s, p))
        return result

    def unused_terminals(self):
        if False:
            while True:
                i = 10
        unused_tok = []
        for (s, v) in self.Terminals.items():
            if s != 'error' and (not v):
                unused_tok.append(s)
        return unused_tok

    def unused_rules(self):
        if False:
            print('Hello World!')
        unused_prod = []
        for (s, v) in self.Nonterminals.items():
            if not v:
                p = self.Prodnames[s][0]
                unused_prod.append(p)
        return unused_prod

    def unused_precedence(self):
        if False:
            i = 10
            return i + 15
        unused = []
        for termname in self.Precedence:
            if not (termname in self.Terminals or termname in self.UsedPrecedence):
                unused.append((termname, self.Precedence[termname][0]))
        return unused

    def _first(self, beta):
        if False:
            while True:
                i = 10
        result = []
        for x in beta:
            x_produces_empty = False
            for f in self.First[x]:
                if f == '<empty>':
                    x_produces_empty = True
                elif f not in result:
                    result.append(f)
            if x_produces_empty:
                pass
            else:
                break
        else:
            result.append('<empty>')
        return result

    def compute_first(self):
        if False:
            return 10
        if self.First:
            return self.First
        for t in self.Terminals:
            self.First[t] = [t]
        self.First['$end'] = ['$end']
        for n in self.Nonterminals:
            self.First[n] = []
        while True:
            some_change = False
            for n in self.Nonterminals:
                for p in self.Prodnames[n]:
                    for f in self._first(p.prod):
                        if f not in self.First[n]:
                            self.First[n].append(f)
                            some_change = True
            if not some_change:
                break
        return self.First

    def compute_follow(self, start=None):
        if False:
            print('Hello World!')
        if self.Follow:
            return self.Follow
        if not self.First:
            self.compute_first()
        for k in self.Nonterminals:
            self.Follow[k] = []
        if not start:
            start = self.Productions[1].name
        self.Follow[start] = ['$end']
        while True:
            didadd = False
            for p in self.Productions[1:]:
                for (i, B) in enumerate(p.prod):
                    if B in self.Nonterminals:
                        fst = self._first(p.prod[i + 1:])
                        hasempty = False
                        for f in fst:
                            if f != '<empty>' and f not in self.Follow[B]:
                                self.Follow[B].append(f)
                                didadd = True
                            if f == '<empty>':
                                hasempty = True
                        if hasempty or i == len(p.prod) - 1:
                            for f in self.Follow[p.name]:
                                if f not in self.Follow[B]:
                                    self.Follow[B].append(f)
                                    didadd = True
            if not didadd:
                break
        return self.Follow

    def build_lritems(self):
        if False:
            return 10
        for p in self.Productions:
            lastlri = p
            i = 0
            lr_items = []
            while True:
                if i > len(p):
                    lri = None
                else:
                    lri = LRItem(p, i)
                    try:
                        lri.lr_after = self.Prodnames[lri.prod[i + 1]]
                    except (IndexError, KeyError):
                        lri.lr_after = []
                    try:
                        lri.lr_before = lri.prod[i - 1]
                    except IndexError:
                        lri.lr_before = None
                lastlri.lr_next = lri
                if not lri:
                    break
                lr_items.append(lri)
                lastlri = lri
                i += 1
            p.lr_items = lr_items

class VersionError(YaccError):
    pass

class LRTable(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.lr_action = None
        self.lr_goto = None
        self.lr_productions = None
        self.lr_method = None

    def read_table(self, module):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(module, types.ModuleType):
            parsetab = module
        else:
            exec('import %s' % module)
            parsetab = sys.modules[module]
        if parsetab._tabversion != __tabversion__:
            raise VersionError('yacc table file version is out of date')
        self.lr_action = parsetab._lr_action
        self.lr_goto = parsetab._lr_goto
        self.lr_productions = []
        for p in parsetab._lr_productions:
            self.lr_productions.append(MiniProduction(*p))
        self.lr_method = parsetab._lr_method
        return parsetab._lr_signature

    def read_pickle(self, filename):
        if False:
            for i in range(10):
                print('nop')
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        if not os.path.exists(filename):
            raise ImportError
        in_f = open(filename, 'rb')
        tabversion = pickle.load(in_f)
        if tabversion != __tabversion__:
            raise VersionError('yacc table file version is out of date')
        self.lr_method = pickle.load(in_f)
        signature = pickle.load(in_f)
        self.lr_action = pickle.load(in_f)
        self.lr_goto = pickle.load(in_f)
        productions = pickle.load(in_f)
        self.lr_productions = []
        for p in productions:
            self.lr_productions.append(MiniProduction(*p))
        in_f.close()
        return signature

    def bind_callables(self, pdict):
        if False:
            while True:
                i = 10
        for p in self.lr_productions:
            p.bind(pdict)

def digraph(X, R, FP):
    if False:
        while True:
            i = 10
    N = {}
    for x in X:
        N[x] = 0
    stack = []
    F = {}
    for x in X:
        if N[x] == 0:
            traverse(x, N, stack, F, X, R, FP)
    return F

def traverse(x, N, stack, F, X, R, FP):
    if False:
        print('Hello World!')
    stack.append(x)
    d = len(stack)
    N[x] = d
    F[x] = FP(x)
    rel = R(x)
    for y in rel:
        if N[y] == 0:
            traverse(y, N, stack, F, X, R, FP)
        N[x] = min(N[x], N[y])
        for a in F.get(y, []):
            if a not in F[x]:
                F[x].append(a)
    if N[x] == d:
        N[stack[-1]] = MAXINT
        F[stack[-1]] = F[x]
        element = stack.pop()
        while element != x:
            N[stack[-1]] = MAXINT
            F[stack[-1]] = F[x]
            element = stack.pop()

class LALRError(YaccError):
    pass

class LRGeneratedTable(LRTable):

    def __init__(self, grammar, method='LALR', log=None):
        if False:
            while True:
                i = 10
        if method not in ['SLR', 'LALR']:
            raise LALRError('Unsupported method %s' % method)
        self.grammar = grammar
        self.lr_method = method
        if not log:
            log = NullLogger()
        self.log = log
        self.lr_action = {}
        self.lr_goto = {}
        self.lr_productions = grammar.Productions
        self.lr_goto_cache = {}
        self.lr0_cidhash = {}
        self._add_count = 0
        self.sr_conflict = 0
        self.rr_conflict = 0
        self.conflicts = []
        self.sr_conflicts = []
        self.rr_conflicts = []
        self.grammar.build_lritems()
        self.grammar.compute_first()
        self.grammar.compute_follow()
        self.lr_parse_table()

    def lr0_closure(self, I):
        if False:
            print('Hello World!')
        self._add_count += 1
        J = I[:]
        didadd = True
        while didadd:
            didadd = False
            for j in J:
                for x in j.lr_after:
                    if getattr(x, 'lr0_added', 0) == self._add_count:
                        continue
                    J.append(x.lr_next)
                    x.lr0_added = self._add_count
                    didadd = True
        return J

    def lr0_goto(self, I, x):
        if False:
            print('Hello World!')
        g = self.lr_goto_cache.get((id(I), x))
        if g:
            return g
        s = self.lr_goto_cache.get(x)
        if not s:
            s = {}
            self.lr_goto_cache[x] = s
        gs = []
        for p in I:
            n = p.lr_next
            if n and n.lr_before == x:
                s1 = s.get(id(n))
                if not s1:
                    s1 = {}
                    s[id(n)] = s1
                gs.append(n)
                s = s1
        g = s.get('$end')
        if not g:
            if gs:
                g = self.lr0_closure(gs)
                s['$end'] = g
            else:
                s['$end'] = gs
        self.lr_goto_cache[id(I), x] = g
        return g

    def lr0_items(self):
        if False:
            return 10
        C = [self.lr0_closure([self.grammar.Productions[0].lr_next])]
        i = 0
        for I in C:
            self.lr0_cidhash[id(I)] = i
            i += 1
        i = 0
        while i < len(C):
            I = C[i]
            i += 1
            asyms = {}
            for ii in I:
                for s in ii.usyms:
                    asyms[s] = None
            for x in asyms:
                g = self.lr0_goto(I, x)
                if not g or id(g) in self.lr0_cidhash:
                    continue
                self.lr0_cidhash[id(g)] = len(C)
                C.append(g)
        return C

    def compute_nullable_nonterminals(self):
        if False:
            return 10
        nullable = set()
        num_nullable = 0
        while True:
            for p in self.grammar.Productions[1:]:
                if p.len == 0:
                    nullable.add(p.name)
                    continue
                for t in p.prod:
                    if t not in nullable:
                        break
                else:
                    nullable.add(p.name)
            if len(nullable) == num_nullable:
                break
            num_nullable = len(nullable)
        return nullable

    def find_nonterminal_transitions(self, C):
        if False:
            for i in range(10):
                print('nop')
        trans = []
        for (stateno, state) in enumerate(C):
            for p in state:
                if p.lr_index < p.len - 1:
                    t = (stateno, p.prod[p.lr_index + 1])
                    if t[1] in self.grammar.Nonterminals:
                        if t not in trans:
                            trans.append(t)
        return trans

    def dr_relation(self, C, trans, nullable):
        if False:
            i = 10
            return i + 15
        (state, N) = trans
        terms = []
        g = self.lr0_goto(C[state], N)
        for p in g:
            if p.lr_index < p.len - 1:
                a = p.prod[p.lr_index + 1]
                if a in self.grammar.Terminals:
                    if a not in terms:
                        terms.append(a)
        if state == 0 and N == self.grammar.Productions[0].prod[0]:
            terms.append('$end')
        return terms

    def reads_relation(self, C, trans, empty):
        if False:
            return 10
        rel = []
        (state, N) = trans
        g = self.lr0_goto(C[state], N)
        j = self.lr0_cidhash.get(id(g), -1)
        for p in g:
            if p.lr_index < p.len - 1:
                a = p.prod[p.lr_index + 1]
                if a in empty:
                    rel.append((j, a))
        return rel

    def compute_lookback_includes(self, C, trans, nullable):
        if False:
            i = 10
            return i + 15
        lookdict = {}
        includedict = {}
        dtrans = {}
        for t in trans:
            dtrans[t] = 1
        for (state, N) in trans:
            lookb = []
            includes = []
            for p in C[state]:
                if p.name != N:
                    continue
                lr_index = p.lr_index
                j = state
                while lr_index < p.len - 1:
                    lr_index = lr_index + 1
                    t = p.prod[lr_index]
                    if (j, t) in dtrans:
                        li = lr_index + 1
                        while li < p.len:
                            if p.prod[li] in self.grammar.Terminals:
                                break
                            if p.prod[li] not in nullable:
                                break
                            li = li + 1
                        else:
                            includes.append((j, t))
                    g = self.lr0_goto(C[j], t)
                    j = self.lr0_cidhash.get(id(g), -1)
                for r in C[j]:
                    if r.name != p.name:
                        continue
                    if r.len != p.len:
                        continue
                    i = 0
                    while i < r.lr_index:
                        if r.prod[i] != p.prod[i + 1]:
                            break
                        i = i + 1
                    else:
                        lookb.append((j, r))
            for i in includes:
                if i not in includedict:
                    includedict[i] = []
                includedict[i].append((state, N))
            lookdict[state, N] = lookb
        return (lookdict, includedict)

    def compute_read_sets(self, C, ntrans, nullable):
        if False:
            for i in range(10):
                print('nop')
        FP = lambda x: self.dr_relation(C, x, nullable)
        R = lambda x: self.reads_relation(C, x, nullable)
        F = digraph(ntrans, R, FP)
        return F

    def compute_follow_sets(self, ntrans, readsets, inclsets):
        if False:
            while True:
                i = 10
        FP = lambda x: readsets[x]
        R = lambda x: inclsets.get(x, [])
        F = digraph(ntrans, R, FP)
        return F

    def add_lookaheads(self, lookbacks, followset):
        if False:
            return 10
        for (trans, lb) in lookbacks.items():
            for (state, p) in lb:
                if state not in p.lookaheads:
                    p.lookaheads[state] = []
                f = followset.get(trans, [])
                for a in f:
                    if a not in p.lookaheads[state]:
                        p.lookaheads[state].append(a)

    def add_lalr_lookaheads(self, C):
        if False:
            i = 10
            return i + 15
        nullable = self.compute_nullable_nonterminals()
        trans = self.find_nonterminal_transitions(C)
        readsets = self.compute_read_sets(C, trans, nullable)
        (lookd, included) = self.compute_lookback_includes(C, trans, nullable)
        followsets = self.compute_follow_sets(trans, readsets, included)
        self.add_lookaheads(lookd, followsets)

    def lr_parse_table(self):
        if False:
            return 10
        Productions = self.grammar.Productions
        Precedence = self.grammar.Precedence
        goto = self.lr_goto
        action = self.lr_action
        log = self.log
        actionp = {}
        log.info('Parsing method: %s', self.lr_method)
        C = self.lr0_items()
        if self.lr_method == 'LALR':
            self.add_lalr_lookaheads(C)
        st = 0
        for I in C:
            actlist = []
            st_action = {}
            st_actionp = {}
            st_goto = {}
            log.info('')
            log.info('state %d', st)
            log.info('')
            for p in I:
                log.info('    (%d) %s', p.number, p)
            log.info('')
            for p in I:
                if p.len == p.lr_index + 1:
                    if p.name == "S'":
                        st_action['$end'] = 0
                        st_actionp['$end'] = p
                    else:
                        if self.lr_method == 'LALR':
                            laheads = p.lookaheads[st]
                        else:
                            laheads = self.grammar.Follow[p.name]
                        for a in laheads:
                            actlist.append((a, p, 'reduce using rule %d (%s)' % (p.number, p)))
                            r = st_action.get(a)
                            if r is not None:
                                if r > 0:
                                    (sprec, slevel) = Precedence.get(a, ('right', 0))
                                    (rprec, rlevel) = Productions[p.number].prec
                                    if slevel < rlevel or (slevel == rlevel and rprec == 'left'):
                                        st_action[a] = -p.number
                                        st_actionp[a] = p
                                        if not slevel and (not rlevel):
                                            log.info('  ! shift/reduce conflict for %s resolved as reduce', a)
                                            self.sr_conflicts.append((st, a, 'reduce'))
                                        Productions[p.number].reduced += 1
                                    elif slevel == rlevel and rprec == 'nonassoc':
                                        st_action[a] = None
                                    elif not rlevel:
                                        log.info('  ! shift/reduce conflict for %s resolved as shift', a)
                                        self.sr_conflicts.append((st, a, 'shift'))
                                elif r < 0:
                                    oldp = Productions[-r]
                                    pp = Productions[p.number]
                                    if oldp.line > pp.line:
                                        st_action[a] = -p.number
                                        st_actionp[a] = p
                                        (chosenp, rejectp) = (pp, oldp)
                                        Productions[p.number].reduced += 1
                                        Productions[oldp.number].reduced -= 1
                                    else:
                                        (chosenp, rejectp) = (oldp, pp)
                                    self.rr_conflicts.append((st, chosenp, rejectp))
                                    log.info('  ! reduce/reduce conflict for %s resolved using rule %d (%s)', a, st_actionp[a].number, st_actionp[a])
                                else:
                                    raise LALRError('Unknown conflict in state %d' % st)
                            else:
                                st_action[a] = -p.number
                                st_actionp[a] = p
                                Productions[p.number].reduced += 1
                else:
                    i = p.lr_index
                    a = p.prod[i + 1]
                    if a in self.grammar.Terminals:
                        g = self.lr0_goto(I, a)
                        j = self.lr0_cidhash.get(id(g), -1)
                        if j >= 0:
                            actlist.append((a, p, 'shift and go to state %d' % j))
                            r = st_action.get(a)
                            if r is not None:
                                if r > 0:
                                    if r != j:
                                        raise LALRError('Shift/shift conflict in state %d' % st)
                                elif r < 0:
                                    (sprec, slevel) = Precedence.get(a, ('right', 0))
                                    (rprec, rlevel) = Productions[st_actionp[a].number].prec
                                    if slevel > rlevel or (slevel == rlevel and rprec == 'right'):
                                        Productions[st_actionp[a].number].reduced -= 1
                                        st_action[a] = j
                                        st_actionp[a] = p
                                        if not rlevel:
                                            log.info('  ! shift/reduce conflict for %s resolved as shift', a)
                                            self.sr_conflicts.append((st, a, 'shift'))
                                    elif slevel == rlevel and rprec == 'nonassoc':
                                        st_action[a] = None
                                    elif not slevel and (not rlevel):
                                        log.info('  ! shift/reduce conflict for %s resolved as reduce', a)
                                        self.sr_conflicts.append((st, a, 'reduce'))
                                else:
                                    raise LALRError('Unknown conflict in state %d' % st)
                            else:
                                st_action[a] = j
                                st_actionp[a] = p
            _actprint = {}
            for (a, p, m) in actlist:
                if a in st_action:
                    if p is st_actionp[a]:
                        log.info('    %-15s %s', a, m)
                        _actprint[a, m] = 1
            log.info('')
            not_used = 0
            for (a, p, m) in actlist:
                if a in st_action:
                    if p is not st_actionp[a]:
                        if not (a, m) in _actprint:
                            log.debug('  ! %-15s [ %s ]', a, m)
                            not_used = 1
                            _actprint[a, m] = 1
            if not_used:
                log.debug('')
            nkeys = {}
            for ii in I:
                for s in ii.usyms:
                    if s in self.grammar.Nonterminals:
                        nkeys[s] = None
            for n in nkeys:
                g = self.lr0_goto(I, n)
                j = self.lr0_cidhash.get(id(g), -1)
                if j >= 0:
                    st_goto[n] = j
                    log.info('    %-30s shift and go to state %d', n, j)
            action[st] = st_action
            actionp[st] = st_actionp
            goto[st] = st_goto
            st += 1

    def write_table(self, tabmodule, outputdir='', signature=''):
        if False:
            while True:
                i = 10
        if isinstance(tabmodule, types.ModuleType):
            raise IOError("Won't overwrite existing tabmodule")
        basemodulename = tabmodule.split('.')[-1]
        filename = os.path.join(outputdir, basemodulename) + '.py'
        try:
            f = open(filename, 'w')
            f.write('\n# %s\n# This file is automatically generated. Do not edit.\n# pylint: disable=W,C,R\n_tabversion = %r\n\n_lr_method = %r\n\n_lr_signature = %r\n    ' % (os.path.basename(filename), __tabversion__, self.lr_method, signature))
            smaller = 1
            if smaller:
                items = {}
                for (s, nd) in self.lr_action.items():
                    for (name, v) in nd.items():
                        i = items.get(name)
                        if not i:
                            i = ([], [])
                            items[name] = i
                        i[0].append(s)
                        i[1].append(v)
                f.write('\n_lr_action_items = {')
                for (k, v) in items.items():
                    f.write('%r:([' % k)
                    for i in v[0]:
                        f.write('%r,' % i)
                    f.write('],[')
                    for i in v[1]:
                        f.write('%r,' % i)
                    f.write(']),')
                f.write('}\n')
                f.write('\n_lr_action = {}\nfor _k, _v in _lr_action_items.items():\n   for _x,_y in zip(_v[0],_v[1]):\n      if not _x in _lr_action:  _lr_action[_x] = {}\n      _lr_action[_x][_k] = _y\ndel _lr_action_items\n')
            else:
                f.write('\n_lr_action = { ')
                for (k, v) in self.lr_action.items():
                    f.write('(%r,%r):%r,' % (k[0], k[1], v))
                f.write('}\n')
            if smaller:
                items = {}
                for (s, nd) in self.lr_goto.items():
                    for (name, v) in nd.items():
                        i = items.get(name)
                        if not i:
                            i = ([], [])
                            items[name] = i
                        i[0].append(s)
                        i[1].append(v)
                f.write('\n_lr_goto_items = {')
                for (k, v) in items.items():
                    f.write('%r:([' % k)
                    for i in v[0]:
                        f.write('%r,' % i)
                    f.write('],[')
                    for i in v[1]:
                        f.write('%r,' % i)
                    f.write(']),')
                f.write('}\n')
                f.write('\n_lr_goto = {}\nfor _k, _v in _lr_goto_items.items():\n   for _x, _y in zip(_v[0], _v[1]):\n       if not _x in _lr_goto: _lr_goto[_x] = {}\n       _lr_goto[_x][_k] = _y\ndel _lr_goto_items\n')
            else:
                f.write('\n_lr_goto = { ')
                for (k, v) in self.lr_goto.items():
                    f.write('(%r,%r):%r,' % (k[0], k[1], v))
                f.write('}\n')
            f.write('_lr_productions = [\n')
            for p in self.lr_productions:
                if p.func:
                    f.write('  (%r,%r,%d,%r,%r,%d),\n' % (p.str, p.name, p.len, p.func, os.path.basename(p.file), p.line))
                else:
                    f.write('  (%r,%r,%d,None,None,None),\n' % (str(p), p.name, p.len))
            f.write(']\n')
            f.close()
        except IOError as e:
            raise

    def pickle_table(self, filename, signature=''):
        if False:
            while True:
                i = 10
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        with open(filename, 'wb') as outf:
            pickle.dump(__tabversion__, outf, pickle_protocol)
            pickle.dump(self.lr_method, outf, pickle_protocol)
            pickle.dump(signature, outf, pickle_protocol)
            pickle.dump(self.lr_action, outf, pickle_protocol)
            pickle.dump(self.lr_goto, outf, pickle_protocol)
            outp = []
            for p in self.lr_productions:
                if p.func:
                    outp.append((p.str, p.name, p.len, p.func, os.path.basename(p.file), p.line))
                else:
                    outp.append((str(p), p.name, p.len, None, None, None))
            pickle.dump(outp, outf, pickle_protocol)

def get_caller_module_dict(levels):
    if False:
        return 10
    f = sys._getframe(levels)
    ldict = f.f_globals.copy()
    if f.f_globals != f.f_locals:
        ldict.update(f.f_locals)
    return ldict

def parse_grammar(doc, file, line):
    if False:
        i = 10
        return i + 15
    grammar = []
    pstrings = doc.splitlines()
    lastp = None
    dline = line
    for ps in pstrings:
        dline += 1
        p = ps.split()
        if not p:
            continue
        try:
            if p[0] == '|':
                if not lastp:
                    raise SyntaxError("%s:%d: Misplaced '|'" % (file, dline))
                prodname = lastp
                syms = p[1:]
            else:
                prodname = p[0]
                lastp = prodname
                syms = p[2:]
                assign = p[1]
                if assign != ':' and assign != '::=':
                    raise SyntaxError("%s:%d: Syntax error. Expected ':'" % (file, dline))
            grammar.append((file, dline, prodname, syms))
        except SyntaxError:
            raise
        except Exception:
            raise SyntaxError('%s:%d: Syntax error in rule %r' % (file, dline, ps.strip()))
    return grammar

class ParserReflect(object):

    def __init__(self, pdict, log=None):
        if False:
            i = 10
            return i + 15
        self.pdict = pdict
        self.start = None
        self.error_func = None
        self.tokens = None
        self.modules = set()
        self.grammar = []
        self.error = False
        if log is None:
            self.log = PlyLogger(sys.stderr)
        else:
            self.log = log

    def get_all(self):
        if False:
            i = 10
            return i + 15
        self.get_start()
        self.get_error_func()
        self.get_tokens()
        self.get_precedence()
        self.get_pfunctions()

    def validate_all(self):
        if False:
            while True:
                i = 10
        self.validate_start()
        self.validate_error_func()
        self.validate_tokens()
        self.validate_precedence()
        self.validate_pfunctions()
        self.validate_modules()
        return self.error

    def signature(self):
        if False:
            while True:
                i = 10
        parts = []
        try:
            if self.start:
                parts.append(self.start)
            if self.prec:
                parts.append(''.join([''.join(p) for p in self.prec]))
            if self.tokens:
                parts.append(' '.join(self.tokens))
            for f in self.pfuncs:
                if f[3]:
                    parts.append(f[3])
        except (TypeError, ValueError):
            pass
        return ''.join(parts)

    def validate_modules(self):
        if False:
            i = 10
            return i + 15
        fre = re.compile('\\s*def\\s+(p_[a-zA-Z_0-9]*)\\(')
        for module in self.modules:
            try:
                (lines, linen) = inspect.getsourcelines(module)
            except IOError:
                continue
            counthash = {}
            for (linen, line) in enumerate(lines):
                linen += 1
                m = fre.match(line)
                if m:
                    name = m.group(1)
                    prev = counthash.get(name)
                    if not prev:
                        counthash[name] = linen
                    else:
                        filename = inspect.getsourcefile(module)
                        self.log.warning('%s:%d: Function %s redefined. Previously defined on line %d', filename, linen, name, prev)

    def get_start(self):
        if False:
            return 10
        self.start = self.pdict.get('start')

    def validate_start(self):
        if False:
            while True:
                i = 10
        if self.start is not None:
            if not isinstance(self.start, string_types):
                self.log.error("'start' must be a string")

    def get_error_func(self):
        if False:
            return 10
        self.error_func = self.pdict.get('p_error')

    def validate_error_func(self):
        if False:
            return 10
        if self.error_func:
            if isinstance(self.error_func, types.FunctionType):
                ismethod = 0
            elif isinstance(self.error_func, types.MethodType):
                ismethod = 1
            else:
                self.log.error("'p_error' defined, but is not a function or method")
                self.error = True
                return
            eline = self.error_func.__code__.co_firstlineno
            efile = self.error_func.__code__.co_filename
            module = inspect.getmodule(self.error_func)
            self.modules.add(module)
            argcount = self.error_func.__code__.co_argcount - ismethod
            if argcount != 1:
                self.log.error('%s:%d: p_error() requires 1 argument', efile, eline)
                self.error = True

    def get_tokens(self):
        if False:
            print('Hello World!')
        tokens = self.pdict.get('tokens')
        if not tokens:
            self.log.error('No token list is defined')
            self.error = True
            return
        if not isinstance(tokens, (list, tuple)):
            self.log.error('tokens must be a list or tuple')
            self.error = True
            return
        if not tokens:
            self.log.error('tokens is empty')
            self.error = True
            return
        self.tokens = sorted(tokens)

    def validate_tokens(self):
        if False:
            print('Hello World!')
        if 'error' in self.tokens:
            self.log.error("Illegal token name 'error'. Is a reserved word")
            self.error = True
            return
        terminals = set()
        for n in self.tokens:
            if n in terminals:
                self.log.warning('Token %r multiply defined', n)
            terminals.add(n)

    def get_precedence(self):
        if False:
            while True:
                i = 10
        self.prec = self.pdict.get('precedence')

    def validate_precedence(self):
        if False:
            i = 10
            return i + 15
        preclist = []
        if self.prec:
            if not isinstance(self.prec, (list, tuple)):
                self.log.error('precedence must be a list or tuple')
                self.error = True
                return
            for (level, p) in enumerate(self.prec):
                if not isinstance(p, (list, tuple)):
                    self.log.error('Bad precedence table')
                    self.error = True
                    return
                if len(p) < 2:
                    self.log.error('Malformed precedence entry %s. Must be (assoc, term, ..., term)', p)
                    self.error = True
                    return
                assoc = p[0]
                if not isinstance(assoc, string_types):
                    self.log.error('precedence associativity must be a string')
                    self.error = True
                    return
                for term in p[1:]:
                    if not isinstance(term, string_types):
                        self.log.error('precedence items must be strings')
                        self.error = True
                        return
                    preclist.append((term, assoc, level + 1))
        self.preclist = preclist

    def get_pfunctions(self):
        if False:
            for i in range(10):
                print('nop')
        p_functions = []
        for (name, item) in self.pdict.items():
            if not name.startswith('p_') or name == 'p_error':
                continue
            if isinstance(item, (types.FunctionType, types.MethodType)):
                line = getattr(item, 'co_firstlineno', item.__code__.co_firstlineno)
                module = inspect.getmodule(item)
                p_functions.append((line, module, name, item.__doc__))
        p_functions.sort(key=lambda p_function: (p_function[0], str(p_function[1]), p_function[2], p_function[3]))
        self.pfuncs = p_functions

    def validate_pfunctions(self):
        if False:
            while True:
                i = 10
        grammar = []
        if len(self.pfuncs) == 0:
            self.log.error('no rules of the form p_rulename are defined')
            self.error = True
            return
        for (line, module, name, doc) in self.pfuncs:
            file = inspect.getsourcefile(module)
            func = self.pdict[name]
            if isinstance(func, types.MethodType):
                reqargs = 2
            else:
                reqargs = 1
            if func.__code__.co_argcount > reqargs:
                self.log.error('%s:%d: Rule %r has too many arguments', file, line, func.__name__)
                self.error = True
            elif func.__code__.co_argcount < reqargs:
                self.log.error('%s:%d: Rule %r requires an argument', file, line, func.__name__)
                self.error = True
            elif not func.__doc__:
                self.log.warning('%s:%d: No documentation string specified in function %r (ignored)', file, line, func.__name__)
            else:
                try:
                    parsed_g = parse_grammar(doc, file, line)
                    for g in parsed_g:
                        grammar.append((name, g))
                except SyntaxError as e:
                    self.log.error(str(e))
                    self.error = True
                self.modules.add(module)
        for (n, v) in self.pdict.items():
            if n.startswith('p_') and isinstance(v, (types.FunctionType, types.MethodType)):
                continue
            if n.startswith('t_'):
                continue
            if n.startswith('p_') and n != 'p_error':
                self.log.warning('%r not defined as a function', n)
            if isinstance(v, types.FunctionType) and v.__code__.co_argcount == 1 or (isinstance(v, types.MethodType) and v.__func__.__code__.co_argcount == 2):
                if v.__doc__:
                    try:
                        doc = v.__doc__.split(' ')
                        if doc[1] == ':':
                            self.log.warning('%s:%d: Possible grammar rule %r defined without p_ prefix', v.__code__.co_filename, v.__code__.co_firstlineno, n)
                    except IndexError:
                        pass
        self.grammar = grammar

def yacc(method='LALR', debug=yaccdebug, module=None, tabmodule=tab_module, start=None, check_recursion=True, optimize=False, write_tables=True, debugfile=debug_file, outputdir=None, debuglog=None, errorlog=None, picklefile=None):
    if False:
        i = 10
        return i + 15
    if tabmodule is None:
        tabmodule = tab_module
    global parse
    if picklefile:
        write_tables = 0
    if errorlog is None:
        errorlog = PlyLogger(sys.stderr)
    if module:
        _items = [(k, getattr(module, k)) for k in dir(module)]
        pdict = dict(_items)
        if '__file__' not in pdict:
            pdict['__file__'] = sys.modules[pdict['__module__']].__file__
        if '__package__' not in pdict and '__module__' in pdict:
            if hasattr(sys.modules[pdict['__module__']], '__package__'):
                pdict['__package__'] = sys.modules[pdict['__module__']].__package__
    else:
        pdict = get_caller_module_dict(2)
    if outputdir is None:
        if isinstance(tabmodule, types.ModuleType):
            srcfile = tabmodule.__file__
        elif '.' not in tabmodule:
            srcfile = pdict['__file__']
        else:
            parts = tabmodule.split('.')
            pkgname = '.'.join(parts[:-1])
            exec('import %s' % pkgname)
            srcfile = getattr(sys.modules[pkgname], '__file__', '')
        outputdir = os.path.dirname(srcfile)
    pkg = pdict.get('__package__')
    if pkg and isinstance(tabmodule, str):
        if '.' not in tabmodule:
            tabmodule = pkg + '.' + tabmodule
    if start is not None:
        pdict['start'] = start
    pinfo = ParserReflect(pdict, log=errorlog)
    pinfo.get_all()
    if pinfo.error:
        raise YaccError('Unable to build parser')
    signature = pinfo.signature()
    try:
        lr = LRTable()
        if picklefile:
            read_signature = lr.read_pickle(picklefile)
        else:
            read_signature = lr.read_table(tabmodule)
        if optimize or read_signature == signature:
            try:
                lr.bind_callables(pinfo.pdict)
                parser = LRParser(lr, pinfo.error_func)
                parse = parser.parse
                return parser
            except Exception as e:
                errorlog.warning('There was a problem loading the table file: %r', e)
    except VersionError as e:
        errorlog.warning(str(e))
    except ImportError:
        pass
    if debuglog is None:
        if debug:
            try:
                debuglog = PlyLogger(open(os.path.join(outputdir, debugfile), 'w'))
            except IOError as e:
                errorlog.warning("Couldn't open %r. %s" % (debugfile, e))
                debuglog = NullLogger()
        else:
            debuglog = NullLogger()
    debuglog.info('Created by PLY version %s (http://www.dabeaz.com/ply)', __version__)
    errors = False
    if pinfo.validate_all():
        raise YaccError('Unable to build parser')
    if not pinfo.error_func:
        errorlog.warning('no p_error() function is defined')
    grammar = Grammar(pinfo.tokens)
    for (term, assoc, level) in pinfo.preclist:
        try:
            grammar.set_precedence(term, assoc, level)
        except GrammarError as e:
            errorlog.warning('%s', e)
    for (funcname, gram) in pinfo.grammar:
        (file, line, prodname, syms) = gram
        try:
            grammar.add_production(prodname, syms, funcname, file, line)
        except GrammarError as e:
            errorlog.error('%s', e)
            errors = True
    try:
        if start is None:
            grammar.set_start(pinfo.start)
        else:
            grammar.set_start(start)
    except GrammarError as e:
        errorlog.error(str(e))
        errors = True
    if errors:
        raise YaccError('Unable to build parser')
    undefined_symbols = grammar.undefined_symbols()
    for (sym, prod) in undefined_symbols:
        errorlog.error('%s:%d: Symbol %r used, but not defined as a token or a rule', prod.file, prod.line, sym)
        errors = True
    unused_terminals = grammar.unused_terminals()
    if unused_terminals:
        debuglog.info('')
        debuglog.info('Unused terminals:')
        debuglog.info('')
        for term in unused_terminals:
            errorlog.warning('Token %r defined, but not used', term)
            debuglog.info('    %s', term)
    if debug:
        debuglog.info('')
        debuglog.info('Grammar')
        debuglog.info('')
        for (n, p) in enumerate(grammar.Productions):
            debuglog.info('Rule %-5d %s', n, p)
    unused_rules = grammar.unused_rules()
    for prod in unused_rules:
        errorlog.warning('%s:%d: Rule %r defined, but not used', prod.file, prod.line, prod.name)
    if len(unused_terminals) == 1:
        errorlog.warning('There is 1 unused token')
    if len(unused_terminals) > 1:
        errorlog.warning('There are %d unused tokens', len(unused_terminals))
    if len(unused_rules) == 1:
        errorlog.warning('There is 1 unused rule')
    if len(unused_rules) > 1:
        errorlog.warning('There are %d unused rules', len(unused_rules))
    if debug:
        debuglog.info('')
        debuglog.info('Terminals, with rules where they appear')
        debuglog.info('')
        terms = list(grammar.Terminals)
        terms.sort()
        for term in terms:
            debuglog.info('%-20s : %s', term, ' '.join([str(s) for s in grammar.Terminals[term]]))
        debuglog.info('')
        debuglog.info('Nonterminals, with rules where they appear')
        debuglog.info('')
        nonterms = list(grammar.Nonterminals)
        nonterms.sort()
        for nonterm in nonterms:
            debuglog.info('%-20s : %s', nonterm, ' '.join([str(s) for s in grammar.Nonterminals[nonterm]]))
        debuglog.info('')
    if check_recursion:
        unreachable = grammar.find_unreachable()
        for u in unreachable:
            errorlog.warning('Symbol %r is unreachable', u)
        infinite = grammar.infinite_cycles()
        for inf in infinite:
            errorlog.error('Infinite recursion detected for symbol %r', inf)
            errors = True
    unused_prec = grammar.unused_precedence()
    for (term, assoc) in unused_prec:
        errorlog.error('Precedence rule %r defined for unknown symbol %r', assoc, term)
        errors = True
    if errors:
        raise YaccError('Unable to build parser')
    if debug:
        errorlog.debug('Generating %s tables', method)
    lr = LRGeneratedTable(grammar, method, debuglog)
    if debug:
        num_sr = len(lr.sr_conflicts)
        if num_sr == 1:
            errorlog.warning('1 shift/reduce conflict')
        elif num_sr > 1:
            errorlog.warning('%d shift/reduce conflicts', num_sr)
        num_rr = len(lr.rr_conflicts)
        if num_rr == 1:
            errorlog.warning('1 reduce/reduce conflict')
        elif num_rr > 1:
            errorlog.warning('%d reduce/reduce conflicts', num_rr)
    if debug and (lr.sr_conflicts or lr.rr_conflicts):
        debuglog.warning('')
        debuglog.warning('Conflicts:')
        debuglog.warning('')
        for (state, tok, resolution) in lr.sr_conflicts:
            debuglog.warning('shift/reduce conflict for %s in state %d resolved as %s', tok, state, resolution)
        already_reported = set()
        for (state, rule, rejected) in lr.rr_conflicts:
            if (state, id(rule), id(rejected)) in already_reported:
                continue
            debuglog.warning('reduce/reduce conflict in state %d resolved using rule (%s)', state, rule)
            debuglog.warning('rejected rule (%s) in state %d', rejected, state)
            errorlog.warning('reduce/reduce conflict in state %d resolved using rule (%s)', state, rule)
            errorlog.warning('rejected rule (%s) in state %d', rejected, state)
            already_reported.add((state, id(rule), id(rejected)))
        warned_never = []
        for (state, rule, rejected) in lr.rr_conflicts:
            if not rejected.reduced and rejected not in warned_never:
                debuglog.warning('Rule (%s) is never reduced', rejected)
                errorlog.warning('Rule (%s) is never reduced', rejected)
                warned_never.append(rejected)
    if write_tables:
        try:
            lr.write_table(tabmodule, outputdir, signature)
            if tabmodule in sys.modules:
                del sys.modules[tabmodule]
        except IOError as e:
            errorlog.warning("Couldn't create %r. %s" % (tabmodule, e))
    if picklefile:
        try:
            lr.pickle_table(picklefile, signature)
        except IOError as e:
            errorlog.warning("Couldn't create %r. %s" % (picklefile, e))
    lr.bind_callables(pinfo.pdict)
    parser = LRParser(lr, pinfo.error_func)
    parse = parser.parse
    return parser