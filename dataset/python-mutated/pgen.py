from io import StringIO
from . import grammar
from . import token
from . import tokenize

class PgenGrammar(grammar.Grammar):
    pass

class ParserGenerator(object):

    def __init__(self, filename=None, stream=None):
        if False:
            i = 10
            return i + 15
        close_stream = None
        if filename is None and stream is None:
            raise RuntimeError('Either a filename or a stream is expected, both were none')
        if stream is None:
            stream = open(filename, encoding='utf-8')
            close_stream = stream.close
        self.filename = filename
        self.stream = stream
        self.generator = tokenize.generate_tokens(stream.readline)
        self.gettoken()
        (self.dfas, self.startsymbol) = self.parse()
        if close_stream is not None:
            close_stream()
        self.first = {}
        self.addfirstsets()

    def make_grammar(self):
        if False:
            return 10
        c = PgenGrammar()
        names = list(self.dfas.keys())
        names.sort()
        names.remove(self.startsymbol)
        names.insert(0, self.startsymbol)
        for name in names:
            i = 256 + len(c.symbol2number)
            c.symbol2number[name] = i
            c.number2symbol[i] = name
        for name in names:
            dfa = self.dfas[name]
            states = []
            for state in dfa:
                arcs = []
                for (label, next) in sorted(state.arcs.items()):
                    arcs.append((self.make_label(c, label), dfa.index(next)))
                if state.isfinal:
                    arcs.append((0, dfa.index(state)))
                states.append(arcs)
            c.states.append(states)
            c.dfas[c.symbol2number[name]] = (states, self.make_first(c, name))
        c.start = c.symbol2number[self.startsymbol]
        return c

    def make_first(self, c, name):
        if False:
            return 10
        rawfirst = self.first[name]
        first = {}
        for label in sorted(rawfirst):
            ilabel = self.make_label(c, label)
            first[ilabel] = 1
        return first

    def make_label(self, c, label):
        if False:
            print('Hello World!')
        ilabel = len(c.labels)
        if label[0].isalpha():
            if label in c.symbol2number:
                if label in c.symbol2label:
                    return c.symbol2label[label]
                else:
                    c.labels.append((c.symbol2number[label], None))
                    c.symbol2label[label] = ilabel
                    return ilabel
            else:
                itoken = getattr(token, label, None)
                assert isinstance(itoken, int), label
                assert itoken in token.tok_name, label
                if itoken in c.tokens:
                    return c.tokens[itoken]
                else:
                    c.labels.append((itoken, None))
                    c.tokens[itoken] = ilabel
                    return ilabel
        else:
            assert label[0] in ('"', "'"), label
            value = eval(label)
            if value[0].isalpha():
                if label[0] == '"':
                    keywords = c.soft_keywords
                else:
                    keywords = c.keywords
                if value in keywords:
                    return keywords[value]
                else:
                    c.labels.append((token.NAME, value))
                    keywords[value] = ilabel
                    return ilabel
            else:
                itoken = grammar.opmap[value]
                if itoken in c.tokens:
                    return c.tokens[itoken]
                else:
                    c.labels.append((itoken, None))
                    c.tokens[itoken] = ilabel
                    return ilabel

    def addfirstsets(self):
        if False:
            i = 10
            return i + 15
        names = list(self.dfas.keys())
        names.sort()
        for name in names:
            if name not in self.first:
                self.calcfirst(name)

    def calcfirst(self, name):
        if False:
            while True:
                i = 10
        dfa = self.dfas[name]
        self.first[name] = None
        state = dfa[0]
        totalset = {}
        overlapcheck = {}
        for (label, next) in state.arcs.items():
            if label in self.dfas:
                if label in self.first:
                    fset = self.first[label]
                    if fset is None:
                        raise ValueError('recursion for rule %r' % name)
                else:
                    self.calcfirst(label)
                    fset = self.first[label]
                totalset.update(fset)
                overlapcheck[label] = fset
            else:
                totalset[label] = 1
                overlapcheck[label] = {label: 1}
        inverse = {}
        for (label, itsfirst) in overlapcheck.items():
            for symbol in itsfirst:
                if symbol in inverse:
                    raise ValueError('rule %s is ambiguous; %s is in the first sets of %s as well as %s' % (name, symbol, label, inverse[symbol]))
                inverse[symbol] = label
        self.first[name] = totalset

    def parse(self):
        if False:
            for i in range(10):
                print('nop')
        dfas = {}
        startsymbol = None
        while self.type != token.ENDMARKER:
            while self.type == token.NEWLINE:
                self.gettoken()
            name = self.expect(token.NAME)
            self.expect(token.OP, ':')
            (a, z) = self.parse_rhs()
            self.expect(token.NEWLINE)
            dfa = self.make_dfa(a, z)
            self.simplify_dfa(dfa)
            dfas[name] = dfa
            if startsymbol is None:
                startsymbol = name
        return (dfas, startsymbol)

    def make_dfa(self, start, finish):
        if False:
            i = 10
            return i + 15
        assert isinstance(start, NFAState)
        assert isinstance(finish, NFAState)

        def closure(state):
            if False:
                for i in range(10):
                    print('nop')
            base = {}
            addclosure(state, base)
            return base

        def addclosure(state, base):
            if False:
                i = 10
                return i + 15
            assert isinstance(state, NFAState)
            if state in base:
                return
            base[state] = 1
            for (label, next) in state.arcs:
                if label is None:
                    addclosure(next, base)
        states = [DFAState(closure(start), finish)]
        for state in states:
            arcs = {}
            for nfastate in state.nfaset:
                for (label, next) in nfastate.arcs:
                    if label is not None:
                        addclosure(next, arcs.setdefault(label, {}))
            for (label, nfaset) in sorted(arcs.items()):
                for st in states:
                    if st.nfaset == nfaset:
                        break
                else:
                    st = DFAState(nfaset, finish)
                    states.append(st)
                state.addarc(st, label)
        return states

    def dump_nfa(self, name, start, finish):
        if False:
            i = 10
            return i + 15
        print('Dump of NFA for', name)
        todo = [start]
        for (i, state) in enumerate(todo):
            print('  State', i, state is finish and '(final)' or '')
            for (label, next) in state.arcs:
                if next in todo:
                    j = todo.index(next)
                else:
                    j = len(todo)
                    todo.append(next)
                if label is None:
                    print('    -> %d' % j)
                else:
                    print('    %s -> %d' % (label, j))

    def dump_dfa(self, name, dfa):
        if False:
            i = 10
            return i + 15
        print('Dump of DFA for', name)
        for (i, state) in enumerate(dfa):
            print('  State', i, state.isfinal and '(final)' or '')
            for (label, next) in sorted(state.arcs.items()):
                print('    %s -> %d' % (label, dfa.index(next)))

    def simplify_dfa(self, dfa):
        if False:
            while True:
                i = 10
        changes = True
        while changes:
            changes = False
            for (i, state_i) in enumerate(dfa):
                for j in range(i + 1, len(dfa)):
                    state_j = dfa[j]
                    if state_i == state_j:
                        del dfa[j]
                        for state in dfa:
                            state.unifystate(state_j, state_i)
                        changes = True
                        break

    def parse_rhs(self):
        if False:
            for i in range(10):
                print('nop')
        (a, z) = self.parse_alt()
        if self.value != '|':
            return (a, z)
        else:
            aa = NFAState()
            zz = NFAState()
            aa.addarc(a)
            z.addarc(zz)
            while self.value == '|':
                self.gettoken()
                (a, z) = self.parse_alt()
                aa.addarc(a)
                z.addarc(zz)
            return (aa, zz)

    def parse_alt(self):
        if False:
            return 10
        (a, b) = self.parse_item()
        while self.value in ('(', '[') or self.type in (token.NAME, token.STRING):
            (c, d) = self.parse_item()
            b.addarc(c)
            b = d
        return (a, b)

    def parse_item(self):
        if False:
            for i in range(10):
                print('nop')
        if self.value == '[':
            self.gettoken()
            (a, z) = self.parse_rhs()
            self.expect(token.OP, ']')
            a.addarc(z)
            return (a, z)
        else:
            (a, z) = self.parse_atom()
            value = self.value
            if value not in ('+', '*'):
                return (a, z)
            self.gettoken()
            z.addarc(a)
            if value == '+':
                return (a, z)
            else:
                return (a, a)

    def parse_atom(self):
        if False:
            i = 10
            return i + 15
        if self.value == '(':
            self.gettoken()
            (a, z) = self.parse_rhs()
            self.expect(token.OP, ')')
            return (a, z)
        elif self.type in (token.NAME, token.STRING):
            a = NFAState()
            z = NFAState()
            a.addarc(z, self.value)
            self.gettoken()
            return (a, z)
        else:
            self.raise_error('expected (...) or NAME or STRING, got %s/%s', self.type, self.value)

    def expect(self, type, value=None):
        if False:
            print('Hello World!')
        if self.type != type or (value is not None and self.value != value):
            self.raise_error('expected %s/%s, got %s/%s', type, value, self.type, self.value)
        value = self.value
        self.gettoken()
        return value

    def gettoken(self):
        if False:
            i = 10
            return i + 15
        tup = next(self.generator)
        while tup[0] in (tokenize.COMMENT, tokenize.NL):
            tup = next(self.generator)
        (self.type, self.value, self.begin, self.end, self.line) = tup

    def raise_error(self, msg, *args):
        if False:
            while True:
                i = 10
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = ' '.join([msg] + list(map(str, args)))
        raise SyntaxError(msg, (self.filename, self.end[0], self.end[1], self.line))

class NFAState(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.arcs = []

    def addarc(self, next, label=None):
        if False:
            while True:
                i = 10
        assert label is None or isinstance(label, str)
        assert isinstance(next, NFAState)
        self.arcs.append((label, next))

class DFAState(object):

    def __init__(self, nfaset, final):
        if False:
            while True:
                i = 10
        assert isinstance(nfaset, dict)
        assert isinstance(next(iter(nfaset)), NFAState)
        assert isinstance(final, NFAState)
        self.nfaset = nfaset
        self.isfinal = final in nfaset
        self.arcs = {}

    def addarc(self, next, label):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(label, str)
        assert label not in self.arcs
        assert isinstance(next, DFAState)
        self.arcs[label] = next

    def unifystate(self, old, new):
        if False:
            return 10
        for (label, next) in self.arcs.items():
            if next is old:
                self.arcs[label] = new

    def __eq__(self, other):
        if False:
            return 10
        assert isinstance(other, DFAState)
        if self.isfinal != other.isfinal:
            return False
        if len(self.arcs) != len(other.arcs):
            return False
        for (label, next) in self.arcs.items():
            if next is not other.arcs.get(label):
                return False
        return True
    __hash__ = None

def generate_grammar(filename_or_stream='Grammar.txt'):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(filename_or_stream, str):
        p = ParserGenerator(filename_or_stream)
    elif isinstance(filename_or_stream, StringIO):
        p = ParserGenerator(stream=filename_or_stream)
    else:
        raise NotImplementedError('Type %s not implemented' % type(filename_or_stream))
    return p.make_grammar()