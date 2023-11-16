"""

qasm.py - Functions to parse a set of qasm commands into a SymPy Circuit.

Examples taken from Chuang's page: https://web.archive.org/web/20220120121541/https://www.media.mit.edu/quanta/qasm2circ/

The code returns a circuit and an associated list of labels.

>>> from sympy.physics.quantum.qasm import Qasm
>>> q = Qasm('qubit q0', 'qubit q1', 'h q0', 'cnot q0,q1')
>>> q.get_circuit()
CNOT(1,0)*H(1)

>>> q = Qasm('qubit q0', 'qubit q1', 'cnot q0,q1', 'cnot q1,q0', 'cnot q0,q1')
>>> q.get_circuit()
CNOT(1,0)*CNOT(0,1)*CNOT(1,0)
"""
__all__ = ['Qasm']
from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T, CPHASE
from sympy.physics.quantum.circuitplot import Mz

def read_qasm(lines):
    if False:
        i = 10
        return i + 15
    return Qasm(*lines.splitlines())

def read_qasm_file(filename):
    if False:
        return 10
    return Qasm(*open(filename).readlines())

def flip_index(i, n):
    if False:
        print('Hello World!')
    'Reorder qubit indices from largest to smallest.\n\n    >>> from sympy.physics.quantum.qasm import flip_index\n    >>> flip_index(0, 2)\n    1\n    >>> flip_index(1, 2)\n    0\n    '
    return n - i - 1

def trim(line):
    if False:
        i = 10
        return i + 15
    "Remove everything following comment # characters in line.\n\n    >>> from sympy.physics.quantum.qasm import trim\n    >>> trim('nothing happens here')\n    'nothing happens here'\n    >>> trim('something #happens here')\n    'something '\n    "
    if '#' not in line:
        return line
    return line.split('#')[0]

def get_index(target, labels):
    if False:
        for i in range(10):
            print('nop')
    "Get qubit labels from the rest of the line,and return indices\n\n    >>> from sympy.physics.quantum.qasm import get_index\n    >>> get_index('q0', ['q0', 'q1'])\n    1\n    >>> get_index('q1', ['q0', 'q1'])\n    0\n    "
    nq = len(labels)
    return flip_index(labels.index(target), nq)

def get_indices(targets, labels):
    if False:
        while True:
            i = 10
    return [get_index(t, labels) for t in targets]

def nonblank(args):
    if False:
        for i in range(10):
            print('nop')
    for line in args:
        line = trim(line)
        if line.isspace():
            continue
        yield line
    return

def fullsplit(line):
    if False:
        while True:
            i = 10
    words = line.split()
    rest = ' '.join(words[1:])
    return (fixcommand(words[0]), [s.strip() for s in rest.split(',')])

def fixcommand(c):
    if False:
        i = 10
        return i + 15
    "Fix Qasm command names.\n\n    Remove all of forbidden characters from command c, and\n    replace 'def' with 'qdef'.\n    "
    forbidden_characters = ['-']
    c = c.lower()
    for char in forbidden_characters:
        c = c.replace(char, '')
    if c == 'def':
        return 'qdef'
    return c

def stripquotes(s):
    if False:
        while True:
            i = 10
    'Replace explicit quotes in a string.\n\n    >>> from sympy.physics.quantum.qasm import stripquotes\n    >>> stripquotes("\'S\'") == \'S\'\n    True\n    >>> stripquotes(\'"S"\') == \'S\'\n    True\n    >>> stripquotes(\'S\') == \'S\'\n    True\n    '
    s = s.replace('"', '')
    s = s.replace("'", '')
    return s

class Qasm:
    """Class to form objects from Qasm lines

    >>> from sympy.physics.quantum.qasm import Qasm
    >>> q = Qasm('qubit q0', 'qubit q1', 'h q0', 'cnot q0,q1')
    >>> q.get_circuit()
    CNOT(1,0)*H(1)
    >>> q = Qasm('qubit q0', 'qubit q1', 'cnot q0,q1', 'cnot q1,q0', 'cnot q0,q1')
    >>> q.get_circuit()
    CNOT(1,0)*CNOT(0,1)*CNOT(1,0)
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.defs = {}
        self.circuit = []
        self.labels = []
        self.inits = {}
        self.add(*args)
        self.kwargs = kwargs

    def add(self, *lines):
        if False:
            while True:
                i = 10
        for line in nonblank(lines):
            (command, rest) = fullsplit(line)
            if self.defs.get(command):
                function = self.defs.get(command)
                indices = self.indices(rest)
                if len(indices) == 1:
                    self.circuit.append(function(indices[0]))
                else:
                    self.circuit.append(function(indices[:-1], indices[-1]))
            elif hasattr(self, command):
                function = getattr(self, command)
                function(*rest)
            else:
                print('Function %s not defined. Skipping' % command)

    def get_circuit(self):
        if False:
            i = 10
            return i + 15
        return prod(reversed(self.circuit))

    def get_labels(self):
        if False:
            i = 10
            return i + 15
        return list(reversed(self.labels))

    def plot(self):
        if False:
            return 10
        from sympy.physics.quantum.circuitplot import CircuitPlot
        (circuit, labels) = (self.get_circuit(), self.get_labels())
        CircuitPlot(circuit, len(labels), labels=labels, inits=self.inits)

    def qubit(self, arg, init=None):
        if False:
            for i in range(10):
                print('nop')
        self.labels.append(arg)
        if init:
            self.inits[arg] = init

    def indices(self, args):
        if False:
            return 10
        return get_indices(args, self.labels)

    def index(self, arg):
        if False:
            print('Hello World!')
        return get_index(arg, self.labels)

    def nop(self, *args):
        if False:
            return 10
        pass

    def x(self, arg):
        if False:
            print('Hello World!')
        self.circuit.append(X(self.index(arg)))

    def z(self, arg):
        if False:
            return 10
        self.circuit.append(Z(self.index(arg)))

    def h(self, arg):
        if False:
            return 10
        self.circuit.append(H(self.index(arg)))

    def s(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.circuit.append(S(self.index(arg)))

    def t(self, arg):
        if False:
            return 10
        self.circuit.append(T(self.index(arg)))

    def measure(self, arg):
        if False:
            i = 10
            return i + 15
        self.circuit.append(Mz(self.index(arg)))

    def cnot(self, a1, a2):
        if False:
            for i in range(10):
                print('nop')
        self.circuit.append(CNOT(*self.indices([a1, a2])))

    def swap(self, a1, a2):
        if False:
            i = 10
            return i + 15
        self.circuit.append(SWAP(*self.indices([a1, a2])))

    def cphase(self, a1, a2):
        if False:
            return 10
        self.circuit.append(CPHASE(*self.indices([a1, a2])))

    def toffoli(self, a1, a2, a3):
        if False:
            return 10
        (i1, i2, i3) = self.indices([a1, a2, a3])
        self.circuit.append(CGateS((i1, i2), X(i3)))

    def cx(self, a1, a2):
        if False:
            return 10
        (fi, fj) = self.indices([a1, a2])
        self.circuit.append(CGate(fi, X(fj)))

    def cz(self, a1, a2):
        if False:
            for i in range(10):
                print('nop')
        (fi, fj) = self.indices([a1, a2])
        self.circuit.append(CGate(fi, Z(fj)))

    def defbox(self, *args):
        if False:
            return 10
        print('defbox not supported yet. Skipping: ', args)

    def qdef(self, name, ncontrols, symbol):
        if False:
            i = 10
            return i + 15
        from sympy.physics.quantum.circuitplot import CreateOneQubitGate, CreateCGate
        ncontrols = int(ncontrols)
        command = fixcommand(name)
        symbol = stripquotes(symbol)
        if ncontrols > 0:
            self.defs[command] = CreateCGate(symbol)
        else:
            self.defs[command] = CreateOneQubitGate(symbol)