"""Shor's algorithm and helper functions.

Todo:

* Get the CMod gate working again using the new Gate API.
* Fix everything.
* Update docstrings and reformat.
"""
import math
import random
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.intfunc import igcd
from sympy.ntheory import continued_fraction_periodic as continued_fraction
from sympy.utilities.iterables import variations
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import Qubit, measure_partial_oneshot
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qft import QFT
from sympy.physics.quantum.qexpr import QuantumError

class OrderFindingException(QuantumError):
    pass

class CMod(Gate):
    """A controlled mod gate.

    This is black box controlled Mod function for use by shor's algorithm.
    TODO: implement a decompose property that returns how to do this in terms
    of elementary gates
    """

    @classmethod
    def _eval_args(cls, args):
        if False:
            print('Hello World!')
        raise NotImplementedError('The CMod gate has not been completed.')

    @property
    def t(self):
        if False:
            return 10
        'Size of 1/2 input register.  First 1/2 holds output.'
        return self.label[0]

    @property
    def a(self):
        if False:
            for i in range(10):
                print('nop')
        'Base of the controlled mod function.'
        return self.label[1]

    @property
    def N(self):
        if False:
            i = 10
            return i + 15
        'N is the type of modular arithmetic we are doing.'
        return self.label[2]

    def _apply_operator_Qubit(self, qubits, **options):
        if False:
            while True:
                i = 10
        '\n            This directly calculates the controlled mod of the second half of\n            the register and puts it in the second\n            This will look pretty when we get Tensor Symbolically working\n        '
        n = 1
        k = 0
        for i in range(self.t):
            k += n * qubits[self.t + i]
            n *= 2
        out = int(self.a ** k % self.N)
        outarray = list(qubits.args[0][:self.t])
        for i in reversed(range(self.t)):
            outarray.append(out >> i & 1)
        return Qubit(*outarray)

def shor(N):
    if False:
        return 10
    "This function implements Shor's factoring algorithm on the Integer N\n\n    The algorithm starts by picking a random number (a) and seeing if it is\n    coprime with N. If it is not, then the gcd of the two numbers is a factor\n    and we are done. Otherwise, it begins the period_finding subroutine which\n    finds the period of a in modulo N arithmetic. This period, if even, can\n    be used to calculate factors by taking a**(r/2)-1 and a**(r/2)+1.\n    These values are returned.\n    "
    a = random.randrange(N - 2) + 2
    if igcd(N, a) != 1:
        return igcd(N, a)
    r = period_find(a, N)
    if r % 2 == 1:
        shor(N)
    answer = (igcd(a ** (r / 2) - 1, N), igcd(a ** (r / 2) + 1, N))
    return answer

def getr(x, y, N):
    if False:
        return 10
    fraction = continued_fraction(x, y)
    total = ratioize(fraction, N)
    return total

def ratioize(list, N):
    if False:
        while True:
            i = 10
    if list[0] > N:
        return S.Zero
    if len(list) == 1:
        return list[0]
    return list[0] + ratioize(list[1:], N)

def period_find(a, N):
    if False:
        i = 10
        return i + 15
    "Finds the period of a in modulo N arithmetic\n\n    This is quantum part of Shor's algorithm. It takes two registers,\n    puts first in superposition of states with Hadamards so: ``|k>|0>``\n    with k being all possible choices. It then does a controlled mod and\n    a QFT to determine the order of a.\n    "
    epsilon = 0.5
    t = int(2 * math.ceil(log(N, 2)))
    start = [0 for x in range(t)]
    factor = 1 / sqrt(2 ** t)
    qubits = 0
    for arr in variations(range(2), t, repetition=True):
        qbitArray = list(arr) + start
        qubits = qubits + Qubit(*qbitArray)
    circuit = (factor * qubits).expand()
    circuit = CMod(t, a, N) * circuit
    circuit = qapply(circuit)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i)
    circuit = qapply(QFT(t, t * 2).decompose() * circuit, floatingPoint=True)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i + t)
    if isinstance(circuit, Qubit):
        register = circuit
    elif isinstance(circuit, Mul):
        register = circuit.args[-1]
    else:
        register = circuit.args[-1].args[-1]
    n = 1
    answer = 0
    for i in range(len(register) / 2):
        answer += n * register[i + t]
        n = n << 1
    if answer == 0:
        raise OrderFindingException('Order finder returned 0. Happens with chance %f' % epsilon)
    g = getr(answer, 2 ** t, N)
    return g