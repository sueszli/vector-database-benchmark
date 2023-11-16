"""An implementation of qubits and gates acting on them.

Todo:

* Update docstrings.
* Update tests.
* Implement apply using decompose.
* Implement represent using decompose or something smarter. For this to
  work we first have to implement represent for SWAP.
* Decide if we want upper index to be inclusive in the constructor.
* Fix the printing of Rk gates in plotting.
"""
from sympy.core.expr import Expr
from sympy.core.numbers import I, Integer, pi
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.functions import sqrt
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError, QExpr
from sympy.matrices import eye
from sympy.physics.quantum.tensorproduct import matrix_tensor_product
from sympy.physics.quantum.gate import Gate, HadamardGate, SwapGate, OneQubitGate, CGate, PhaseGate, TGate, ZGate
from sympy.functions.elementary.complexes import sign
__all__ = ['QFT', 'IQFT', 'RkGate', 'Rk']

class RkGate(OneQubitGate):
    """This is the R_k gate of the QTF."""
    gate_name = 'Rk'
    gate_name_latex = 'R'

    def __new__(cls, *args):
        if False:
            return 10
        if len(args) != 2:
            raise QuantumError('Rk gates only take two arguments, got: %r' % args)
        target = args[0]
        k = args[1]
        if k == 1:
            return ZGate(target)
        elif k == 2:
            return PhaseGate(target)
        elif k == 3:
            return TGate(target)
        args = cls._eval_args(args)
        inst = Expr.__new__(cls, *args)
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _eval_args(cls, args):
        if False:
            print('Hello World!')
        return QExpr._eval_args(args)

    @property
    def k(self):
        if False:
            for i in range(10):
                print('nop')
        return self.label[1]

    @property
    def targets(self):
        if False:
            return 10
        return self.label[:1]

    @property
    def gate_name_plot(self):
        if False:
            for i in range(10):
                print('nop')
        return '$%s_%s$' % (self.gate_name_latex, str(self.k))

    def get_target_matrix(self, format='sympy'):
        if False:
            return 10
        if format == 'sympy':
            return Matrix([[1, 0], [0, exp(sign(self.k) * Integer(2) * pi * I / Integer(2) ** abs(self.k))]])
        raise NotImplementedError('Invalid format for the R_k gate: %r' % format)
Rk = RkGate

class Fourier(Gate):
    """Superclass of Quantum Fourier and Inverse Quantum Fourier Gates."""

    @classmethod
    def _eval_args(self, args):
        if False:
            print('Hello World!')
        if len(args) != 2:
            raise QuantumError('QFT/IQFT only takes two arguments, got: %r' % args)
        if args[0] >= args[1]:
            raise QuantumError('Start must be smaller than finish')
        return Gate._eval_args(args)

    def _represent_default_basis(self, **options):
        if False:
            return 10
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        if False:
            i = 10
            return i + 15
        '\n            Represents the (I)QFT In the Z Basis\n        '
        nqubits = options.get('nqubits', 0)
        if nqubits == 0:
            raise QuantumError('The number of qubits must be given as nqubits.')
        if nqubits < self.min_qubits:
            raise QuantumError('The number of qubits %r is too small for the gate.' % nqubits)
        size = self.size
        omega = self.omega
        arrayFT = [[omega ** (i * j % size) / sqrt(size) for i in range(size)] for j in range(size)]
        matrixFT = Matrix(arrayFT)
        if self.label[0] != 0:
            matrixFT = matrix_tensor_product(eye(2 ** self.label[0]), matrixFT)
        if self.min_qubits < nqubits:
            matrixFT = matrix_tensor_product(matrixFT, eye(2 ** (nqubits - self.min_qubits)))
        return matrixFT

    @property
    def targets(self):
        if False:
            print('Hello World!')
        return range(self.label[0], self.label[1])

    @property
    def min_qubits(self):
        if False:
            while True:
                i = 10
        return self.label[1]

    @property
    def size(self):
        if False:
            print('Hello World!')
        'Size is the size of the QFT matrix'
        return 2 ** (self.label[1] - self.label[0])

    @property
    def omega(self):
        if False:
            i = 10
            return i + 15
        return Symbol('omega')

class QFT(Fourier):
    """The forward quantum Fourier transform."""
    gate_name = 'QFT'
    gate_name_latex = 'QFT'

    def decompose(self):
        if False:
            while True:
                i = 10
        'Decomposes QFT into elementary gates.'
        start = self.label[0]
        finish = self.label[1]
        circuit = 1
        for level in reversed(range(start, finish)):
            circuit = HadamardGate(level) * circuit
            for i in range(level - start):
                circuit = CGate(level - i - 1, RkGate(level, i + 2)) * circuit
        for i in range((finish - start) // 2):
            circuit = SwapGate(i + start, finish - i - 1) * circuit
        return circuit

    def _apply_operator_Qubit(self, qubits, **options):
        if False:
            return 10
        return qapply(self.decompose() * qubits)

    def _eval_inverse(self):
        if False:
            for i in range(10):
                print('nop')
        return IQFT(*self.args)

    @property
    def omega(self):
        if False:
            for i in range(10):
                print('nop')
        return exp(2 * pi * I / self.size)

class IQFT(Fourier):
    """The inverse quantum Fourier transform."""
    gate_name = 'IQFT'
    gate_name_latex = '{QFT^{-1}}'

    def decompose(self):
        if False:
            for i in range(10):
                print('nop')
        'Decomposes IQFT into elementary gates.'
        start = self.args[0]
        finish = self.args[1]
        circuit = 1
        for i in range((finish - start) // 2):
            circuit = SwapGate(i + start, finish - i - 1) * circuit
        for level in range(start, finish):
            for i in reversed(range(level - start)):
                circuit = CGate(level - i - 1, RkGate(level, -i - 2)) * circuit
            circuit = HadamardGate(level) * circuit
        return circuit

    def _eval_inverse(self):
        if False:
            i = 10
            return i + 15
        return QFT(*self.args)

    @property
    def omega(self):
        if False:
            return 10
        return exp(-2 * pi * I / self.size)