"""Reference circuits used by the tests."""
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister

class ReferenceCircuits:
    """Container for reference circuits used by the tests."""

    @staticmethod
    def bell():
        if False:
            while True:
                i = 10
        'Return a Bell circuit.'
        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='qc')
        qc = QuantumCircuit(qr, cr, name='bell')
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)
        return qc

    @staticmethod
    def bell_no_measure():
        if False:
            return 10
        'Return a Bell circuit.'
        qr = QuantumRegister(2, name='qr')
        qc = QuantumCircuit(qr, name='bell_no_measure')
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        return qc