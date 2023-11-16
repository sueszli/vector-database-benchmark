"""InnerProduct circuit."""
from qiskit.circuit import QuantumRegister, QuantumCircuit

class InnerProduct(QuantumCircuit):
    """A 2n-qubit Boolean function that computes the inner product of
    two n-qubit vectors over :math:`F_2`.

    This implementation is a phase oracle which computes the following transform.

    .. math::

        \\mathcal{IP}_{2n} : F_2^{2n} \\rightarrow {-1, 1}
        \\mathcal{IP}_{2n}(x_1, \\cdots, x_n, y_1, \\cdots, y_n) = (-1)^{x.y}

    The corresponding unitary is a diagonal, which induces a -1 phase on any inputs
    where the inner product of the top and bottom registers is 1. Otherwise it keeps
    the input intact.

    .. parsed-literal::


        q0_0: ─■──────────
               │
        q0_1: ─┼──■───────
               │  │
        q0_2: ─┼──┼──■────
               │  │  │
        q0_3: ─┼──┼──┼──■─
               │  │  │  │
        q1_0: ─■──┼──┼──┼─
                  │  │  │
        q1_1: ────■──┼──┼─
                     │  │
        q1_2: ───────■──┼─
                        │
        q1_3: ──────────■─


    Reference Circuit:
        .. plot::

           from qiskit.circuit.library import InnerProduct
           from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
           circuit = InnerProduct(4)
           _generate_circuit_library_visualization(circuit)
    """

    def __init__(self, num_qubits: int) -> None:
        if False:
            i = 10
            return i + 15
        'Return a circuit to compute the inner product of 2 n-qubit registers.\n\n        Args:\n            num_qubits: width of top and bottom registers (half total circuit width)\n        '
        qr_a = QuantumRegister(num_qubits)
        qr_b = QuantumRegister(num_qubits)
        inner = QuantumCircuit(qr_a, qr_b, name='inner_product')
        for i in range(num_qubits):
            inner.cz(qr_a[i], qr_b[i])
        super().__init__(*inner.qregs, name='inner_product')
        self.compose(inner.to_gate(), qubits=self.qubits, inplace=True)