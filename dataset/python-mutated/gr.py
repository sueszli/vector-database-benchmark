"""Global R gates."""
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit

class GR(QuantumCircuit):
    """Global R gate.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────┐
        q_0: ┤0         ├
             │          │
        q_1: ┤1 GR(ϴ,φ) ├
             │          │
        q_2: ┤2         ├
             └──────────┘

    The global R gate is native to atomic systems (ion traps, cold neutrals). The global R
    can be applied to multiple qubits simultaneously.

    In the one-qubit case, this is equivalent to an R(theta, phi) operation,
    and is thus reduced to the RGate. The global R gate is a direct sum of R
    operations on all individual qubits.

    .. math::

        GR(\\theta, \\phi) = \\exp(-i \\sum_{i=1}^{n} (\\cos(\\phi)X_i + \\sin(\\phi)Y_i) \\theta/2)

    **Expanded Circuit:**

    .. plot::

       from qiskit.circuit.library import GR
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       import numpy as np
       circuit = GR(num_qubits=3, theta=np.pi/4, phi=np.pi/2)
       _generate_circuit_library_visualization(circuit)

    """

    def __init__(self, num_qubits: int, theta: float, phi: float) -> None:
        if False:
            return 10
        'Create a new Global R (GR) gate.\n\n        Args:\n            num_qubits: number of qubits.\n            theta: rotation angle about axis determined by phi\n            phi: angle of rotation axis in xy-plane\n        '
        name = f'GR({theta:.2f}, {phi:.2f})'
        circuit = QuantumCircuit(num_qubits, name=name)
        circuit.r(theta, phi, circuit.qubits)
        super().__init__(num_qubits, name=name)
        self.append(circuit.to_gate(), self.qubits)

class GRX(GR):
    """Global RX gate.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────┐
        q_0: ┤0         ├
             │          │
        q_1: ┤1  GRX(ϴ) ├
             │          │
        q_2: ┤2         ├
             └──────────┘

    The global RX gate is native to atomic systems (ion traps, cold neutrals). The global RX
    can be applied to multiple qubits simultaneously.

    In the one-qubit case, this is equivalent to an RX(theta) operations,
    and is thus reduced to the RXGate. The global RX gate is a direct sum of RX
    operations on all individual qubits.

    .. math::

        GRX(\\theta) = \\exp(-i \\sum_{i=1}^{n} X_i \\theta/2)

    **Expanded Circuit:**

    .. plot::

        from qiskit.circuit.library import GRX
        from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
        import numpy as np
        circuit = GRX(num_qubits=3, theta=np.pi/4)
        _generate_circuit_library_visualization(circuit)

    """

    def __init__(self, num_qubits: int, theta: float) -> None:
        if False:
            return 10
        'Create a new Global RX (GRX) gate.\n\n        Args:\n            num_qubits: number of qubits.\n            theta: rotation angle about x-axis\n        '
        super().__init__(num_qubits, theta, phi=0)

class GRY(GR):
    """Global RY gate.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────┐
        q_0: ┤0         ├
             │          │
        q_1: ┤1  GRY(ϴ) ├
             │          │
        q_2: ┤2         ├
             └──────────┘

    The global RY gate is native to atomic systems (ion traps, cold neutrals). The global RY
    can be applied to multiple qubits simultaneously.

    In the one-qubit case, this is equivalent to an RY(theta) operation,
    and is thus reduced to the RYGate. The global RY gate is a direct sum of RY
    operations on all individual qubits.

    .. math::

        GRY(\\theta) = \\exp(-i \\sum_{i=1}^{n} Y_i \\theta/2)

    **Expanded Circuit:**

    .. plot::

       from qiskit.circuit.library import GRY
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       import numpy as np
       circuit = GRY(num_qubits=3, theta=np.pi/4)
       _generate_circuit_library_visualization(circuit)

    """

    def __init__(self, num_qubits: int, theta: float) -> None:
        if False:
            i = 10
            return i + 15
        'Create a new Global RY (GRY) gate.\n\n        Args:\n            num_qubits: number of qubits.\n            theta: rotation angle about y-axis\n        '
        super().__init__(num_qubits, theta, phi=np.pi / 2)

class GRZ(QuantumCircuit):
    """Global RZ gate.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────┐
        q_0: ┤0         ├
             │          │
        q_1: ┤1  GRZ(φ) ├
             │          │
        q_2: ┤2         ├
             └──────────┘

    The global RZ gate is native to atomic systems (ion traps, cold neutrals). The global RZ
    can be applied to multiple qubits simultaneously.

    In the one-qubit case, this is equivalent to an RZ(phi) operation,
    and is thus reduced to the RZGate. The global RZ gate is a direct sum of RZ
    operations on all individual qubits.

    .. math::

        GRZ(\\phi) = \\exp(-i \\sum_{i=1}^{n} Z_i \\phi)

    **Expanded Circuit:**

    .. plot::

       from qiskit.circuit.library import GRZ
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       import numpy as np
       circuit = GRZ(num_qubits=3, phi=np.pi/2)
       _generate_circuit_library_visualization(circuit)

    """

    def __init__(self, num_qubits: int, phi: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create a new Global RZ (GRZ) gate.\n\n        Args:\n            num_qubits: number of qubits.\n            phi: rotation angle about z-axis\n        '
        super().__init__(num_qubits, name='grz')
        self.rz(phi, self.qubits)