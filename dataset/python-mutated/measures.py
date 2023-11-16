"""
Quantum information measures, metrics, and related functions for states.
"""
from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.utils import partial_trace, shannon_entropy, _format_state, _funm_svd

def state_fidelity(state1: Statevector | DensityMatrix, state2: Statevector | DensityMatrix, validate: bool=True) -> float:
    if False:
        return 10
    'Return the state fidelity between two quantum states.\n\n    The state fidelity :math:`F` for density matrix input states\n    :math:`\\rho_1, \\rho_2` is given by\n\n    .. math::\n        F(\\rho_1, \\rho_2) = Tr[\\sqrt{\\sqrt{\\rho_1}\\rho_2\\sqrt{\\rho_1}}]^2.\n\n    If one of the states is a pure state this simplifies to\n    :math:`F(\\rho_1, \\rho_2) = \\langle\\psi_1|\\rho_2|\\psi_1\\rangle`, where\n    :math:`\\rho_1 = |\\psi_1\\rangle\\!\\langle\\psi_1|`.\n\n    Args:\n        state1 (Statevector or DensityMatrix): the first quantum state.\n        state2 (Statevector or DensityMatrix): the second quantum state.\n        validate (bool): check if the inputs are valid quantum states\n                         [Default: True]\n\n    Returns:\n        float: The state fidelity :math:`F(\\rho_1, \\rho_2)`.\n\n    Raises:\n        QiskitError: if ``validate=True`` and the inputs are invalid quantum states.\n    '
    state1 = _format_state(state1, validate=validate)
    state2 = _format_state(state2, validate=validate)
    arr1 = state1.data
    arr2 = state2.data
    if isinstance(state1, Statevector):
        if isinstance(state2, Statevector):
            fid = np.abs(arr2.conj().dot(arr1)) ** 2
        else:
            fid = arr1.conj().dot(arr2).dot(arr1)
    elif isinstance(state2, Statevector):
        fid = arr2.conj().dot(arr1).dot(arr2)
    else:
        s1sq = _funm_svd(arr1, np.sqrt)
        s2sq = _funm_svd(arr2, np.sqrt)
        fid = np.linalg.norm(s1sq.dot(s2sq), ord='nuc') ** 2
    return float(np.real(fid))

def purity(state: Statevector | DensityMatrix, validate: bool=True) -> float:
    if False:
        for i in range(10):
            print('nop')
    "Calculate the purity of a quantum state.\n\n    The purity of a density matrix :math:`\\rho` is\n\n    .. math::\n\n        \\text{Purity}(\\rho) = Tr[\\rho^2]\n\n    Args:\n        state (Statevector or DensityMatrix): a quantum state.\n        validate (bool): check if input state is valid [Default: True]\n\n    Returns:\n        float: the purity :math:`Tr[\\rho^2]`.\n\n    Raises:\n        QiskitError: if the input isn't a valid quantum state.\n    "
    state = _format_state(state, validate=validate)
    return state.purity()

def entropy(state: Statevector | DensityMatrix, base: int=2) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Calculate the von-Neumann entropy of a quantum state.\n\n    The entropy :math:`S` is given by\n\n    .. math::\n\n        S(\\rho) = - Tr[\\rho \\log(\\rho)]\n\n    Args:\n        state (Statevector or DensityMatrix): a quantum state.\n        base (int): the base of the logarithm [Default: 2].\n\n    Returns:\n        float: The von-Neumann entropy S(rho).\n\n    Raises:\n        QiskitError: if the input state is not a valid QuantumState.\n    '
    import scipy.linalg as la
    state = _format_state(state, validate=True)
    if isinstance(state, Statevector):
        return 0
    evals = np.maximum(np.real(la.eigvals(state.data)), 0.0)
    return shannon_entropy(evals, base=base)

def mutual_information(state: Statevector | DensityMatrix, base: int=2) -> float:
    if False:
        return 10
    'Calculate the mutual information of a bipartite state.\n\n    The mutual information :math:`I` is given by:\n\n    .. math::\n\n        I(\\rho_{AB}) = S(\\rho_A) + S(\\rho_B) - S(\\rho_{AB})\n\n    where :math:`\\rho_A=Tr_B[\\rho_{AB}], \\rho_B=Tr_A[\\rho_{AB}]`, are the\n    reduced density matrices of the bipartite state :math:`\\rho_{AB}`.\n\n    Args:\n        state (Statevector or DensityMatrix): a bipartite state.\n        base (int): the base of the logarithm [Default: 2].\n\n    Returns:\n        float: The mutual information :math:`I(\\rho_{AB})`.\n\n    Raises:\n        QiskitError: if the input state is not a valid QuantumState.\n        QiskitError: if input is not a bipartite QuantumState.\n    '
    state = _format_state(state, validate=True)
    if len(state.dims()) != 2:
        raise QiskitError('Input must be a bipartite quantum state.')
    rho_a = partial_trace(state, [1])
    rho_b = partial_trace(state, [0])
    return entropy(rho_a, base=base) + entropy(rho_b, base=base) - entropy(state, base=base)

def concurrence(state: Statevector | DensityMatrix) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Calculate the concurrence of a quantum state.\n\n    The concurrence of a bipartite\n    :class:`~qiskit.quantum_info.Statevector` :math:`|\\psi\\rangle` is\n    given by\n\n    .. math::\n\n        C(|\\psi\\rangle) = \\sqrt{2(1 - Tr[\\rho_0^2])}\n\n    where :math:`\\rho_0 = Tr_1[|\\psi\\rangle\\!\\langle\\psi|]` is the\n    reduced state from by taking the\n    :func:`~qiskit.quantum_info.partial_trace` of the input state.\n\n    For density matrices the concurrence is only defined for\n    2-qubit states, it is given by:\n\n    .. math::\n\n        C(\\rho) = \\max(0, \\lambda_1 - \\lambda_2 - \\lambda_3 - \\lambda_4)\n\n    where  :math:`\\lambda _1 \\ge \\lambda _2 \\ge \\lambda _3 \\ge \\lambda _4`\n    are the ordered eigenvalues of the matrix\n    :math:`R=\\sqrt{\\sqrt{\\rho }(Y\\otimes Y)\\overline{\\rho}(Y\\otimes Y)\\sqrt{\\rho}}`.\n\n    Args:\n        state (Statevector or DensityMatrix): a 2-qubit quantum state.\n\n    Returns:\n        float: The concurrence.\n\n    Raises:\n        QiskitError: if the input state is not a valid QuantumState.\n        QiskitError: if input is not a bipartite QuantumState.\n        QiskitError: if density matrix input is not a 2-qubit state.\n    '
    import scipy.linalg as la
    state = _format_state(state, validate=True)
    if isinstance(state, Statevector):
        dims = state.dims()
        if len(dims) != 2:
            raise QiskitError('Input is not a bipartite quantum state.')
        qargs = [0] if dims[0] > dims[1] else [1]
        rho = partial_trace(state, qargs)
        return float(np.sqrt(2 * (1 - np.real(purity(rho)))))
    if state.dim != 4:
        raise QiskitError('Input density matrix must be a 2-qubit state.')
    rho = DensityMatrix(state).data
    yy_mat = np.fliplr(np.diag([-1, 1, 1, -1]))
    sigma = rho.dot(yy_mat).dot(rho.conj()).dot(yy_mat)
    w = np.sort(np.real(la.eigvals(sigma)))
    w = np.sqrt(np.maximum(w, 0.0))
    return max(0.0, w[-1] - np.sum(w[0:-1]))

def entanglement_of_formation(state: Statevector | DensityMatrix) -> float:
    if False:
        i = 10
        return i + 15
    'Calculate the entanglement of formation of quantum state.\n\n    The input quantum state must be either a bipartite state vector, or a\n    2-qubit density matrix.\n\n    Args:\n        state (Statevector or DensityMatrix): a 2-qubit quantum state.\n\n    Returns:\n        float: The entanglement of formation.\n\n    Raises:\n        QiskitError: if the input state is not a valid QuantumState.\n        QiskitError: if input is not a bipartite QuantumState.\n        QiskitError: if density matrix input is not a 2-qubit state.\n    '
    state = _format_state(state, validate=True)
    if isinstance(state, Statevector):
        dims = state.dims()
        if len(dims) != 2:
            raise QiskitError('Input is not a bipartite quantum state.')
        qargs = [0] if dims[0] > dims[1] else [1]
        return entropy(partial_trace(state, qargs), base=2)
    if state.dim != 4:
        raise QiskitError('Input density matrix must be a 2-qubit state.')
    conc = concurrence(state)
    val = (1 + np.sqrt(1 - conc ** 2)) / 2
    return shannon_entropy([val, 1 - val])

def negativity(state, qargs):
    if False:
        for i in range(10):
            print('nop')
    'Calculates the negativity\n\n    The mathematical expression for negativity is given by:\n    .. math::\n        {\\cal{N}}(\\rho) = \\frac{|| \\rho^{T_A}|| - 1 }{2}\n\n    Args:\n        state (Statevector or DensityMatrix): a quantum state.\n        qargs (list): The subsystems to be transposed.\n\n    Returns:\n        negv (float): Negativity value of the quantum state\n\n    Raises:\n        QiskitError: if the input state is not a valid QuantumState.\n    '
    if isinstance(state, Statevector):
        state = DensityMatrix(state)
    state = state.partial_transpose(qargs)
    singular_values = np.linalg.svd(state.data, compute_uv=False)
    eigvals = np.sum(singular_values)
    negv = (eigvals - 1) / 2
    return negv