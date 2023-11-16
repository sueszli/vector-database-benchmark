"""Contains a Python simulator that returns the unitary of the circuit.

It simulates a unitary of a quantum circuit that has been compiled to run on
the simulator. It is exponential in the number of qubits.

.. code-block:: python

    UnitarySimulator().run(qobj)

Where the input is a Qobj object and the output is a BasicAerJob object, which can
later be queried for the Result object. The result will contain a 'unitary'
data field, which is a 2**n x 2**n complex numpy array representing the
circuit's unitary matrix.
"""
import logging
import uuid
import time
from math import log2, sqrt
import warnings
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.utils.multiprocessing import local_hardware_info
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.providers.backend import BackendV1
from qiskit.providers.options import Options
from qiskit.providers.basicaer.basicaerjob import BasicAerJob
from qiskit.result import Result
from .exceptions import BasicAerError
from .basicaertools import single_gate_matrix
from .basicaertools import SINGLE_QUBIT_GATES
from .basicaertools import cx_gate_matrix
from .basicaertools import einsum_matmul_index
logger = logging.getLogger(__name__)

class UnitarySimulatorPy(BackendV1):
    """Python implementation of a unitary simulator."""
    MAX_QUBITS_MEMORY = int(log2(sqrt(local_hardware_info()['memory'] * 1024 ** 3 / 16)))
    DEFAULT_CONFIGURATION = {'backend_name': 'unitary_simulator', 'backend_version': '1.1.0', 'n_qubits': min(24, MAX_QUBITS_MEMORY), 'url': 'https://github.com/Qiskit/qiskit-terra', 'simulator': True, 'local': True, 'conditional': False, 'open_pulse': False, 'memory': False, 'max_shots': 0, 'coupling_map': None, 'description': 'A python simulator for unitary matrix corresponding to a circuit', 'basis_gates': ['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id', 'unitary'], 'gates': [{'name': 'u1', 'parameters': ['lambda'], 'qasm_def': 'gate u1(lambda) q { U(0,0,lambda) q; }'}, {'name': 'u2', 'parameters': ['phi', 'lambda'], 'qasm_def': 'gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }'}, {'name': 'u3', 'parameters': ['theta', 'phi', 'lambda'], 'qasm_def': 'gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }'}, {'name': 'rz', 'parameters': ['phi'], 'qasm_def': 'gate rz(phi) q { U(0,0,phi) q; }'}, {'name': 'sx', 'parameters': [], 'qasm_def': 'gate sx(phi) q { U(pi/2,7*pi/2,pi/2) q; }'}, {'name': 'x', 'parameters': [], 'qasm_def': 'gate x q { U(pi,7*pi/2,pi/2) q; }'}, {'name': 'cx', 'parameters': [], 'qasm_def': 'gate cx c,t { CX c,t; }'}, {'name': 'id', 'parameters': [], 'qasm_def': 'gate id a { U(0,0,0) a; }'}, {'name': 'unitary', 'parameters': ['matrix'], 'qasm_def': 'unitary(matrix) q1, q2,...'}]}
    DEFAULT_OPTIONS = {'initial_unitary': None, 'chop_threshold': 1e-15}

    def __init__(self, configuration=None, provider=None, **fields):
        if False:
            return 10
        super().__init__(configuration=configuration or QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION), provider=provider, **fields)
        self._unitary = None
        self._number_of_qubits = 0
        self._initial_unitary = None
        self._global_phase = 0
        self._chop_threshold = self.options.get('chop_threshold')

    @classmethod
    def _default_options(cls):
        if False:
            return 10
        return Options(shots=1, initial_unitary=None, chop_threshold=1e-15, parameter_binds=None)

    def _add_unitary(self, gate, qubits):
        if False:
            i = 10
            return i + 15
        'Apply an N-qubit unitary matrix.\n\n        Args:\n            gate (matrix_like): an N-qubit unitary matrix\n            qubits (list): the list of N-qubits.\n        '
        num_qubits = len(qubits)
        indexes = einsum_matmul_index(qubits, self._number_of_qubits)
        gate_tensor = np.reshape(np.array(gate, dtype=complex), num_qubits * [2, 2])
        self._unitary = np.einsum(indexes, gate_tensor, self._unitary, dtype=complex, casting='no')

    def _validate_initial_unitary(self):
        if False:
            for i in range(10):
                print('nop')
        'Validate an initial unitary matrix'
        if self._initial_unitary is None:
            return
        shape = np.shape(self._initial_unitary)
        required_shape = (2 ** self._number_of_qubits, 2 ** self._number_of_qubits)
        if shape != required_shape:
            raise BasicAerError(f'initial unitary is incorrect shape: {shape} != 2 ** {required_shape}')

    def _set_options(self, qobj_config=None, backend_options=None):
        if False:
            return 10
        'Set the backend options for all experiments in a qobj'
        self._initial_unitary = self.options.get('initial_unitary')
        self._chop_threshold = self.options.get('chop_threshold')
        if 'backend_options' in backend_options:
            backend_options = backend_options['backend_options']
        if 'initial_unitary' in backend_options and backend_options['initial_unitary'] is not None:
            self._initial_unitary = np.array(backend_options['initial_unitary'], dtype=complex)
        elif hasattr(qobj_config, 'initial_unitary'):
            self._initial_unitary = np.array(qobj_config.initial_unitary, dtype=complex)
        if self._initial_unitary is not None:
            shape = np.shape(self._initial_unitary)
            if len(shape) != 2 or shape[0] != shape[1]:
                raise BasicAerError('initial unitary is not a square matrix')
            iden = np.eye(len(self._initial_unitary))
            u_dagger_u = np.dot(self._initial_unitary.T.conj(), self._initial_unitary)
            norm = np.linalg.norm(u_dagger_u - iden)
            if round(norm, 10) != 0:
                raise BasicAerError('initial unitary is not unitary')
        if 'chop_threshold' in backend_options:
            self._chop_threshold = backend_options['chop_threshold']
        elif hasattr(qobj_config, 'chop_threshold'):
            self._chop_threshold = qobj_config.chop_threshold

    def _initialize_unitary(self):
        if False:
            for i in range(10):
                print('nop')
        'Set the initial unitary for simulation'
        self._validate_initial_unitary()
        if self._initial_unitary is None:
            self._unitary = np.eye(2 ** self._number_of_qubits, dtype=complex)
        else:
            self._unitary = self._initial_unitary.copy()
        self._unitary = np.reshape(self._unitary, self._number_of_qubits * [2, 2])

    def _get_unitary(self):
        if False:
            i = 10
            return i + 15
        'Return the current unitary'
        unitary = np.reshape(self._unitary, 2 * [2 ** self._number_of_qubits])
        if self._global_phase:
            unitary *= np.exp(1j * float(self._global_phase))
        unitary[abs(unitary) < self._chop_threshold] = 0.0
        return unitary

    def run(self, qobj, **backend_options):
        if False:
            for i in range(10):
                print('nop')
        'Run qobj asynchronously.\n\n        Args:\n            qobj (Qobj): payload of the experiment\n            backend_options (dict): backend options\n\n        Returns:\n            BasicAerJob: derived from BaseJob\n\n        Additional Information::\n\n            backend_options: Is a dict of options for the backend. It may contain\n                * "initial_unitary": matrix_like\n                * "chop_threshold": double\n\n            The "initial_unitary" option specifies a custom initial unitary\n            matrix for the simulator to be used instead of the identity\n            matrix. This size of this matrix must be correct for the number\n            of qubits inall experiments in the qobj.\n\n            The "chop_threshold" option specifies a truncation value for\n            setting small values to zero in the output unitary. The default\n            value is 1e-15.\n\n            Example::\n\n                backend_options = {\n                    "initial_unitary": np.array([[1, 0, 0, 0],\n                                                 [0, 0, 0, 1],\n                                                 [0, 0, 1, 0],\n                                                 [0, 1, 0, 0]])\n                    "chop_threshold": 1e-15\n                }\n        '
        if isinstance(qobj, (QuantumCircuit, list)):
            from qiskit.compiler import assemble
            out_options = {}
            for key in backend_options:
                if not hasattr(self.options, key):
                    warnings.warn('Option %s is not used by this backend' % key, UserWarning, stacklevel=2)
                else:
                    out_options[key] = backend_options[key]
            qobj = assemble(qobj, self, **out_options)
            qobj_options = qobj.config
        else:
            qobj_options = None
        self._set_options(qobj_config=qobj_options, backend_options=backend_options)
        job_id = str(uuid.uuid4())
        job = BasicAerJob(self, job_id, self._run_job(job_id, qobj))
        return job

    def _run_job(self, job_id, qobj):
        if False:
            print('Hello World!')
        'Run experiments in qobj.\n\n        Args:\n            job_id (str): unique id for the job.\n            qobj (Qobj): job description\n\n        Returns:\n            Result: Result object\n        '
        self._validate(qobj)
        result_list = []
        start = time.time()
        for experiment in qobj.experiments:
            result_list.append(self.run_experiment(experiment))
        end = time.time()
        result = {'backend_name': self.name(), 'backend_version': self._configuration.backend_version, 'qobj_id': qobj.qobj_id, 'job_id': job_id, 'results': result_list, 'status': 'COMPLETED', 'success': True, 'time_taken': end - start, 'header': qobj.header.to_dict()}
        return Result.from_dict(result)

    def run_experiment(self, experiment):
        if False:
            print('Hello World!')
        'Run an experiment (circuit) and return a single experiment result.\n\n        Args:\n            experiment (QasmQobjExperiment): experiment from qobj experiments list\n\n        Returns:\n            dict: A result dictionary which looks something like::\n\n                {\n                "name": name of this experiment (obtained from qobj.experiment header)\n                "seed": random seed used for simulation\n                "shots": number of shots used in the simulation\n                "data":\n                    {\n                    "unitary": [[[0.0, 0.0], [1.0, 0.0]],\n                                [[1.0, 0.0], [0.0, 0.0]]]\n                    },\n                "status": status string for the simulation\n                "success": boolean\n                "time taken": simulation time of this single experiment\n                }\n\n        Raises:\n            BasicAerError: if the number of qubits in the circuit is greater than 24.  Note that the\n                practical qubit limit is much lower than 24.\n        '
        start = time.time()
        self._number_of_qubits = experiment.header.n_qubits
        self._global_phase = experiment.header.global_phase
        self._validate_initial_unitary()
        self._initialize_unitary()
        for operation in experiment.instructions:
            if operation.name == 'unitary':
                qubits = operation.qubits
                gate = operation.params[0]
                self._add_unitary(gate, qubits)
            elif operation.name in SINGLE_QUBIT_GATES:
                params = getattr(operation, 'params', None)
                qubit = operation.qubits[0]
                gate = single_gate_matrix(operation.name, params)
                self._add_unitary(gate, [qubit])
            elif operation.name in ('id', 'u0'):
                pass
            elif operation.name in ('CX', 'cx'):
                qubit0 = operation.qubits[0]
                qubit1 = operation.qubits[1]
                gate = cx_gate_matrix()
                self._add_unitary(gate, [qubit0, qubit1])
            elif operation.name == 'barrier':
                pass
            else:
                backend = self.name()
                err_msg = '{0} encountered unrecognized operation "{1}"'
                raise BasicAerError(err_msg.format(backend, operation.name))
        data = {'unitary': self._get_unitary()}
        end = time.time()
        return {'name': experiment.header.name, 'shots': 1, 'data': data, 'status': 'DONE', 'success': True, 'time_taken': end - start, 'header': experiment.header.to_dict()}

    def _validate(self, qobj):
        if False:
            i = 10
            return i + 15
        'Semantic validations of the qobj which cannot be done via schemas.\n        Some of these may later move to backend schemas.\n        1. No shots\n        2. No measurements in the middle\n        '
        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise BasicAerError(f'Number of qubits {n_qubits} is greater than maximum ({max_qubits}) for "{self.name()}".')
        if hasattr(qobj.config, 'shots') and qobj.config.shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.', self.name())
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            name = experiment.header.name
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info('"%s" only supports 1 shot. Setting shots=1 for circuit "%s".', self.name(), name)
                experiment.config.shots = 1
            for operation in experiment.instructions:
                if operation.name in ['measure', 'reset']:
                    raise BasicAerError(f'Unsupported "{self.name()}" instruction "{operation.name}" in circuit "{name}".')