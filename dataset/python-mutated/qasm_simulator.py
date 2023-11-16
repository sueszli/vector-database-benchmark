"""Contains a (slow) Python simulator.

It simulates an OpenQASM 2 quantum circuit (an experiment) that has been compiled
to run on the simulator. It is exponential in the number of qubits.

The simulator is run using

.. code-block:: python

    QasmSimulatorPy().run(run_input)

Where the input is a QuantumCircuit object and the output is a BasicAerJob object, which can
later be queried for the Result object. The result will contain a 'memory' data
field, which is a result of measurements for each shot.
"""
import uuid
import time
import logging
import warnings
from math import log2
from collections import Counter
import numpy as np
from qiskit.utils.multiprocessing import local_hardware_info
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.providers.backend import BackendV1
from qiskit.providers.options import Options
from qiskit.providers.basicaer.basicaerjob import BasicAerJob
from .exceptions import BasicAerError
from .basicaertools import single_gate_matrix
from .basicaertools import SINGLE_QUBIT_GATES
from .basicaertools import cx_gate_matrix
from .basicaertools import einsum_vecmul_index
logger = logging.getLogger(__name__)

class QasmSimulatorPy(BackendV1):
    """Python implementation of an OpenQASM 2 simulator."""
    MAX_QUBITS_MEMORY = int(log2(local_hardware_info()['memory'] * 1024 ** 3 / 16))
    DEFAULT_CONFIGURATION = {'backend_name': 'qasm_simulator', 'backend_version': '2.1.0', 'n_qubits': min(24, MAX_QUBITS_MEMORY), 'url': 'https://github.com/Qiskit/qiskit-terra', 'simulator': True, 'local': True, 'conditional': True, 'open_pulse': False, 'memory': True, 'max_shots': 0, 'coupling_map': None, 'description': 'A python simulator for qasm experiments', 'basis_gates': ['h', 'u', 'p', 'u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id', 'unitary'], 'gates': [{'name': 'h', 'parameters': [], 'qasm_def': 'gate h q { U(pi/2,0,pi) q; }'}, {'name': 'p', 'parameters': ['lambda'], 'qasm_def': 'gate p(lambda) q { U(0,0,lambda) q; }'}, {'name': 'u', 'parameters': ['theta', 'phi', 'lambda'], 'qasm_def': 'gate u(theta,phi,lambda) q { U(theta,phi,lambda) q; }'}, {'name': 'u1', 'parameters': ['lambda'], 'qasm_def': 'gate u1(lambda) q { U(0,0,lambda) q; }'}, {'name': 'u2', 'parameters': ['phi', 'lambda'], 'qasm_def': 'gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }'}, {'name': 'u3', 'parameters': ['theta', 'phi', 'lambda'], 'qasm_def': 'gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }'}, {'name': 'rz', 'parameters': ['phi'], 'qasm_def': 'gate rz(phi) q { U(0,0,phi) q; }'}, {'name': 'sx', 'parameters': [], 'qasm_def': 'gate sx(phi) q { U(pi/2,7*pi/2,pi/2) q; }'}, {'name': 'x', 'parameters': [], 'qasm_def': 'gate x q { U(pi,7*pi/2,pi/2) q; }'}, {'name': 'cx', 'parameters': [], 'qasm_def': 'gate cx c,t { CX c,t; }'}, {'name': 'id', 'parameters': [], 'qasm_def': 'gate id a { U(0,0,0) a; }'}, {'name': 'unitary', 'parameters': ['matrix'], 'qasm_def': 'unitary(matrix) q1, q2,...'}]}
    DEFAULT_OPTIONS = {'initial_statevector': None, 'chop_threshold': 1e-15}
    SHOW_FINAL_STATE = False

    def __init__(self, configuration=None, provider=None, **fields):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(configuration=configuration or QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION), provider=provider, **fields)
        self._local_random = np.random.RandomState()
        self._classical_memory = 0
        self._classical_register = 0
        self._statevector = 0
        self._number_of_cmembits = 0
        self._number_of_qubits = 0
        self._shots = 0
        self._memory = False
        self._initial_statevector = self.options.get('initial_statevector')
        self._chop_threshold = self.options.get('chop_threashold')
        self._qobj_config = None
        self._sample_measure = False

    @classmethod
    def _default_options(cls):
        if False:
            return 10
        return Options(shots=1024, memory=False, initial_statevector=None, chop_threshold=1e-15, allow_sample_measuring=True, seed_simulator=None, parameter_binds=None)

    def _add_unitary(self, gate, qubits):
        if False:
            print('Hello World!')
        'Apply an N-qubit unitary matrix.\n\n        Args:\n            gate (matrix_like): an N-qubit unitary matrix\n            qubits (list): the list of N-qubits.\n        '
        num_qubits = len(qubits)
        indexes = einsum_vecmul_index(qubits, self._number_of_qubits)
        gate_tensor = np.reshape(np.array(gate, dtype=complex), num_qubits * [2, 2])
        self._statevector = np.einsum(indexes, gate_tensor, self._statevector, dtype=complex, casting='no')

    def _get_measure_outcome(self, qubit):
        if False:
            print('Hello World!')
        "Simulate the outcome of measurement of a qubit.\n\n        Args:\n            qubit (int): the qubit to measure\n\n        Return:\n            tuple: pair (outcome, probability) where outcome is '0' or '1' and\n            probability is the probability of the returned outcome.\n        "
        axis = list(range(self._number_of_qubits))
        axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis))
        random_number = self._local_random.rand()
        if random_number < probabilities[0]:
            return ('0', probabilities[0])
        return ('1', probabilities[1])

    def _add_sample_measure(self, measure_params, num_samples):
        if False:
            i = 10
            return i + 15
        'Generate memory samples from current statevector.\n\n        Args:\n            measure_params (list): List of (qubit, cmembit) values for\n                                   measure instructions to sample.\n            num_samples (int): The number of memory samples to generate.\n\n        Returns:\n            list: A list of memory values in hex format.\n        '
        measured_qubits = sorted({qubit for (qubit, cmembit) in measure_params})
        num_measured = len(measured_qubits)
        axis = list(range(self._number_of_qubits))
        for qubit in reversed(measured_qubits):
            axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.reshape(np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis)), 2 ** num_measured)
        samples = self._local_random.choice(range(2 ** num_measured), num_samples, p=probabilities)
        memory = []
        for sample in samples:
            classical_memory = self._classical_memory
            for (qubit, cmembit) in measure_params:
                pos = measured_qubits.index(qubit)
                qubit_outcome = int((sample & 1 << pos) >> pos)
                membit = 1 << cmembit
                classical_memory = classical_memory & ~membit | qubit_outcome << cmembit
            value = bin(classical_memory)[2:]
            memory.append(hex(int(value, 2)))
        return memory

    def _add_qasm_measure(self, qubit, cmembit, cregbit=None):
        if False:
            print('Hello World!')
        'Apply a measure instruction to a qubit.\n\n        Args:\n            qubit (int): qubit is the qubit measured.\n            cmembit (int): is the classical memory bit to store outcome in.\n            cregbit (int, optional): is the classical register bit to store outcome in.\n        '
        (outcome, probability) = self._get_measure_outcome(qubit)
        membit = 1 << cmembit
        self._classical_memory = self._classical_memory & ~membit | int(outcome) << cmembit
        if cregbit is not None:
            regbit = 1 << cregbit
            self._classical_register = self._classical_register & ~regbit | int(outcome) << cregbit
        if outcome == '0':
            update_diag = [[1 / np.sqrt(probability), 0], [0, 0]]
        else:
            update_diag = [[0, 0], [0, 1 / np.sqrt(probability)]]
        self._add_unitary(update_diag, [qubit])

    def _add_qasm_reset(self, qubit):
        if False:
            for i in range(10):
                print('nop')
        'Apply a reset instruction to a qubit.\n\n        Args:\n            qubit (int): the qubit being rest\n\n        This is done by doing a simulating a measurement\n        outcome and projecting onto the outcome state while\n        renormalizing.\n        '
        (outcome, probability) = self._get_measure_outcome(qubit)
        if outcome == '0':
            update = [[1 / np.sqrt(probability), 0], [0, 0]]
            self._add_unitary(update, [qubit])
        else:
            update = [[0, 1 / np.sqrt(probability)], [0, 0]]
            self._add_unitary(update, [qubit])

    def _validate_initial_statevector(self):
        if False:
            return 10
        'Validate an initial statevector'
        if self._initial_statevector is None:
            return
        length = len(self._initial_statevector)
        required_dim = 2 ** self._number_of_qubits
        if length != required_dim:
            raise BasicAerError(f'initial statevector is incorrect length: {length} != {required_dim}')

    def _set_options(self, qobj_config=None, backend_options=None):
        if False:
            return 10
        'Set the backend options for all experiments in a qobj'
        self._initial_statevector = self.options.get('initial_statevector')
        self._chop_threshold = self.options.get('chop_threshold')
        if 'backend_options' in backend_options and backend_options['backend_options']:
            backend_options = backend_options['backend_options']
        if 'initial_statevector' in backend_options and backend_options['initial_statevector'] is not None:
            self._initial_statevector = np.array(backend_options['initial_statevector'], dtype=complex)
        elif hasattr(qobj_config, 'initial_statevector'):
            self._initial_statevector = np.array(qobj_config.initial_statevector, dtype=complex)
        if self._initial_statevector is not None:
            norm = np.linalg.norm(self._initial_statevector)
            if round(norm, 12) != 1:
                raise BasicAerError(f'initial statevector is not normalized: norm {norm} != 1')
        if 'chop_threshold' in backend_options:
            self._chop_threshold = backend_options['chop_threshold']
        elif hasattr(qobj_config, 'chop_threshold'):
            self._chop_threshold = qobj_config.chop_threshold

    def _initialize_statevector(self):
        if False:
            for i in range(10):
                print('nop')
        'Set the initial statevector for simulation'
        if self._initial_statevector is None:
            self._statevector = np.zeros(2 ** self._number_of_qubits, dtype=complex)
            self._statevector[0] = 1
        else:
            self._statevector = self._initial_statevector.copy()
        self._statevector = np.reshape(self._statevector, self._number_of_qubits * [2])

    def _get_statevector(self):
        if False:
            i = 10
            return i + 15
        'Return the current statevector'
        vec = np.reshape(self._statevector, 2 ** self._number_of_qubits)
        vec[abs(vec) < self._chop_threshold] = 0.0
        return vec

    def _validate_measure_sampling(self, experiment):
        if False:
            return 10
        'Determine if measure sampling is allowed for an experiment\n\n        Args:\n            experiment (QasmQobjExperiment): a qobj experiment.\n        '
        if self._shots <= 1:
            self._sample_measure = False
            return
        if hasattr(experiment.config, 'allows_measure_sampling'):
            self._sample_measure = experiment.config.allows_measure_sampling
        else:
            measure_flag = False
            for instruction in experiment.instructions:
                if instruction.name == 'reset':
                    self._sample_measure = False
                    return
                if measure_flag:
                    if instruction.name not in ['measure', 'barrier', 'id', 'u0']:
                        self._sample_measure = False
                        return
                elif instruction.name == 'measure':
                    measure_flag = True
            self._sample_measure = True

    def run(self, run_input, **backend_options):
        if False:
            while True:
                i = 10
        'Run on the backend.\n\n        Args:\n            run_input (QuantumCircuit or list): payload of the experiment\n            backend_options (dict): backend options\n\n        Returns:\n            BasicAerJob: derived from BaseJob\n\n        Additional Information:\n            backend_options: Is a dict of options for the backend. It may contain\n                * "initial_statevector": vector_like\n\n            The "initial_statevector" option specifies a custom initial\n            initial statevector for the simulator to be used instead of the all\n            zero state. This size of this vector must be correct for the number\n            of qubits in ``run_input`` parameter.\n\n            Example::\n\n                backend_options = {\n                    "initial_statevector": np.array([1, 0, 0, 1j]) / np.sqrt(2),\n                }\n        '
        from qiskit.compiler import assemble
        out_options = {}
        for key in backend_options:
            if not hasattr(self.options, key):
                warnings.warn('Option %s is not used by this backend' % key, UserWarning, stacklevel=2)
            else:
                out_options[key] = backend_options[key]
        qobj = assemble(run_input, self, **out_options)
        qobj_options = qobj.config
        self._set_options(qobj_config=qobj_options, backend_options=backend_options)
        job_id = str(uuid.uuid4())
        job = BasicAerJob(self, job_id, self._run_job(job_id, qobj))
        return job

    def _run_job(self, job_id, qobj):
        if False:
            for i in range(10):
                print('nop')
        'Run experiments in qobj\n\n        Args:\n            job_id (str): unique id for the job.\n            qobj (Qobj): job description\n\n        Returns:\n            Result: Result object\n        '
        self._validate(qobj)
        result_list = []
        self._shots = qobj.config.shots
        self._memory = getattr(qobj.config, 'memory', False)
        self._qobj_config = qobj.config
        start = time.time()
        for experiment in qobj.experiments:
            result_list.append(self.run_experiment(experiment))
        end = time.time()
        result = {'backend_name': self.name(), 'backend_version': self._configuration.backend_version, 'qobj_id': qobj.qobj_id, 'job_id': job_id, 'results': result_list, 'status': 'COMPLETED', 'success': True, 'time_taken': end - start, 'header': qobj.header.to_dict()}
        return Result.from_dict(result)

    def run_experiment(self, experiment):
        if False:
            i = 10
            return i + 15
        'Run an experiment (circuit) and return a single experiment result.\n\n        Args:\n            experiment (QasmQobjExperiment): experiment from qobj experiments list\n\n        Returns:\n             dict: A result dictionary which looks something like::\n\n                {\n                "name": name of this experiment (obtained from qobj.experiment header)\n                "seed": random seed used for simulation\n                "shots": number of shots used in the simulation\n                "data":\n                    {\n                    "counts": {\'0x9: 5, ...},\n                    "memory": [\'0x9\', \'0xF\', \'0x1D\', ..., \'0x9\']\n                    },\n                "status": status string for the simulation\n                "success": boolean\n                "time_taken": simulation time of this single experiment\n                }\n        Raises:\n            BasicAerError: if an error occurred.\n        '
        start = time.time()
        self._number_of_qubits = experiment.config.n_qubits
        self._number_of_cmembits = experiment.config.memory_slots
        self._statevector = 0
        self._classical_memory = 0
        self._classical_register = 0
        self._sample_measure = False
        global_phase = experiment.header.global_phase
        self._validate_initial_statevector()
        if hasattr(experiment.config, 'seed_simulator'):
            seed_simulator = experiment.config.seed_simulator
        elif hasattr(self._qobj_config, 'seed_simulator'):
            seed_simulator = self._qobj_config.seed_simulator
        else:
            seed_simulator = np.random.randint(2147483647, dtype='int32')
        self._local_random.seed(seed=seed_simulator)
        self._validate_measure_sampling(experiment)
        memory = []
        if self._sample_measure:
            shots = 1
            measure_sample_ops = []
        else:
            shots = self._shots
        for _ in range(shots):
            self._initialize_statevector()
            self._statevector *= np.exp(1j * global_phase)
            self._classical_memory = 0
            self._classical_register = 0
            for operation in experiment.instructions:
                conditional = getattr(operation, 'conditional', None)
                if isinstance(conditional, int):
                    conditional_bit_set = self._classical_register >> conditional & 1
                    if not conditional_bit_set:
                        continue
                elif conditional is not None:
                    mask = int(operation.conditional.mask, 16)
                    if mask > 0:
                        value = self._classical_memory & mask
                        while mask & 1 == 0:
                            mask >>= 1
                            value >>= 1
                        if value != int(operation.conditional.val, 16):
                            continue
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
                elif operation.name == 'reset':
                    qubit = operation.qubits[0]
                    self._add_qasm_reset(qubit)
                elif operation.name == 'barrier':
                    pass
                elif operation.name == 'measure':
                    qubit = operation.qubits[0]
                    cmembit = operation.memory[0]
                    cregbit = operation.register[0] if hasattr(operation, 'register') else None
                    if self._sample_measure:
                        measure_sample_ops.append((qubit, cmembit))
                    else:
                        self._add_qasm_measure(qubit, cmembit, cregbit)
                elif operation.name == 'bfunc':
                    mask = int(operation.mask, 16)
                    relation = operation.relation
                    val = int(operation.val, 16)
                    cregbit = operation.register
                    cmembit = operation.memory if hasattr(operation, 'memory') else None
                    compared = (self._classical_register & mask) - val
                    if relation == '==':
                        outcome = compared == 0
                    elif relation == '!=':
                        outcome = compared != 0
                    elif relation == '<':
                        outcome = compared < 0
                    elif relation == '<=':
                        outcome = compared <= 0
                    elif relation == '>':
                        outcome = compared > 0
                    elif relation == '>=':
                        outcome = compared >= 0
                    else:
                        raise BasicAerError('Invalid boolean function relation.')
                    regbit = 1 << cregbit
                    self._classical_register = self._classical_register & ~regbit | int(outcome) << cregbit
                    if cmembit is not None:
                        membit = 1 << cmembit
                        self._classical_memory = self._classical_memory & ~membit | int(outcome) << cmembit
                else:
                    backend = self.name()
                    err_msg = '{0} encountered unrecognized operation "{1}"'
                    raise BasicAerError(err_msg.format(backend, operation.name))
            if self._number_of_cmembits > 0:
                if self._sample_measure:
                    memory = self._add_sample_measure(measure_sample_ops, self._shots)
                else:
                    outcome = bin(self._classical_memory)[2:]
                    memory.append(hex(int(outcome, 2)))
        data = {'counts': dict(Counter(memory))}
        if self._memory:
            data['memory'] = memory
        if self.SHOW_FINAL_STATE:
            data['statevector'] = self._get_statevector()
            if not data['counts']:
                data.pop('counts')
            if 'memory' in data and (not data['memory']):
                data.pop('memory')
        end = time.time()
        return {'name': experiment.header.name, 'seed_simulator': seed_simulator, 'shots': self._shots, 'data': data, 'status': 'DONE', 'success': True, 'time_taken': end - start, 'header': experiment.header.to_dict()}

    def _validate(self, qobj):
        if False:
            i = 10
            return i + 15
        'Semantic validations of the qobj which cannot be done via schemas.'
        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise BasicAerError(f'Number of qubits {n_qubits} is greater than maximum ({max_qubits}) for "{self.name()}".')
        for experiment in qobj.experiments:
            name = experiment.header.name
            if experiment.config.memory_slots == 0:
                logger.warning('No classical registers in circuit "%s", counts will be empty.', name)
            elif 'measure' not in [op.name for op in experiment.instructions]:
                logger.warning('No measurements in circuit "%s", classical register will remain all zeros.', name)