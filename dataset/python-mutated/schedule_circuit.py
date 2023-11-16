"""QuantumCircuit to Pulse scheduler."""
from typing import Optional, Union
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse.schedule import Schedule
from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.methods import as_soon_as_possible, as_late_as_possible
from qiskit.providers import BackendV1, BackendV2

def schedule_circuit(circuit: QuantumCircuit, schedule_config: ScheduleConfig, method: Optional[str]=None, backend: Optional[Union[BackendV1, BackendV2]]=None) -> Schedule:
    if False:
        print('Hello World!')
    "\n    Basic scheduling pass from a circuit to a pulse Schedule, using the backend. If no method is\n    specified, then a basic, as late as possible scheduling pass is performed, i.e. pulses are\n    scheduled to occur as late as possible.\n\n    Supported methods:\n\n        * ``'as_soon_as_possible'``: Schedule pulses greedily, as early as possible on a\n          qubit resource. (alias: ``'asap'``)\n        * ``'as_late_as_possible'``: Schedule pulses late-- keep qubits in the ground state when\n          possible. (alias: ``'alap'``)\n\n    Args:\n        circuit: The quantum circuit to translate.\n        schedule_config: Backend specific parameters used for building the Schedule.\n        method: The scheduling pass method to use.\n        backend: A backend used to build the Schedule, the backend could be BackendV1\n                 or BackendV2.\n\n    Returns:\n        Schedule corresponding to the input circuit.\n\n    Raises:\n        QiskitError: If method isn't recognized.\n    "
    methods = {'as_soon_as_possible': as_soon_as_possible, 'asap': as_soon_as_possible, 'as_late_as_possible': as_late_as_possible, 'alap': as_late_as_possible}
    if method is None:
        method = 'as_late_as_possible'
    try:
        return methods[method](circuit, schedule_config, backend)
    except KeyError as ex:
        raise QiskitError(f"Scheduling method {method} isn't recognized.") from ex