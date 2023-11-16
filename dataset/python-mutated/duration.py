"""
Utilities for handling duration of a circuit instruction.
"""
import warnings
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.units import apply_prefix

def duration_in_dt(duration_in_sec: float, dt_in_sec: float) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Return duration in dt.\n\n    Args:\n        duration_in_sec: duration [s] to be converted.\n        dt_in_sec: duration of dt in seconds used for conversion.\n\n    Returns:\n        Duration in dt.\n    '
    res = round(duration_in_sec / dt_in_sec)
    rounding_error = abs(duration_in_sec - res * dt_in_sec)
    if rounding_error > 1e-15:
        warnings.warn('Duration is rounded to %d [dt] = %e [s] from %e [s]' % (res, res * dt_in_sec, duration_in_sec), UserWarning)
    return res

def convert_durations_to_dt(qc: QuantumCircuit, dt_in_sec: float, inplace=True):
    if False:
        while True:
            i = 10
    'Convert all the durations in SI (seconds) into those in dt.\n\n    Returns a new circuit if `inplace=False`.\n\n    Parameters:\n        qc (QuantumCircuit): Duration of dt in seconds used for conversion.\n        dt_in_sec (float): Duration of dt in seconds used for conversion.\n        inplace (bool): All durations are converted inplace or return new circuit.\n\n    Returns:\n        QuantumCircuit: Converted circuit if `inplace = False`, otherwise None.\n\n    Raises:\n        CircuitError: if fail to convert durations.\n    '
    if inplace:
        circ = qc
    else:
        circ = qc.copy()
    for instruction in circ.data:
        operation = instruction.operation
        if operation.unit == 'dt' or operation.duration is None:
            continue
        if not operation.unit.endswith('s'):
            raise CircuitError(f"Invalid time unit: '{operation.unit}'")
        duration = operation.duration
        if operation.unit != 's':
            duration = apply_prefix(duration, operation.unit)
        operation.duration = duration_in_dt(duration, dt_in_sec)
        operation.unit = 'dt'
    if circ.duration is not None:
        circ.duration = duration_in_dt(circ.duration, dt_in_sec)
        circ.unit = 'dt'
    if not inplace:
        return circ
    else:
        return None