"""A module for monitoring various qiskit functionality"""
import sys
import time

def _text_checker(job, interval, _interval_set=False, quiet=False, output=sys.stdout, line_discipline='\r'):
    if False:
        for i in range(10):
            print('nop')
    'A text-based job status checker\n\n    Args:\n        job (BaseJob): The job to check.\n        interval (int): The interval at which to check.\n        _interval_set (bool): Was interval time set by user?\n        quiet (bool): If True, do not print status messages.\n        output (file): The file like object to write status messages to.\n        By default this is sys.stdout.\n        line_discipline (string): character emitted at start of a line of job monitor output,\n        This defaults to \\r.\n\n    '
    status = job.status()
    msg = status.value
    prev_msg = msg
    msg_len = len(msg)
    if not quiet:
        print('{}{}: {}'.format(line_discipline, 'Job Status', msg), end='', file=output)
    while status.name not in ['DONE', 'CANCELLED', 'ERROR']:
        time.sleep(interval)
        status = job.status()
        msg = status.value
        if status.name == 'QUEUED':
            msg += ' (%s)' % job.queue_position()
            if job.queue_position() is None:
                interval = 2
            elif not _interval_set:
                interval = max(job.queue_position(), 2)
        elif not _interval_set:
            interval = 2
        if len(msg) < msg_len:
            msg += ' ' * (msg_len - len(msg))
        elif len(msg) > msg_len:
            msg_len = len(msg)
        if msg != prev_msg and (not quiet):
            print('{}{}: {}'.format(line_discipline, 'Job Status', msg), end='', file=output)
            prev_msg = msg
    if not quiet:
        print('', file=output)

def job_monitor(job, interval=None, quiet=False, output=sys.stdout, line_discipline='\r'):
    if False:
        for i in range(10):
            print('nop')
    'Monitor the status of a IBMQJob instance.\n\n    Args:\n        job (BaseJob): Job to monitor.\n        interval (int): Time interval between status queries.\n        quiet (bool): If True, do not print status messages.\n        output (file): The file like object to write status messages to.\n        By default this is sys.stdout.\n        line_discipline (string): character emitted at start of a line of job monitor output,\n        This defaults to \\r.\n\n    Examples:\n\n        .. code-block:: python\n\n            from qiskit import BasicAer, transpile\n            from qiskit.circuit import QuantumCircuit\n            from qiskit.tools.monitor import job_monitor\n            sim_backend = BasicAer.get_backend("qasm_simulator")\n            qc = QuantumCircuit(2, 2)\n            qc.h(0)\n            qc.cx(0, 1)\n            qc.measure_all()\n            tqc = transpile(qc, sim_backend)\n            job_sim = sim_backend.run(tqc)\n            job_monitor(job_sim)\n    '
    if interval is None:
        _interval_set = False
        interval = 5
    else:
        _interval_set = True
    _text_checker(job, interval, _interval_set, quiet=quiet, output=output, line_discipline=line_discipline)