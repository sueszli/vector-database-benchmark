"""run circuits functions"""
from typing import Optional, Dict, Callable, List, Union, Tuple
import sys
import logging
import time
import copy
import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend, JobStatus, JobError, Job
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.result import Result
from qiskit.utils.deprecation import deprecate_func
from ..exceptions import QiskitError, MissingOptionalLibraryError
from .backend_utils import is_aer_provider, is_basicaer_provider, is_simulator_backend, is_local_backend, is_ibmq_provider, _get_backend_interface_version
MAX_CIRCUITS_PER_JOB = os.environ.get('QISKIT_AQUA_MAX_CIRCUITS_PER_JOB', None)
MAX_GATES_PER_JOB = os.environ.get('QISKIT_AQUA_MAX_GATES_PER_JOB', None)
logger = logging.getLogger(__name__)

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def find_regs_by_name(circuit: QuantumCircuit, name: str, qreg: bool=True) -> Optional[Union[QuantumRegister, ClassicalRegister]]:
    if False:
        for i in range(10):
            print('nop')
    'Deprecated: Find the registers in the circuits.\n\n    Args:\n        circuit: the quantum circuit.\n        name: name of register\n        qreg: quantum or classical register\n\n    Returns:\n        if not found, return None.\n\n    '
    found_reg = None
    regs = circuit.qregs if qreg else circuit.cregs
    for reg in regs:
        if reg.name == name:
            found_reg = reg
            break
    return found_reg

def _combine_result_objects(results: List[Result]) -> Result:
    if False:
        return 10
    'Temporary helper function.\n\n    TODO: This function would be removed after Terra supports job with infinite circuits.\n    '
    if len(results) == 1:
        return results[0]
    new_result = copy.deepcopy(results[0])
    for idx in range(1, len(results)):
        new_result.results.extend(results[idx].results)
    return new_result

def _safe_get_job_status(job: Job, job_id: str, max_job_retries: int, wait: float) -> JobStatus:
    if False:
        i = 10
        return i + 15
    for _ in range(max_job_retries):
        try:
            job_status = job.status()
            break
        except JobError as ex:
            logger.warning("FAILURE: job id: %s, status: 'FAIL_TO_GET_STATUS' Terra job error: %s", job_id, ex)
            time.sleep(wait)
        except Exception as ex:
            raise QiskitError(f"job id: {job_id}, status: 'FAIL_TO_GET_STATUS' Unknown error: ({ex})") from ex
    else:
        raise QiskitError(f'Max retry limit reached. Failed to get status for job with id {job_id}')
    return job_status

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def run_circuits(circuits: Union[QuantumCircuit, List[QuantumCircuit]], backend: Backend, qjob_config: Dict, backend_options: Optional[Dict]=None, noise_config: Optional[Dict]=None, run_config: Optional[Dict]=None, job_callback: Optional[Callable]=None, max_job_retries: int=50) -> Result:
    if False:
        return 10
    '\n    Deprecated: An execution wrapper with Qiskit-Terra, with job auto recover capability.\n\n    The auto-recovery feature is only applied for non-simulator backend.\n    This wrapper will try to get the result no matter how long it takes.\n\n    Args:\n        circuits: circuits to execute\n        backend: backend instance\n        qjob_config: configuration for quantum job object\n        backend_options: backend options\n        noise_config: configuration for noise model\n        run_config: configuration for run\n        job_callback: callback used in querying info of the submitted job, and providing the\n            following arguments: job_id, job_status, queue_position, job.\n        max_job_retries(int): positive non-zero number of trials for the job set (-1 for infinite\n            trials) (default: 50)\n\n    Returns:\n        Result object\n\n    Raises:\n        QiskitError: Any error except for JobError raised by Qiskit Terra\n    '
    backend_interface_version = _get_backend_interface_version(backend)
    backend_options = backend_options or {}
    noise_config = noise_config or {}
    run_config = run_config or {}
    if backend_interface_version <= 1:
        with_autorecover = not is_simulator_backend(backend)
    else:
        with_autorecover = False
    if MAX_CIRCUITS_PER_JOB is not None:
        max_circuits_per_job = int(MAX_CIRCUITS_PER_JOB)
    elif backend_interface_version <= 1:
        if is_local_backend(backend):
            max_circuits_per_job = sys.maxsize
        else:
            max_circuits_per_job = backend.configuration().max_experiments
    elif backend.max_circuits is not None:
        max_circuits_per_job = backend.max_circuits
    else:
        max_circuits_per_job = sys.maxsize
    if len(circuits) > max_circuits_per_job:
        jobs = []
        job_ids = []
        split_circuits = []
        count = 0
        while count < len(circuits):
            some_circuits = circuits[count:count + max_circuits_per_job]
            split_circuits.append(some_circuits)
            (job, job_id) = _safe_submit_circuits(some_circuits, backend, qjob_config=qjob_config, backend_options=backend_options, noise_config=noise_config, run_config=run_config, max_job_retries=max_job_retries)
            jobs.append(job)
            job_ids.append(job_id)
            count += max_circuits_per_job
    else:
        (job, job_id) = _safe_submit_circuits(circuits, backend, qjob_config=qjob_config, backend_options=backend_options, noise_config=noise_config, run_config=run_config, max_job_retries=max_job_retries)
        jobs = [job]
        job_ids = [job_id]
        split_circuits = [circuits]
    results = []
    if with_autorecover:
        logger.info('Backend status: %s', backend.status())
        logger.info('There are %s jobs are submitted.', len(jobs))
        logger.info('All job ids:\n%s', job_ids)
        for (idx, _) in enumerate(jobs):
            result = None
            logger.info('Backend status: %s', backend.status())
            logger.info('There is one jobs are submitted: id: %s', job_id)
            job = jobs[idx]
            job_id = job_ids[idx]
            for _ in range(max_job_retries):
                logger.info('Running job id: %s', job_id)
                while True:
                    job_status = _safe_get_job_status(job, job_id, max_job_retries, qjob_config['wait'])
                    queue_position = 0
                    if job_status in JOB_FINAL_STATES:
                        if job_callback is not None:
                            job_callback(job_id, job_status, queue_position, job)
                        break
                    if job_status == JobStatus.QUEUED and hasattr(job, 'queue_position'):
                        queue_position = job.queue_position()
                        logger.info('Job id: %s is queued at position %s', job_id, queue_position)
                    else:
                        logger.info('Job id: %s, status: %s', job_id, job_status)
                    if job_callback is not None:
                        job_callback(job_id, job_status, queue_position, job)
                    time.sleep(qjob_config['wait'])
                if job_status == JobStatus.DONE:
                    for _ in range(max_job_retries):
                        result = job.result()
                        if result.success:
                            results.append(result)
                            logger.info('COMPLETED the %s-th job, job id: %s', idx, job_id)
                            break
                        logger.warning('FAILURE: Job id: %s', job_id)
                        logger.warning('Job (%s) is completed anyway, retrieve result from backend again.', job_id)
                        job = backend.retrieve_job(job_id)
                    else:
                        raise QiskitError(f'Max retry limit reached. Failed to get result for job id {job_id}')
                    break
                if job_status == JobStatus.CANCELLED:
                    logger.warning('FAILURE: Job id: %s is cancelled. Re-submit the circuits.', job_id)
                elif job_status == JobStatus.ERROR:
                    logger.warning('FAILURE: Job id: %s encounters the error. Error is : %s. Re-submit the circuits.', job_id, job.error_message())
                else:
                    logging.warning('FAILURE: Job id: %s. Unknown status: %s. Re-submit the circuits.', job_id, job_status)
                (job, job_id) = _safe_submit_circuits(split_circuits[idx], backend, qjob_config=qjob_config, backend_options=backend_options, noise_config=noise_config, run_config=run_config, max_job_retries=max_job_retries)
            else:
                raise QiskitError(f'Max retry limit reached. Failed to get result for job with id {job_id} ')
    else:
        results = []
        for job in jobs:
            results.append(job.result())
    result = _combine_result_objects(results) if results else None
    if not result.success:
        msg = result.status
        if result.status == 'PARTIAL COMPLETED':
            for res in result.results:
                if not res.success:
                    msg += ', ' + res.status
                    break
        raise QiskitError(f'Circuit execution failed: {msg}')
    if not hasattr(result, 'time_taken'):
        setattr(result, 'time_taken', 0.0)
    return result

def _safe_submit_circuits(circuits: Union[QuantumCircuit, List[QuantumCircuit]], backend: Backend, qjob_config: Dict, backend_options: Dict, noise_config: Dict, run_config: Dict, max_job_retries: int) -> Tuple[Job, str]:
    if False:
        i = 10
        return i + 15
    for _ in range(max_job_retries):
        try:
            job = _run_circuits_on_backend(backend, circuits, backend_options=backend_options, noise_config=noise_config, run_config=run_config)
            job_id = job.job_id()
            break
        except QiskitError as ex:
            failure_warn = True
            if is_ibmq_provider(backend):
                try:
                    from qiskit.providers.ibmq import IBMQBackendJobLimitError
                except ImportError as ex1:
                    raise MissingOptionalLibraryError(libname='qiskit-ibmq-provider', name='_safe_submit_circuits', pip_install='pip install qiskit-ibmq-provider') from ex1
                if isinstance(ex, IBMQBackendJobLimitError):
                    oldest_running = backend.jobs(limit=1, descending=False, status=['QUEUED', 'VALIDATING', 'RUNNING'])
                    if oldest_running:
                        oldest_running = oldest_running[0]
                        logger.warning('Job limit reached, waiting for job %s to finish before submitting the next one.', oldest_running.job_id())
                        failure_warn = False
                        try:
                            oldest_running.wait_for_final_state(timeout=qjob_config['timeout'], wait=qjob_config['wait'])
                        except Exception:
                            pass
            if failure_warn:
                logger.warning('FAILURE: Can not get job id, Resubmit the qobj to get job id. Terra job error: %s ', ex)
        except Exception as ex:
            logger.warning('FAILURE: Can not get job id, Resubmit the qobj to get job id. Error: %s ', ex)
    else:
        raise QiskitError('Max retry limit reached. Failed to submit the qobj correctly')
    return (job, job_id)

def _run_circuits_on_backend(backend: Backend, circuits: Union[QuantumCircuit, List[QuantumCircuit]], backend_options: Dict, noise_config: Dict, run_config: Dict) -> Job:
    if False:
        for i in range(10):
            print('nop')
    'Run on backend.'
    run_kwargs = {}
    if is_aer_provider(backend) or is_basicaer_provider(backend):
        for (key, value) in backend_options.items():
            if key == 'backend_options':
                for (k, v) in value.items():
                    run_kwargs[k] = v
            else:
                run_kwargs[key] = value
    else:
        run_kwargs.update(backend_options)
    run_kwargs.update(noise_config)
    run_kwargs.update(run_config)
    if is_basicaer_provider(backend):
        for key in list(run_kwargs.keys()):
            if not hasattr(backend.options, key):
                del run_kwargs[key]
    return backend.run(circuits, **run_kwargs)