"""Services for managing Apache Beam jobs."""
from __future__ import annotations
from core.constants import constants
from core.domain import beam_job_domain
from core.jobs import base_jobs
from core.jobs import jobs_manager
from core.jobs import registry as jobs_registry
from core.platform import models
from typing import List, Optional, Type
MYPY = False
if MYPY:
    from mypy_imports import beam_job_models
    from mypy_imports import datastore_services
(beam_job_models,) = models.Registry.import_models([models.Names.BEAM_JOB])
datastore_services = models.Registry.import_datastore_services()

def run_beam_job(job_name: Optional[str]=None, job_class: Optional[Type[base_jobs.JobBase]]=None) -> beam_job_domain.BeamJobRun:
    if False:
        print('Hello World!')
    "Starts a new Apache Beam job and returns metadata about its execution.\n\n    Args:\n        job_name: str. The name of the job to run. If not provided, then\n            job_class must not be None.\n        job_class: type(JobBase). A subclass of JobBase to begin running. This\n            value takes precedence over job_name.\n\n    Returns:\n        BeamJobRun. Metadata about the run's execution.\n\n    Raises:\n        ValueError. Both name and class of the job are not specified.\n    "
    if job_class is None:
        if job_name:
            job_class = jobs_registry.get_job_class_by_name(job_name)
        else:
            raise ValueError('Must specify the job class or name to run')
    run_synchronously = constants.EMULATOR_MODE
    run_model = jobs_manager.run_job(job_class, run_synchronously)
    return get_beam_job_run_from_model(run_model)

def cancel_beam_job(job_id: str) -> beam_job_domain.BeamJobRun:
    if False:
        print('Hello World!')
    "Cancels an existing Apache Beam job and returns its updated metadata.\n\n    Args:\n        job_id: str. The Oppia-provided ID of the job.\n\n    Returns:\n        BeamJobRun. Metadata about the updated run's execution.\n\n    Raises:\n        ValueError. Job does not exist.\n    "
    beam_job_run_model = beam_job_models.BeamJobRunModel.get(job_id, strict=False)
    if beam_job_run_model is None:
        raise ValueError('No such job with id="%s"' % job_id)
    if beam_job_run_model.dataflow_job_id is None:
        raise ValueError('Job with id="%s" cannot be cancelled' % job_id)
    jobs_manager.cancel_job(beam_job_run_model)
    return get_beam_job_run_from_model(beam_job_run_model)

def get_beam_jobs() -> List[beam_job_domain.BeamJob]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the list of all registered Apache Beam jobs.\n\n    Returns:\n        list(BeamJob). The list of registered Apache Beam jobs.\n    '
    return [beam_job_domain.BeamJob(j) for j in jobs_registry.get_all_jobs()]

def is_state_terminal(job_state: str) -> bool:
    if False:
        while True:
            i = 10
    'Returns whether the job state is a terminal state, meaning\n    that the job is longer executing.\n\n    Returns:\n        bool. Whether the state is a terminal state.\n    '
    return job_state in (beam_job_models.BeamJobState.CANCELLED.value, beam_job_models.BeamJobState.DRAINED.value, beam_job_models.BeamJobState.UPDATED.value, beam_job_models.BeamJobState.DONE.value, beam_job_models.BeamJobState.FAILED.value)

def get_beam_job_runs(refresh: bool=True) -> List[beam_job_domain.BeamJobRun]:
    if False:
        return 10
    "Returns all of the Apache Beam job runs recorded in the datastore.\n\n    Args:\n        refresh: bool. Whether to refresh the jobs' state before returning them.\n\n    Returns:\n        list(BeamJobRun). A list of every job run recorded in the datastore.\n    "
    beam_job_run_models = list(beam_job_models.BeamJobRunModel.query())
    beam_job_runs = [get_beam_job_run_from_model(m) for m in beam_job_run_models]
    if refresh:
        updated_beam_job_run_models = []
        for (i, beam_job_run_model) in enumerate(beam_job_run_models):
            if is_state_terminal(beam_job_runs[i].job_state):
                continue
            jobs_manager.refresh_state_of_beam_job_run_model(beam_job_run_model)
            beam_job_run_model.update_timestamps(update_last_updated_time=False)
            updated_beam_job_run_models.append(beam_job_run_model)
            beam_job_runs[i] = get_beam_job_run_from_model(beam_job_run_model)
        if updated_beam_job_run_models:
            datastore_services.put_multi(updated_beam_job_run_models)
    return beam_job_runs

def get_beam_job_run_result(job_id: str) -> beam_job_domain.AggregateBeamJobRunResult:
    if False:
        print('Hello World!')
    'Returns the result of the given Apache Beam job run.\n\n    Args:\n        job_id: str. The ID of the job run to fetch.\n\n    Returns:\n        AggregateBeamJobRunResult. The result of the given Apache Beam job run.\n    '
    beam_job_run_result_models = beam_job_models.BeamJobRunResultModel.query(beam_job_models.BeamJobRunResultModel.job_id == job_id).iter()
    (stdouts, stderrs) = ([], [])
    for beam_job_run_result_model in beam_job_run_result_models:
        if beam_job_run_result_model.stdout:
            stdouts.append(beam_job_run_result_model.stdout)
        if beam_job_run_result_model.stderr:
            stderrs.append(beam_job_run_result_model.stderr)
    return beam_job_domain.AggregateBeamJobRunResult(stdout='\n'.join(stdouts), stderr='\n'.join(stderrs))

def create_beam_job_run_model(job_name: str, dataflow_job_id: Optional[str]=None) -> beam_job_models.BeamJobRunModel:
    if False:
        for i in range(10):
            print('nop')
    "Creates a new BeamJobRunModel without putting it into storage.\n\n    Args:\n        job_name: str. The name of the job class that implements the job's\n            logic.\n        dataflow_job_id: str|None. The ID of the dataflow job this model\n            corresponds to. If the job is run synchronously, then this value\n            should be None.\n\n    Returns:\n        BeamJobRunModel. The model.\n    "
    model_id = beam_job_models.BeamJobRunModel.get_new_id()
    model = beam_job_models.BeamJobRunModel(id=model_id, job_name=job_name, dataflow_job_id=dataflow_job_id, latest_job_state=beam_job_models.BeamJobState.PENDING.value)
    model.update_timestamps()
    return model

def create_beam_job_run_result_model(job_id: str, stdout: str, stderr: str) -> beam_job_models.BeamJobRunResultModel:
    if False:
        for i in range(10):
            print('nop')
    'Creates a new BeamJobRunResultModel without putting it into storage.\n\n    Args:\n        job_id: str. The ID of the job run to fetch.\n        stdout: str. The standard output from a job run.\n        stderr: str. The error output from a job run.\n\n    Returns:\n        BeamJobRunResultModel. The model.\n    '
    model_id = beam_job_models.BeamJobRunResultModel.get_new_id()
    model = beam_job_models.BeamJobRunResultModel(id=model_id, job_id=job_id, stdout=stdout, stderr=stderr)
    model.update_timestamps()
    return model

def get_beam_job_run_from_model(beam_job_run_model: beam_job_models.BeamJobRunModel) -> beam_job_domain.BeamJobRun:
    if False:
        while True:
            i = 10
    'Returns a domain object corresponding to the given BeamJobRunModel.\n\n    Args:\n        beam_job_run_model: BeamJobRunModel. The model.\n\n    Returns:\n        BeamJobRun. The corresponding domain object.\n    '
    return beam_job_domain.BeamJobRun(beam_job_run_model.id, beam_job_run_model.job_name, beam_job_run_model.latest_job_state, beam_job_run_model.created_on, beam_job_run_model.last_updated, beam_job_run_model.dataflow_job_id is None)