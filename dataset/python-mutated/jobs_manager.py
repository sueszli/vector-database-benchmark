"""Services for executing Apache Beam jobs."""
from __future__ import annotations
import contextlib
import logging
import pprint
from core import feconf
from core.domain import beam_job_services
from core.domain import caching_services
from core.jobs import base_jobs
from core.jobs import job_options
from core.jobs.io import cache_io
from core.jobs.io import job_io
from core.platform import models
from core.storage.beam_job import gae_models as beam_job_models
import apache_beam as beam
from apache_beam import runners
from google.cloud import dataflow
from typing import Iterator, Optional, Type
MYPY = False
if MYPY:
    from mypy_imports import datastore_services
datastore_services = models.Registry.import_datastore_services()
_GCLOUD_DATAFLOW_JOB_STATE_TO_OPPIA_BEAM_JOB_STATE = {dataflow.JobState.JOB_STATE_STOPPED: beam_job_models.BeamJobState.STOPPED, dataflow.JobState.JOB_STATE_RUNNING: beam_job_models.BeamJobState.RUNNING, dataflow.JobState.JOB_STATE_DONE: beam_job_models.BeamJobState.DONE, dataflow.JobState.JOB_STATE_FAILED: beam_job_models.BeamJobState.FAILED, dataflow.JobState.JOB_STATE_CANCELLED: beam_job_models.BeamJobState.CANCELLED, dataflow.JobState.JOB_STATE_UPDATED: beam_job_models.BeamJobState.UPDATED, dataflow.JobState.JOB_STATE_DRAINING: beam_job_models.BeamJobState.DRAINING, dataflow.JobState.JOB_STATE_DRAINED: beam_job_models.BeamJobState.DRAINED, dataflow.JobState.JOB_STATE_PENDING: beam_job_models.BeamJobState.PENDING, dataflow.JobState.JOB_STATE_QUEUED: beam_job_models.BeamJobState.PENDING, dataflow.JobState.JOB_STATE_CANCELLING: beam_job_models.BeamJobState.CANCELLING}

def run_job(job_class: Type[base_jobs.JobBase], sync: bool, namespace: Optional[str]=None, pipeline: Optional[beam.Pipeline]=None) -> beam_job_models.BeamJobRunModel:
    if False:
        print('Hello World!')
    'Runs the specified job synchronously.\n\n    In other words, the function will wait for the job to finish running before\n    returning a value.\n\n    Args:\n        job_class: type(base_jobs.JobBase). The type of job to run.\n        sync: bool. Whether to run the job synchronously.\n        namespace: str. The namespace in which models should be created.\n        pipeline: Pipeline. The pipeline to run the job upon. If omitted, then a\n            new pipeline will be used instead.\n\n    Returns:\n        BeamJobRun. Contains metadata related to the execution status of the\n        job.\n\n    Raises:\n        RuntimeError. Failed to deploy given job to the Dataflow service.\n    '
    if pipeline is None:
        pipeline = beam.Pipeline(runner=runners.DirectRunner() if sync else runners.DataflowRunner(), options=job_options.JobOptions(namespace=namespace))
    job = job_class(pipeline)
    job_name = job_class.__name__
    caching_services.flush_memory_caches()
    with _job_bookkeeping_context(job_name) as run_model:
        _ = job.run() | job_io.PutResults(run_model.id) | cache_io.FlushCache()
        run_result = pipeline.run()
        if sync:
            run_result.wait_until_finish()
            run_model.latest_job_state = beam_job_models.BeamJobState.DONE.value
        elif run_result.has_job:
            run_model.dataflow_job_id = run_result.job_id()
            run_model.latest_job_state = run_result.state
        else:
            raise RuntimeError('Failed to deploy %s to the Dataflow service. Please try again after a few minutes.' % job_name)
    with datastore_services.get_ndb_context() as ndb_context:
        ndb_context.clear_cache()
    return run_model

def refresh_state_of_beam_job_run_model(beam_job_run_model: beam_job_models.BeamJobRunModel) -> None:
    if False:
        print('Hello World!')
    'Refreshs the state of the given BeamJobRunModel.\n\n    Args:\n        beam_job_run_model: BeamJobRunModel. The model to update.\n    '
    job_id = beam_job_run_model.dataflow_job_id
    if job_id is None:
        beam_job_run_model.latest_job_state = beam_job_models.BeamJobState.UNKNOWN.value
        beam_job_run_model.update_timestamps(update_last_updated_time=False)
        return
    try:
        job = dataflow.JobsV1Beta3Client().get_job(dataflow.GetJobRequest(job_id=job_id, project_id=feconf.OPPIA_PROJECT_ID, location=feconf.GOOGLE_APP_ENGINE_REGION))
    except Exception as e:
        job_state = beam_job_models.BeamJobState.UNKNOWN.value
        job_state_updated = beam_job_run_model.last_updated
        logging.warning('Failed to update job_id="%s": %s', job_id, e)
    else:
        job_state = _GCLOUD_DATAFLOW_JOB_STATE_TO_OPPIA_BEAM_JOB_STATE.get(job.current_state, beam_job_models.BeamJobState.UNKNOWN).value
        job_state_updated = job.current_state_time.replace(tzinfo=None)
        if beam_job_run_model.latest_job_state == beam_job_models.BeamJobState.CANCELLING.value and job_state != beam_job_models.BeamJobState.CANCELLED.value:
            job_state = beam_job_run_model.latest_job_state
            job_state_updated = beam_job_run_model.last_updated
        if beam_job_run_model.latest_job_state != job_state and job_state == beam_job_models.BeamJobState.FAILED.value:
            _put_job_stderr(beam_job_run_model.id, pprint.pformat(job))
    beam_job_run_model.latest_job_state = job_state
    beam_job_run_model.last_updated = job_state_updated
    beam_job_run_model.update_timestamps(update_last_updated_time=False)

def cancel_job(beam_job_run_model: beam_job_models.BeamJobRunModel) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Refreshs the state of the given BeamJobRunModel.\n\n    Args:\n        beam_job_run_model: BeamJobRunModel. The model to update.\n\n    Raises:\n        ValueError. The given model has no job ID.\n    '
    job_id = beam_job_run_model.dataflow_job_id
    if job_id is None:
        raise ValueError('dataflow_job_id must not be None')
    try:
        dataflow.JobsV1Beta3Client().update_job(dataflow.UpdateJobRequest(job_id=job_id, project_id=feconf.OPPIA_PROJECT_ID, location=feconf.GOOGLE_APP_ENGINE_REGION, job=dataflow.Job(requested_state=dataflow.JobState.JOB_STATE_CANCELLED)))
    except Exception:
        logging.exception('Failed to cancel job_id="%s"!' % job_id)
    else:
        beam_job_run_model.latest_job_state = beam_job_models.BeamJobState.CANCELLING.value
        beam_job_run_model.update_timestamps()

@contextlib.contextmanager
def _job_bookkeeping_context(job_name: str) -> Iterator[beam_job_models.BeamJobRunModel]:
    if False:
        i = 10
        return i + 15
    'Returns a context manager which commits failure details if an exception\n    occurs.\n\n    Args:\n        job_name: str. The name of the job.\n\n    Yields:\n        BeamJobRunModel. The bookkeeping model used to record execution details.\n    '
    run_model = beam_job_services.create_beam_job_run_model(job_name)
    try:
        yield run_model
    except Exception as exception:
        run_model.latest_job_state = beam_job_models.BeamJobState.FAILED.value
        _put_job_stderr(run_model.id, str(exception))
    finally:
        run_model.put()

def _put_job_stderr(job_id: str, stderr: str) -> None:
    if False:
        i = 10
        return i + 15
    'Puts the given error string as a result from the given job.\n\n    Args:\n        job_id: str. The ID of the job that failed.\n        stderr: str. The error output for the given job.\n    '
    result_model = beam_job_services.create_beam_job_run_result_model(job_id, '', stderr)
    result_model.put()