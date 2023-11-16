"""Provides PTransforms for writing job results to the datastore."""
from __future__ import annotations
from core.domain import beam_job_services
from core.jobs.io import ndb_io
from core.jobs.types import job_run_result
from core.platform import models
import apache_beam as beam
from typing import Optional
MYPY = False
if MYPY:
    from mypy_imports import beam_job_models
    from mypy_imports import datastore_services
(beam_job_models,) = models.Registry.import_models([models.Names.BEAM_JOB])
datastore_services = models.Registry.import_datastore_services()

class PutResults(beam.PTransform):
    """Writes Job Results into the NDB datastore."""
    _MAX_RESULT_INSTANCES_PER_MODEL = 1000

    def __init__(self, job_id: str, label: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        'Initializes the GetModels PTransform.\n\n        Args:\n            job_id: str. The Oppia ID associated with the current pipeline.\n            label: str|None. The label of the PTransform.\n        '
        super().__init__(label=label)
        self.job_id = job_id

    def expand(self, results: beam.PCollection[job_run_result.JobRunResult]) -> beam.pvalue.PDone:
        if False:
            for i in range(10):
                print('nop')
        'Writes the given job results to the NDB datastore.\n\n        This overrides expand from parent class.\n\n        Args:\n            results: PCollection. Models, can also contain just one model.\n\n        Returns:\n            PCollection. An empty PCollection.\n        '
        return results | beam.WithKeys(None) | beam.GroupIntoBatches(self._MAX_RESULT_INSTANCES_PER_MODEL) | beam.Values() | beam.FlatMap(job_run_result.JobRunResult.accumulate) | beam.Map(self.create_beam_job_run_result_model, results.pipeline.options.namespace) | ndb_io.PutModels()

    def create_beam_job_run_result_model(self, result: job_run_result.JobRunResult, namespace: Optional[str]) -> beam_job_models.BeamJobRunResultModel:
        if False:
            for i in range(10):
                print('nop')
        'Returns an NDB model for storing the given JobRunResult.\n\n        Args:\n            result: job_run_result.JobRunResult. The result.\n            namespace: str. The namespace in which models should be created.\n\n        Returns:\n            BeamJobRunResultModel. The NDB model.\n        '
        with datastore_services.get_ndb_context(namespace=namespace):
            return beam_job_services.create_beam_job_run_result_model(self.job_id, result.stdout, result.stderr)