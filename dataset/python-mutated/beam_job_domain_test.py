"""Unit tests for core.domain.beam_job_domain."""
from __future__ import annotations
import datetime
from core import utils
from core.domain import beam_job_domain
from core.jobs.batch_jobs import model_validation_jobs
from core.tests import test_utils

class BeamJobTests(test_utils.TestBase):
    NOW = datetime.datetime.utcnow()

    def test_usage(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        job = beam_job_domain.BeamJob(model_validation_jobs.AuditAllStorageModelsJob)
        self.assertEqual(job.name, 'AuditAllStorageModelsJob')

    def test_to_dict(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        job = beam_job_domain.BeamJob(model_validation_jobs.AuditAllStorageModelsJob)
        self.assertEqual(job.to_dict(), {'name': 'AuditAllStorageModelsJob'})

class BeamJobRunTests(test_utils.TestBase):
    NOW = datetime.datetime.utcnow()

    def test_usage(self) -> None:
        if False:
            i = 10
            return i + 15
        run = beam_job_domain.BeamJobRun('123', 'FooJob', 'RUNNING', self.NOW, self.NOW, True)
        self.assertEqual(run.job_id, '123')
        self.assertEqual(run.job_name, 'FooJob')
        self.assertEqual(run.job_state, 'RUNNING')
        self.assertEqual(run.job_started_on, self.NOW)
        self.assertEqual(run.job_updated_on, self.NOW)
        self.assertTrue(run.job_is_synchronous)

    def test_to_dict(self) -> None:
        if False:
            i = 10
            return i + 15
        run = beam_job_domain.BeamJobRun('123', 'FooJob', 'RUNNING', self.NOW, self.NOW, True)
        self.assertEqual(run.to_dict(), {'job_id': '123', 'job_name': 'FooJob', 'job_state': 'RUNNING', 'job_started_on_msecs': utils.get_time_in_millisecs(self.NOW), 'job_updated_on_msecs': utils.get_time_in_millisecs(self.NOW), 'job_is_synchronous': True})

class AggregateBeamJobRunResultTests(test_utils.TestBase):

    def test_usage(self) -> None:
        if False:
            i = 10
            return i + 15
        result = beam_job_domain.AggregateBeamJobRunResult('abc', '123')
        self.assertEqual(result.stdout, 'abc')
        self.assertEqual(result.stderr, '123')

    def test_to_dict(self) -> None:
        if False:
            while True:
                i = 10
        result = beam_job_domain.AggregateBeamJobRunResult('abc', '123')
        self.assertEqual(result.to_dict(), {'stdout': 'abc', 'stderr': '123'})