from __future__ import annotations
import contextlib
from io import StringIO
import pytest
from airflow.cli import cli_parser
from airflow.cli.commands import jobs_command
from airflow.jobs.job import Job
from airflow.jobs.scheduler_job_runner import SchedulerJobRunner
from airflow.utils.session import create_session
from airflow.utils.state import JobState, State
from tests.test_utils.db import clear_db_jobs

@pytest.mark.db_test
class TestCliConfigList:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.parser = cli_parser.get_parser()

    def setup_method(self) -> None:
        if False:
            return 10
        clear_db_jobs()
        self.scheduler_job = None
        self.job_runner = None

    def teardown_method(self) -> None:
        if False:
            while True:
                i = 10
        if self.job_runner and self.job_runner.processor_agent:
            self.job_runner.processor_agent.end()
        clear_db_jobs()

    def test_should_report_success_for_one_working_scheduler(self):
        if False:
            return 10
        with create_session() as session:
            self.scheduler_job = Job()
            self.job_runner = SchedulerJobRunner(job=self.scheduler_job)
            self.scheduler_job.state = State.RUNNING
            session.add(self.scheduler_job)
            session.commit()
            self.scheduler_job.heartbeat(heartbeat_callback=self.job_runner.heartbeat_callback)
        with contextlib.redirect_stdout(StringIO()) as temp_stdout:
            jobs_command.check(self.parser.parse_args(['jobs', 'check', '--job-type', 'SchedulerJob']))
        assert 'Found one alive job.' in temp_stdout.getvalue()

    def test_should_report_success_for_one_working_scheduler_with_hostname(self):
        if False:
            return 10
        with create_session() as session:
            self.scheduler_job = Job()
            self.job_runner = SchedulerJobRunner(job=self.scheduler_job)
            self.scheduler_job.state = State.RUNNING
            self.scheduler_job.hostname = 'HOSTNAME'
            session.add(self.scheduler_job)
            session.commit()
            self.scheduler_job.heartbeat(heartbeat_callback=self.job_runner.heartbeat_callback)
        with contextlib.redirect_stdout(StringIO()) as temp_stdout:
            jobs_command.check(self.parser.parse_args(['jobs', 'check', '--job-type', 'SchedulerJob', '--hostname', 'HOSTNAME']))
        assert 'Found one alive job.' in temp_stdout.getvalue()

    def test_should_report_success_for_ha_schedulers(self):
        if False:
            return 10
        scheduler_jobs = []
        job_runners = []
        with create_session() as session:
            for _ in range(3):
                scheduler_job = Job()
                job_runner = SchedulerJobRunner(job=scheduler_job)
                scheduler_job.state = State.RUNNING
                session.add(scheduler_job)
                scheduler_jobs.append(scheduler_job)
                job_runners.append(job_runner)
            session.commit()
            scheduler_job.heartbeat(heartbeat_callback=job_runner.heartbeat_callback)
        try:
            with contextlib.redirect_stdout(StringIO()) as temp_stdout:
                jobs_command.check(self.parser.parse_args(['jobs', 'check', '--job-type', 'SchedulerJob', '--limit', '100', '--allow-multiple']))
            assert 'Found 3 alive jobs.' in temp_stdout.getvalue()
        finally:
            for job_runner in job_runners:
                if job_runner.processor_agent:
                    job_runner.processor_agent.end()

    def test_should_ignore_not_running_jobs(self):
        if False:
            return 10
        scheduler_jobs = []
        job_runners = []
        with create_session() as session:
            for _ in range(3):
                scheduler_job = Job()
                job_runner = SchedulerJobRunner(job=scheduler_job)
                scheduler_job.state = JobState.FAILED
                session.add(scheduler_job)
                scheduler_jobs.append(scheduler_job)
                job_runners.append(job_runner)
            session.commit()
        with pytest.raises(SystemExit, match='No alive jobs found.'):
            jobs_command.check(self.parser.parse_args(['jobs', 'check']))
        for job_runner in job_runners:
            if job_runner.processor_agent:
                job_runner.processor_agent.end()

    def test_should_raise_exception_for_multiple_scheduler_on_one_host(self):
        if False:
            while True:
                i = 10
        scheduler_jobs = []
        job_runners = []
        with create_session() as session:
            for _ in range(3):
                scheduler_job = Job()
                job_runner = SchedulerJobRunner(job=scheduler_job)
                job_runner.job = scheduler_job
                scheduler_job.state = State.RUNNING
                scheduler_job.hostname = 'HOSTNAME'
                session.add(scheduler_job)
                scheduler_jobs.append(scheduler_job)
                job_runners.append(job_runner)
            session.commit()
            scheduler_job.heartbeat(heartbeat_callback=job_runner.heartbeat_callback)
        with pytest.raises(SystemExit, match='Found 3 alive jobs. Expected only one.'):
            jobs_command.check(self.parser.parse_args(['jobs', 'check', '--job-type', 'SchedulerJob', '--limit', '100']))
        for job_runner in job_runners:
            if job_runner.processor_agent:
                job_runner.processor_agent.end()

    def test_should_raise_exception_for_allow_multiple_and_limit_1(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(SystemExit, match='To use option --allow-multiple, you must set the limit to a value greater than 1.'):
            jobs_command.check(self.parser.parse_args(['jobs', 'check', '--allow-multiple']))