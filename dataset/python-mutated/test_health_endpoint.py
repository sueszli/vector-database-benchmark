from __future__ import annotations
from datetime import timedelta
from unittest import mock
import pytest
from airflow.jobs.job import Job
from airflow.jobs.scheduler_job_runner import SchedulerJobRunner
from airflow.utils import timezone
from airflow.utils.session import create_session, provide_session
from airflow.utils.state import State
HEALTHY = 'healthy'
UNHEALTHY = 'unhealthy'
pytestmark = pytest.mark.db_test

class TestHealthTestBase:

    @pytest.fixture(autouse=True)
    def setup_attrs(self, minimal_app_for_api) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.app = minimal_app_for_api
        self.client = self.app.test_client()
        with create_session() as session:
            session.query(Job).delete()

    def teardown_method(self):
        if False:
            return 10
        with create_session() as session:
            session.query(Job).delete()

class TestGetHealth(TestHealthTestBase):

    @provide_session
    def test_healthy_scheduler_status(self, session):
        if False:
            print('Hello World!')
        last_scheduler_heartbeat_for_testing_1 = timezone.utcnow()
        job = Job(state=State.RUNNING, latest_heartbeat=last_scheduler_heartbeat_for_testing_1)
        SchedulerJobRunner(job=job)
        session.add(job)
        session.commit()
        resp_json = self.client.get('/api/v1/health').json
        assert 'healthy' == resp_json['metadatabase']['status']
        assert 'healthy' == resp_json['scheduler']['status']
        assert last_scheduler_heartbeat_for_testing_1.isoformat() == resp_json['scheduler']['latest_scheduler_heartbeat']

    @provide_session
    def test_unhealthy_scheduler_is_slow(self, session):
        if False:
            print('Hello World!')
        last_scheduler_heartbeat_for_testing_2 = timezone.utcnow() - timedelta(minutes=1)
        job = Job(state=State.RUNNING, latest_heartbeat=last_scheduler_heartbeat_for_testing_2)
        SchedulerJobRunner(job=job)
        session.add(job)
        session.commit()
        resp_json = self.client.get('/api/v1/health').json
        assert 'healthy' == resp_json['metadatabase']['status']
        assert 'unhealthy' == resp_json['scheduler']['status']
        assert last_scheduler_heartbeat_for_testing_2.isoformat() == resp_json['scheduler']['latest_scheduler_heartbeat']

    def test_unhealthy_scheduler_no_job(self):
        if False:
            for i in range(10):
                print('nop')
        resp_json = self.client.get('/api/v1/health').json
        assert 'healthy' == resp_json['metadatabase']['status']
        assert 'unhealthy' == resp_json['scheduler']['status']
        assert resp_json['scheduler']['latest_scheduler_heartbeat'] is None

    @mock.patch.object(SchedulerJobRunner, 'most_recent_job')
    def test_unhealthy_metadatabase_status(self, most_recent_job_mock):
        if False:
            i = 10
            return i + 15
        most_recent_job_mock.side_effect = Exception
        resp_json = self.client.get('/api/v1/health').json
        assert 'unhealthy' == resp_json['metadatabase']['status']
        assert resp_json['scheduler']['latest_scheduler_heartbeat'] is None