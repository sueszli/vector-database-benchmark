from __future__ import annotations
from datetime import datetime
from unittest.mock import MagicMock
from airflow.api.common.airflow_health import HEALTHY, UNHEALTHY, DagProcessorJobRunner, SchedulerJobRunner, TriggererJobRunner, get_airflow_health

def test_get_airflow_health_only_metadatabase_healthy():
    if False:
        i = 10
        return i + 15
    SchedulerJobRunner.most_recent_job = MagicMock(return_value=None)
    TriggererJobRunner.most_recent_job = MagicMock(return_value=None)
    DagProcessorJobRunner.most_recent_job = MagicMock(return_value=None)
    health_status = get_airflow_health()
    expected_status = {'metadatabase': {'status': HEALTHY}, 'scheduler': {'status': UNHEALTHY, 'latest_scheduler_heartbeat': None}, 'triggerer': {'status': None, 'latest_triggerer_heartbeat': None}, 'dag_processor': {'status': None, 'latest_dag_processor_heartbeat': None}}
    assert health_status == expected_status

def test_get_airflow_health_metadatabase_unhealthy():
    if False:
        while True:
            i = 10
    SchedulerJobRunner.most_recent_job = MagicMock(side_effect=Exception)
    TriggererJobRunner.most_recent_job = MagicMock(side_effect=Exception)
    DagProcessorJobRunner.most_recent_job = MagicMock(side_effect=Exception)
    health_status = get_airflow_health()
    expected_status = {'metadatabase': {'status': UNHEALTHY}, 'scheduler': {'status': UNHEALTHY, 'latest_scheduler_heartbeat': None}, 'triggerer': {'status': UNHEALTHY, 'latest_triggerer_heartbeat': None}, 'dag_processor': {'status': UNHEALTHY, 'latest_dag_processor_heartbeat': None}}
    assert health_status == expected_status

def test_get_airflow_health_scheduler_healthy_no_triggerer():
    if False:
        while True:
            i = 10
    latest_scheduler_job_mock = MagicMock()
    latest_scheduler_job_mock.latest_heartbeat = datetime.now()
    latest_scheduler_job_mock.is_alive = MagicMock(return_value=True)
    SchedulerJobRunner.most_recent_job = MagicMock(return_value=latest_scheduler_job_mock)
    TriggererJobRunner.most_recent_job = MagicMock(return_value=None)
    DagProcessorJobRunner.most_recent_job = MagicMock(return_value=None)
    health_status = get_airflow_health()
    expected_status = {'metadatabase': {'status': HEALTHY}, 'scheduler': {'status': HEALTHY, 'latest_scheduler_heartbeat': latest_scheduler_job_mock.latest_heartbeat.isoformat()}, 'triggerer': {'status': None, 'latest_triggerer_heartbeat': None}, 'dag_processor': {'status': None, 'latest_dag_processor_heartbeat': None}}
    assert health_status == expected_status

def test_get_airflow_health_triggerer_healthy_no_scheduler_job_record():
    if False:
        i = 10
        return i + 15
    latest_triggerer_job_mock = MagicMock()
    latest_triggerer_job_mock.latest_heartbeat = datetime.now()
    latest_triggerer_job_mock.is_alive = MagicMock(return_value=True)
    latest_dag_processor_job_mock = MagicMock()
    latest_dag_processor_job_mock.latest_heartbeat = datetime.now()
    latest_dag_processor_job_mock.is_alive = MagicMock(return_value=True)
    SchedulerJobRunner.most_recent_job = MagicMock(return_value=None)
    TriggererJobRunner.most_recent_job = MagicMock(return_value=latest_triggerer_job_mock)
    DagProcessorJobRunner.most_recent_job = MagicMock(return_value=latest_dag_processor_job_mock)
    health_status = get_airflow_health()
    expected_status = {'metadatabase': {'status': HEALTHY}, 'scheduler': {'status': UNHEALTHY, 'latest_scheduler_heartbeat': None}, 'triggerer': {'status': HEALTHY, 'latest_triggerer_heartbeat': latest_triggerer_job_mock.latest_heartbeat.isoformat()}, 'dag_processor': {'status': HEALTHY, 'latest_dag_processor_heartbeat': latest_dag_processor_job_mock.latest_heartbeat.isoformat()}}
    assert health_status == expected_status