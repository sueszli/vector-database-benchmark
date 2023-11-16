from __future__ import annotations
import time
import pytest
from kubernetes_tests.test_base import EXECUTOR, BaseK8STest

@pytest.mark.skipif(EXECUTOR == 'KubernetesExecutor', reason='Does not run on KubernetesExecutor')
class TestCeleryAndLocalExecutor(BaseK8STest):

    def test_integration_run_dag(self):
        if False:
            i = 10
            return i + 15
        dag_id = 'example_bash_operator'
        (dag_run_id, execution_date) = self.start_job_in_kubernetes(dag_id, self.host)
        print(f'Found the job with execution_date {execution_date}')
        self.monitor_task(host=self.host, dag_run_id=dag_run_id, dag_id=dag_id, task_id='run_after_loop', expected_final_state='success', timeout=300)
        self.ensure_dag_expected_state(host=self.host, execution_date=execution_date, dag_id=dag_id, expected_final_state='success', timeout=300)

    def test_integration_run_dag_with_scheduler_failure(self):
        if False:
            i = 10
            return i + 15
        dag_id = 'example_xcom'
        (dag_run_id, execution_date) = self.start_job_in_kubernetes(dag_id, self.host)
        self._delete_airflow_pod('scheduler')
        time.sleep(10)
        self.monitor_task(host=self.host, dag_run_id=dag_run_id, dag_id=dag_id, task_id='push', expected_final_state='success', timeout=40)
        self.monitor_task(host=self.host, dag_run_id=dag_run_id, dag_id=dag_id, task_id='puller', expected_final_state='success', timeout=40)
        self.ensure_dag_expected_state(host=self.host, execution_date=execution_date, dag_id=dag_id, expected_final_state='success', timeout=60)
        assert self._num_pods_in_namespace('test-namespace') == 0, 'failed to delete pods in other namespace'