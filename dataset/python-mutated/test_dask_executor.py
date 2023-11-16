from __future__ import annotations
from datetime import timedelta
from unittest import mock
import pytest
from distributed import LocalCluster
from airflow.exceptions import AirflowException
from airflow.jobs.backfill_job_runner import BackfillJobRunner
from airflow.jobs.job import Job, run_job
from airflow.models import DagBag
from airflow.providers.daskexecutor.executors.dask_executor import DaskExecutor
from airflow.utils import timezone
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test
try:
    from distributed import tests
    from distributed.utils_test import cluster as dask_testing_cluster, get_cert, tls_security
    skip_tls_tests = False
except ImportError:
    skip_tls_tests = True

    def get_cert(x):
        if False:
            for i in range(10):
                print('nop')
        return x
DEFAULT_DATE = timezone.datetime(2017, 1, 1)
SUCCESS_COMMAND = ['airflow', 'tasks', 'run', '--help']
FAIL_COMMAND = ['airflow', 'tasks', 'run', 'false']
skip_dask_tests = False

@pytest.mark.skipif(skip_dask_tests, reason='The tests are skipped because it needs testing from Dask team')
class TestBaseDask:

    def assert_tasks_on_executor(self, executor, timeout_executor=120):
        if False:
            print('Hello World!')
        executor.start()
        executor.execute_async(key='success', command=SUCCESS_COMMAND)
        executor.execute_async(key='fail', command=FAIL_COMMAND)
        success_future = next((k for (k, v) in executor.futures.items() if v == 'success'))
        fail_future = next((k for (k, v) in executor.futures.items() if v == 'fail'))
        timeout = timezone.utcnow() + timedelta(seconds=timeout_executor)
        while not (success_future.done() and fail_future.done()):
            if timezone.utcnow() > timeout:
                raise ValueError('The futures should have finished; there is probably an error communicating with the Dask cluster.')
        assert success_future.done()
        assert fail_future.done()
        assert success_future.exception() is None
        assert fail_future.exception() is not None

@pytest.mark.skipif(skip_dask_tests, reason='The tests are skipped because it needs testing from Dask team')
class TestDaskExecutor(TestBaseDask):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.dagbag = DagBag(include_examples=True)
        self.cluster = LocalCluster()

    def test_supports_pickling(self):
        if False:
            for i in range(10):
                print('nop')
        assert not DaskExecutor.supports_pickling

    def test_supports_sentry(self):
        if False:
            return 10
        assert not DaskExecutor.supports_sentry

    def test_dask_executor_functions(self):
        if False:
            while True:
                i = 10
        executor = DaskExecutor(cluster_address=self.cluster.scheduler_address)
        self.assert_tasks_on_executor(executor, timeout_executor=120)

    @pytest.mark.quarantined
    @pytest.mark.execution_timeout(180)
    def test_backfill_integration(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that DaskExecutor can be used to backfill example dags\n        '
        dag = self.dagbag.get_dag('example_bash_operator')
        job = Job(executor=DaskExecutor(cluster_address=self.cluster.scheduler_address))
        job_runner = BackfillJobRunner(job=job, dag=dag, start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_first_depends_on_past=True)
        run_job(job=job, execute_callable=job_runner._execute)

    def teardown_method(self):
        if False:
            print('Hello World!')
        self.cluster.close(timeout=5)

@pytest.mark.skipif(skip_tls_tests, reason='The tests are skipped because distributed framework could not be imported')
class TestDaskExecutorTLS(TestBaseDask):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.dagbag = DagBag(include_examples=True)

    @conf_vars({('dask', 'tls_ca'): 'certs/tls-ca-cert.pem', ('dask', 'tls_cert'): 'certs/tls-key-cert.pem', ('dask', 'tls_key'): 'certs/tls-key.pem'})
    def test_tls(self):
        if False:
            while True:
                i = 10
        with dask_testing_cluster(worker_kwargs={'security': tls_security(), 'protocol': 'tls'}, scheduler_kwargs={'security': tls_security(), 'protocol': 'tls'}) as (cluster, _):
            executor = DaskExecutor(cluster_address=cluster['address'])
            self.assert_tasks_on_executor(executor, timeout_executor=120)
            executor.end()
            executor.client.close()

    @mock.patch('airflow.providers.daskexecutor.executors.dask_executor.DaskExecutor.sync')
    @mock.patch('airflow.executors.base_executor.BaseExecutor.trigger_tasks')
    @mock.patch('airflow.executors.base_executor.Stats.gauge')
    def test_gauge_executor_metrics(self, mock_stats_gauge, mock_trigger_tasks, mock_sync):
        if False:
            for i in range(10):
                print('nop')
        executor = DaskExecutor()
        executor.heartbeat()
        calls = [mock.call('executor.open_slots', mock.ANY), mock.call('executor.queued_tasks', mock.ANY), mock.call('executor.running_tasks', mock.ANY)]
        mock_stats_gauge.assert_has_calls(calls)

@pytest.mark.skipif(skip_dask_tests, reason='The tests are skipped because it needs testing from Dask team')
class TestDaskExecutorQueue:

    def test_dask_queues_no_resources(self):
        if False:
            return 10
        self.cluster = LocalCluster()
        executor = DaskExecutor(cluster_address=self.cluster.scheduler_address)
        executor.start()
        with pytest.raises(AirflowException):
            executor.execute_async(key='success', command=SUCCESS_COMMAND, queue='queue1')

    def test_dask_queues_not_available(self):
        if False:
            while True:
                i = 10
        self.cluster = LocalCluster(resources={'queue1': 1})
        executor = DaskExecutor(cluster_address=self.cluster.scheduler_address)
        executor.start()
        with pytest.raises(AirflowException):
            executor.execute_async(key='success', command=SUCCESS_COMMAND, queue='queue2')

    def test_dask_queues(self):
        if False:
            for i in range(10):
                print('nop')
        self.cluster = LocalCluster(resources={'queue1': 1})
        executor = DaskExecutor(cluster_address=self.cluster.scheduler_address)
        executor.start()
        executor.execute_async(key='success', command=SUCCESS_COMMAND, queue='queue1')
        success_future = next((k for (k, v) in executor.futures.items() if v == 'success'))
        timeout = timezone.utcnow() + timedelta(seconds=120)
        while not success_future.done():
            if timezone.utcnow() > timeout:
                raise ValueError('The futures should have finished; there is probably an error communicating with the Dask cluster.')
        assert success_future.done()
        assert success_future.exception() is None

    @pytest.mark.execution_timeout(120)
    def test_dask_queues_no_queue_specified(self):
        if False:
            return 10
        self.cluster = LocalCluster(resources={'queue1': 1})
        executor = DaskExecutor(cluster_address=self.cluster.scheduler_address)
        executor.start()
        executor.execute_async(key='success', command=SUCCESS_COMMAND)
        success_future = next((k for (k, v) in executor.futures.items() if v == 'success'))
        timeout = timezone.utcnow() + timedelta(seconds=100)
        while not success_future.done():
            if timezone.utcnow() > timeout:
                raise ValueError('The futures should have finished; there is probably an error communicating with the Dask cluster.')
        assert success_future.done()
        assert success_future.exception() is None

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.cluster.close(timeout=5)