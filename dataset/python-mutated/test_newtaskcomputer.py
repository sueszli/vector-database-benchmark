import asyncio
import time
from pathlib import Path
from unittest import mock
from golem_messages.message import ComputeTaskDef
from golem_task_api import ProviderAppClient, TaskApiService
from golem_task_api.envs import DOCKER_CPU_ENV_ID
from twisted.internet import defer
from golem.clientconfigdescriptor import ClientConfigDescriptor
from golem.core.deferred import deferred_from_future
from golem.core.statskeeper import IntStatsKeeper
from golem.envs import Runtime
from golem.envs.docker.cpu import DockerCPUConfig
from golem.task.envmanager import EnvironmentManager
from golem.task.taskcomputer import NewTaskComputer
from golem.testutils import TempDirFixture
from tests.utils.asyncio import TwistedAsyncioTestCase

class NewTaskComputerTestBase(TwistedAsyncioTestCase, TempDirFixture):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')

        def enabled(env_id):
            if False:
                print('Hello World!')
            return env_id == DOCKER_CPU_ENV_ID
        super().setUp()
        self.env_manager = mock.Mock(spec=EnvironmentManager)
        self.env_manager.enabled.side_effect = enabled
        self.stats_keeper = mock.Mock(spec=IntStatsKeeper)
        self.work_dir = self.new_path
        self.task_computer = NewTaskComputer(env_manager=self.env_manager, work_dir=self.work_dir, stats_keeper=self.stats_keeper)

    @property
    def task_id(self):
        if False:
            while True:
                i = 10
        return 'test_task'

    @property
    def subtask_id(self):
        if False:
            while True:
                i = 10
        return 'test_subtask'

    @property
    def subtask_params(self):
        if False:
            i = 10
            return i + 15
        return {'test_param': 'test_value'}

    @property
    def env_id(self):
        if False:
            return 10
        return 'test_env'

    @property
    def prereq_dict(self):
        if False:
            return 10
        return {'test_prereq': 'test_value'}

    @property
    def performance(self):
        if False:
            return 10
        return 2137

    @property
    def subtask_timeout(self):
        if False:
            i = 10
            return i + 15
        return 3600

    @property
    def task_deadline(self):
        if False:
            while True:
                i = 10
        return int(time.time()) + 3600

    @property
    def subtask_deadline(self):
        if False:
            return 10
        return int(time.time()) + 3600

    def _get_task_header(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return mock.Mock(task_id=kwargs.get('task_id') or self.task_id, environment=kwargs.get('env_id') or self.env_id, environment_prerequisites=kwargs.get('prereq_dict') or self.prereq_dict, subtask_timeout=kwargs.get('subtask_timeout') or self.subtask_timeout, deadline=kwargs.get('task_deadline') or self.task_deadline)

    def _get_compute_task_def(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return ComputeTaskDef(subtask_id=kwargs.get('subtask_id') or self.subtask_id, extra_data=kwargs.get('subtask_params') or self.subtask_params, performance=kwargs.get('performance') or self.performance, deadline=kwargs.get('subtask_deadline') or self.subtask_deadline)

    def _patch_async(self, name, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        patcher = mock.patch(f'golem.task.taskcomputer.{name}', *args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _assign_task(self, **kwargs):
        if False:
            while True:
                i = 10
        task_header = self._get_task_header(**kwargs)
        compute_task_def = self._get_compute_task_def(**kwargs)
        self.task_computer.task_given(task_header, compute_task_def)

@mock.patch('golem.task.taskcomputer.ProviderTimer')
class TestTaskGiven(NewTaskComputerTestBase):

    def test_ok(self, provider_timer):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self.task_computer.has_assigned_task())
        self.assertIsNone(self.task_computer.assigned_task_id)
        self.assertIsNone(self.task_computer.assigned_subtask_id)
        self.assertIsNone(self.task_computer.get_current_computing_env())
        task_header = self._get_task_header()
        compute_task_def = self._get_compute_task_def()
        self.task_computer.task_given(task_header, compute_task_def)
        self.assertTrue(self.task_computer.has_assigned_task())
        self.assertEqual(self.task_computer.assigned_task_id, self.task_id)
        self.assertEqual(self.task_computer.assigned_subtask_id, self.subtask_id)
        self.assertEqual(self.task_computer.get_current_computing_env(), self.env_id)
        provider_timer.start.assert_called_once_with()
        self.assertTrue(self.task_computer.get_subtask_inputs_dir().exists())

    def test_has_assigned_task(self, provider_timer):
        if False:
            for i in range(10):
                print('nop')
        task_header = self._get_task_header()
        compute_task_def = self._get_compute_task_def()
        self.task_computer.task_given(task_header, compute_task_def)
        provider_timer.reset_mock()
        with self.assertRaises(AssertionError):
            self.task_computer.task_given(task_header, compute_task_def)
        provider_timer.start.assert_not_called()

class TestCompute(NewTaskComputerTestBase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.runtime = mock.Mock(spec_set=Runtime)
        self.compute_future = asyncio.Future()
        self._patch_async('NewTaskComputer._create_client_and_compute', side_effect=lambda : self.compute_future)
        self.task_dir = Path('task_dir')
        self._patch_async('NewTaskComputer._get_task_dir', return_value=self.task_dir)
        self.provider_timer = self._patch_async('ProviderTimer')
        self.dispatcher = self._patch_async('dispatcher')
        self.logger = self._patch_async('logger')

    @defer.inlineCallbacks
    def test_no_assigned_task(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AssertionError):
            yield self.task_computer.compute()

    @defer.inlineCallbacks
    def test_ok(self):
        if False:
            return 10
        self._assign_task()
        self.compute_future.set_result('result.txt')
        result = (yield self.task_computer.compute())
        self.assertEqual(result, self.task_dir / 'result.txt')
        self.stats_keeper.increase_stat.assert_called_once_with('computed_tasks')
        self.dispatcher.send.assert_has_calls((mock.call(signal='golem.taskcomputer', event='subtask_finished', subtask_id=self.subtask_id, min_performance=self.performance), mock.call(signal='golem.monitor', event='computation_time_spent', success=True, value=self.subtask_timeout)), any_order=True)
        self.provider_timer.finish.assert_called_once()
        self.assertFalse(self.task_computer.has_assigned_task())

    @defer.inlineCallbacks
    def test_task_interrupted(self):
        if False:
            print('Hello World!')
        self._assign_task()
        deferred = self.task_computer.compute()
        self.task_computer.task_interrupted()
        result = (yield deferred)
        self.assertIsNone(result)
        self.logger.warning.assert_called_once()
        self.stats_keeper.increase_stat.assert_not_called()
        self.dispatcher.send.assert_has_calls((mock.call(signal='golem.taskcomputer', event='subtask_finished', subtask_id=self.subtask_id, min_performance=self.performance), mock.call(signal='golem.monitor', event='computation_time_spent', success=False, value=self.subtask_timeout)), any_order=True)
        self.provider_timer.finish.assert_called_once()
        self.assertFalse(self.task_computer.has_assigned_task())

    @defer.inlineCallbacks
    def test_task_timed_out(self):
        if False:
            i = 10
            return i + 15
        self._assign_task(subtask_deadline=time.time())
        self.compute_future = asyncio.sleep(10)
        with self.assertRaisesRegex(RuntimeError, 'Task computation timed out'):
            yield self.task_computer.compute()
        self.logger.error.assert_called_once()
        self.stats_keeper.increase_stat.assert_called_once_with('tasks_with_timeout')
        self.dispatcher.send.assert_has_calls((mock.call(signal='golem.taskcomputer', event='subtask_finished', subtask_id=self.subtask_id, min_performance=self.performance), mock.call(signal='golem.monitor', event='computation_time_spent', success=False, value=self.subtask_timeout)), any_order=True)
        self.provider_timer.finish.assert_called_once()
        self.assertFalse(self.task_computer.has_assigned_task())

    @defer.inlineCallbacks
    def test_task_error(self):
        if False:
            i = 10
            return i + 15
        self._assign_task()
        self.compute_future.set_exception(OSError)
        with self.assertRaises(OSError):
            yield self.task_computer.compute()
        self.logger.exception.assert_called_once()
        self.stats_keeper.increase_stat.assert_called_once_with('tasks_with_errors')
        self.dispatcher.send.assert_has_calls((mock.call(signal='golem.taskcomputer', event='subtask_finished', subtask_id=self.subtask_id, min_performance=self.performance), mock.call(signal='golem.monitor', event='computation_time_spent', success=False, value=self.subtask_timeout)), any_order=True)
        self.provider_timer.finish.assert_called_once()
        self.assertFalse(self.task_computer.has_assigned_task())

class TestCreateClientAndCompute(NewTaskComputerTestBase):

    @defer.inlineCallbacks
    def test_client_client_and_compute(self):
        if False:
            while True:
                i = 10
        service = mock.Mock(spec_set=TaskApiService)
        task_api_service_cls = self._patch_async('EnvironmentTaskApiService')
        task_api_service_cls.return_value = service
        client = mock.Mock(spec_set=ProviderAppClient)
        result_path = Path('test_result')
        compute_future = asyncio.Future()
        compute_future.set_result(result_path)
        client.compute.return_value = compute_future
        client_future = asyncio.Future()
        client_future.set_result(client)
        provider_app_client_cls = self._patch_async('ProviderAppClient')
        provider_app_client_cls.create.return_value = client_future
        self._assign_task()
        result_future = asyncio.ensure_future(self.task_computer._create_client_and_compute())
        result = (yield deferred_from_future(result_future))
        self.assertEqual(result, result_path)
        self.env_manager.environment.assert_called_once_with(self.env_id)
        self.env_manager.payload_builder.assert_called_once_with(self.env_id)
        self.env_manager.environment().parse_prerequisites.assert_called_once_with(self.prereq_dict)
        task_api_service_cls.assert_called_once_with(env=self.env_manager.environment(), prereq=self.env_manager.environment().parse_prerequisites(), shared_dir=self.work_dir / self.env_id / self.task_id, payload_builder=self.env_manager.payload_builder())
        provider_app_client_cls.create.assert_called_once_with(service)
        client.compute.assert_called_once_with(task_id=self.task_id, subtask_id=self.subtask_id, subtask_params=self.subtask_params)

class TestChangeConfig(NewTaskComputerTestBase):

    def test_ok(self):
        if False:
            print('Hello World!')
        config_desc = ClientConfigDescriptor()
        config_desc.num_cores = 13
        config_desc.max_memory_size = 1024 * 1024
        self.task_computer.change_config(config_desc)
        self.env_manager.environment.assert_called_once_with(DOCKER_CPU_ENV_ID)
        self.env_manager.environment().update_config.assert_called_once_with(DockerCPUConfig(work_dirs=[self.work_dir], cpu_count=13, memory_mb=1024))