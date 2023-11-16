from unittest.mock import MagicMock, Mock
from twisted.internet import defer
from twisted.trial.unittest import TestCase as TwistedTestCase
from golem.envs import Environment, EnvMetadata
from golem.model import Performance
from golem.task.task_api import TaskApiPayloadBuilder
from golem.task.envmanager import EnvironmentManager
from golem.testutils import DatabaseFixture

class EnvManagerBaseTest(DatabaseFixture):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.manager = EnvironmentManager(self.new_path)

    def register_env(self, env_id):
        if False:
            return 10
        env = MagicMock(spec=Environment)
        env.prepare.return_value = defer.succeed(None)
        env.clean_up.return_value = defer.succeed(None)
        metadata = EnvMetadata(id=env_id)
        payload_builder = MagicMock(sepc_set=TaskApiPayloadBuilder)
        self.manager.register_env(env, metadata, payload_builder)
        return (env, metadata, payload_builder)

class TestEnvironmentManager(EnvManagerBaseTest):

    def test_register_env(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.manager.environments(), [])
        self.assertEqual(self.manager.state(), {})
        self.register_env('env1')
        self.assertEqual(self.manager.environments(), ['env1'])
        self.assertEqual(self.manager.state(), {'env1': False})

    def test_re_register_env(self):
        if False:
            while True:
                i = 10
        (env, metadata, _) = self.register_env('env1')
        self.manager.set_enabled('env1', True)
        self.assertEqual(self.manager.environments(), ['env1'])
        self.assertTrue(self.manager.enabled('env1'))
        with self.assertRaises(ValueError):
            self.manager.register_env(env, metadata, MagicMock(spec=TaskApiPayloadBuilder))
        self.assertEqual(self.manager.environments(), ['env1'])
        self.assertTrue(self.manager.enabled('env1'))

    def test_set_enabled(self):
        if False:
            i = 10
            return i + 15
        self.register_env('env1')
        self.assertFalse(self.manager.enabled('env1'))
        self.manager.set_enabled('env1', True)
        self.assertTrue(self.manager.enabled('env1'))
        self.manager.set_enabled('env1', False)
        self.assertFalse(self.manager.enabled('env1'))

    def test_set_state(self):
        if False:
            while True:
                i = 10
        self.register_env('env1')
        self.register_env('env2')
        self.assertFalse(self.manager.enabled('env1'))
        self.assertFalse(self.manager.enabled('env2'))
        self.manager.set_state({'env2': True, 'bogus_env': True})
        self.assertFalse(self.manager.enabled('env1'))
        self.assertTrue(self.manager.enabled('env2'))
        self.assertFalse(self.manager.enabled('bogus_env'))

    def test_payload_builder(self):
        if False:
            while True:
                i = 10
        (*_, pb) = self.register_env('env1')
        self.assertEqual(pb, self.manager.payload_builder('env1'))

class TestAutoSetup(EnvManagerBaseTest, TwistedTestCase):

    @defer.inlineCallbacks
    def test_auto_setup(self):
        if False:
            while True:
                i = 10
        (env1, *_) = self.register_env('env1')
        (env2, *_) = self.register_env('env2')
        wrapped_env1 = self.manager.environment('env1')
        wrapped_env2 = self.manager.environment('env2')
        runtime1 = wrapped_env1.runtime(Mock())
        runtime2 = wrapped_env2.runtime(Mock())
        yield runtime1.prepare()
        env1.prepare.assert_called_once()
        yield runtime1.clean_up()
        env1.clean_up.assert_not_called()
        yield runtime2.prepare()
        env1.clean_up.assert_called_once()
        env2.prepare.assert_called_once()

class TestRuntimeLogs(EnvManagerBaseTest, TwistedTestCase):

    @defer.inlineCallbacks
    def test_runtime_logs(self):
        if False:
            print('Hello World!')
        env_id = 'env'
        runtime_id = 'runtime'
        stdout = ['ąąą\n', 'bbb\n', 'ććć\n']
        stderr = ['ddd\n', 'ęęę\n', 'fff\n']
        (env, *_) = self.register_env(env_id)
        env.runtime().id.return_value = runtime_id
        env.runtime().stdout.return_value = stdout
        env.runtime().stderr.return_value = stderr
        wrapped_env = self.manager.environment('env')
        runtime = wrapped_env.runtime(Mock())
        yield runtime.prepare()
        yield runtime.clean_up()
        stdout_path = self.new_path / env_id / f'{runtime_id}_stdout.txt'
        self.assertTrue(stdout_path.exists())
        with stdout_path.open(mode='r', encoding='utf-8') as file:
            self.assertEqual(list(file), stdout)
        stderr_path = self.new_path / env_id / f'{runtime_id}_stderr.txt'
        self.assertTrue(stderr_path.exists())
        with stderr_path.open(mode='r', encoding='utf-8') as file:
            self.assertEqual(list(file), stderr)

class TestEnvironmentManagerDB(EnvManagerBaseTest, TwistedTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.env_id = 'env1'
        (self.env, *_) = self.register_env(self.env_id)

    @defer.inlineCallbacks
    def test_get_performance_running(self):
        if False:
            for i in range(10):
                print('nop')
        self.manager._running_benchmark = True
        result = (yield self.manager.get_benchmark_result(self.env_id))
        self.env.run_benchmark.assert_not_called()
        self.assertIsNone(result)

    @defer.inlineCallbacks
    def test_get_performance_disabled_env(self):
        if False:
            return 10
        self.manager.set_enabled(self.env_id, False)
        with self.assertRaisesRegex(Exception, 'Requested performance for disabled environment'):
            yield self.manager.get_benchmark_result(self.env_id)
        self.env.run_benchmark.assert_not_called()

    @defer.inlineCallbacks
    def test_get_performance_in_db(self):
        if False:
            i = 10
            return i + 15
        perf = 300.0
        self.manager.set_enabled(self.env_id, True)
        Performance.update_or_create(self.env_id, perf, 0)
        result = (yield self.manager.get_benchmark_result(self.env_id))
        self.env.run_benchmark.assert_not_called()
        self.assertEqual(result.performance, perf)

    @defer.inlineCallbacks
    def test_get_performance_benchmark_error(self):
        if False:
            print('Hello World!')
        error_msg = 'Benchmark failed'
        self.env.run_benchmark = Mock(side_effect=Exception(error_msg))
        self.manager.set_enabled(self.env_id, True)
        result = None
        with self.assertRaisesRegex(Exception, error_msg):
            result = (yield self.manager.get_benchmark_result(self.env_id))
        self.env.run_benchmark.assert_called_once()
        self.assertIsNone(result)

    def test_cached_performance(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNone(self.manager.get_cached_benchmark_result(self.env_id))
        perf = 123.4
        Performance.update_or_create(self.env_id, perf, 0)
        self.assertEqual(perf, self.manager.get_cached_benchmark_result(self.env_id).performance)

    def test_remove_cached_performance(self):
        if False:
            return 10
        perf = 123.4
        Performance.update_or_create(self.env_id, perf, 0)
        self.assertEqual(perf, self.manager.get_cached_benchmark_result(self.env_id).performance)
        self.manager.remove_cached_performance(self.env_id)
        self.assertIsNone(self.manager.get_cached_benchmark_result(self.env_id))