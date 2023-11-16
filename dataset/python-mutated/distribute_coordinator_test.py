"""Tests for Distribute Coordinator."""
import contextlib
import copy
import json
import os
import sys
import threading
import time
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_manager
CHIEF = distribute_coordinator._TaskType.CHIEF
WORKER = distribute_coordinator._TaskType.WORKER
PS = distribute_coordinator._TaskType.PS
EVALUATOR = distribute_coordinator._TaskType.EVALUATOR
STANDALONE_CLIENT = distribute_coordinator.CoordinatorMode.STANDALONE_CLIENT
INDEPENDENT_WORKER = distribute_coordinator.CoordinatorMode.INDEPENDENT_WORKER
NUM_WORKERS = 3
NUM_PS = 2
original_sys_exit = sys.exit

def _bytes_to_str(maybe_bytes):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(maybe_bytes, six.string_types):
        return maybe_bytes
    else:
        return str(maybe_bytes, 'utf-8')

def _strip_protocol(target):
    if False:
        return 10
    if '//' in target:
        return target.split('//')[1]
    else:
        return target

class MockExtended(object):

    def __init__(self, between_graph=False, should_init=None, should_checkpoint=None, should_save_summary=None):
        if False:
            while True:
                i = 10
        self.experimental_between_graph = between_graph
        self.experimental_should_init = should_init
        self.should_checkpoint = should_checkpoint
        self.should_save_summary = should_save_summary

class MockStrategy(object):

    def __init__(self, between_graph=False, should_init=None, should_checkpoint=None, should_save_summary=None):
        if False:
            for i in range(10):
                print('nop')
        self.extended = MockExtended(between_graph, should_init, should_checkpoint, should_save_summary)

    def configure(self, session_config=None, cluster_spec=None, task_type=None, task_id=None):
        if False:
            i = 10
            return i + 15
        if self.extended.experimental_should_init is None:
            if task_id == 0:
                self.extended.experimental_should_init = True
            else:
                self.extended.experimental_should_init = False
        if self.extended.should_checkpoint is None:
            if task_id == 0:
                self.extended.should_checkpoint = True
            else:
                self.extended.should_checkpoint = False
        if self.extended.should_save_summary is None:
            if task_id == 0:
                self.extended.should_save_summary = True
            else:
                self.extended.should_save_summary = False
        if session_config:
            if cluster_spec and task_type and (task_id is not None) and self.extended.experimental_between_graph:
                session_config.intra_op_parallelism_threads += 1
                if task_type in ['chief', 'worker']:
                    session_config.device_filters.extend(['/job:%s/task:%d' % (task_type, task_id), '/job:ps'])
            else:
                session_config.inter_op_parallelism_threads += 1
                session_config.device_filters.append('/job:somejob')

class MockServer(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._joined = False
        self._started = False

    def start(self):
        if False:
            i = 10
            return i + 15
        self._started = True

    def join(self):
        if False:
            print('Hello World!')
        assert not self._joined
        self._joined = True

    @property
    def joined(self):
        if False:
            print('Hello World!')
        return self._joined

    @property
    def started(self):
        if False:
            i = 10
            return i + 15
        return self._started

class DistributeCoordinatorTestBase(test.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        (cls._workers, cls._ps) = test_util.create_local_cluster(NUM_WORKERS, num_ps=NUM_PS)
        cls._cluster_spec = {WORKER: [_strip_protocol(_bytes_to_str(w.target)) for w in cls._workers], PS: [_strip_protocol(_bytes_to_str(ps.target)) for ps in cls._ps]}

    def setUp(self):
        if False:
            return 10
        self._result_correct = 0
        self._lock = threading.Lock()
        self._worker_context = {}
        self._strategy_property = {}
        self._std_servers = {}
        self._barrier = distribute_coordinator._Barrier(NUM_WORKERS)
        self._coord = coordinator.Coordinator()

    @contextlib.contextmanager
    def _test_session(self, target):
        if False:
            i = 10
            return i + 15
        config = config_pb2.ConfigProto(allow_soft_placement=True)
        config.graph_options.optimizer_options.opt_level = -1
        with session.Session(graph=None, config=config, target=target) as sess:
            yield sess

    def _create_cluster_spec(self, has_chief=False, num_workers=1, num_ps=0, has_eval=False):
        if False:
            while True:
                i = 10
        cluster_spec = {}
        if has_chief:
            cluster_spec[CHIEF] = ['localhost:%s' % test_util.pick_unused_port()]
        if num_workers:
            cluster_spec[WORKER] = ['localhost:%s' % test_util.pick_unused_port() for _ in range(num_workers)]
        if num_ps:
            cluster_spec[PS] = ['localhost:%s' % test_util.pick_unused_port() for _ in range(num_ps)]
        if has_eval:
            cluster_spec[EVALUATOR] = ['localhost:%s' % test_util.pick_unused_port()]
        return cluster_spec

    def _in_graph_worker_fn(self, strategy):
        if False:
            i = 10
            return i + 15
        context = distribute_coordinator_context.get_current_worker_context()
        self.assertTrue(context is not None)
        with self._test_session(target=context.master_target) as sess:
            xs = []
            expected = 0.0
            for i in range(context.num_workers):
                with ops.device('/job:worker/task:%d' % i):
                    x = variable_scope.get_variable('x_%d' % i, initializer=10.0)
                    x_add = x.assign_add(float(i))
                    xs.append(x_add)
                    expected += i + 10.0
            with ops.device('/job:worker/task:0'):
                result = math_ops.add_n(xs)
            self.evaluate(variables.global_variables_initializer())
            result_value = sess.run(result)
        self.assertEqual(result_value, expected)
        if result_value == expected:
            self._result_correct += 1

    def _wrapped_worker_fn(self, worker_fn):
        if False:
            for i in range(10):
                print('nop')

        def wrapped(*args, **kwargs):
            if False:
                return 10
            with self._coord.stop_on_exception():
                return worker_fn(*args, **kwargs)
        return wrapped

    def _run_coordinator_in_thread(self, worker_fn, strategy, **kwargs):
        if False:
            while True:
                i = 10
        t = threading.Thread(target=distribute_coordinator.run_distribute_coordinator, args=(self._wrapped_worker_fn(worker_fn), strategy), kwargs=kwargs)
        t.start()
        return t

    def _run_multiple_coordinator_in_threads(self, worker_fn, strategy, cluster_spec, **kwargs):
        if False:
            return 10
        threads = {}
        for task_type in cluster_spec.keys():
            threads[task_type] = []
            for task_id in range(len(cluster_spec[task_type])):
                t = self._run_coordinator_in_thread(worker_fn, strategy, cluster_spec=cluster_spec, task_type=task_type, task_id=task_id, **kwargs)
                threads[task_type].append(t)
        return threads

    def _join_threads(self, threads):
        if False:
            i = 10
            return i + 15
        try:
            self._coord.join(threads)
        except errors.UnknownError as e:
            if 'Could not start gRPC server' in e.message:
                self.skipTest('Cannot start std servers.')
            else:
                raise

    def _between_graph_worker_fn(self, strategy):
        if False:
            return 10
        context = distribute_coordinator_context.get_current_worker_context()
        self.assertTrue(context is not None)
        with self._test_session(target=context.master_target) as sess:
            with ops.device('/job:ps/task:0'):
                x = variable_scope.get_variable('x', initializer=10.0, use_resource=True)
            with ops.device('/job:ps/task:1'):
                y = variable_scope.get_variable('y', initializer=20.0, use_resource=True)
            x_add = x.assign_add(2.0)
            y_sub = y.assign_sub(2.0)
            train_op = control_flow_ops.group([x_add, y_sub])
            if context.is_chief:
                self.evaluate(variables.global_variables_initializer())
            if context.has_barrier:
                context.wait_for_other_workers()
            else:
                while True:
                    uninit_vars = sess.run(variables.report_uninitialized_variables())
                    if len(uninit_vars) == 0:
                        break
            sess.run(train_op)
            if context.has_barrier:
                context.wait_for_other_workers()
            else:
                self._barrier.wait()
            (x_val, y_val) = sess.run([x, y])
            self.assertEqual(x_val, 16.0)
            self.assertEqual(y_val, 14.0)
            if x_val == 16.0 and y_val == 14.0:
                with self._lock:
                    self._result_correct += 1

    def _between_graph_with_monitored_session(self, strategy):
        if False:
            return 10
        context = distribute_coordinator_context.get_current_worker_context()
        self.assertTrue(context is not None)
        with ops.device('/job:ps/task:0'):
            x = variable_scope.get_variable('xx', initializer=10.0, use_resource=True)
        with ops.device('/job:ps/task:1'):
            y = variable_scope.get_variable('yy', initializer=20.0, use_resource=True)
        x_add = x.assign_add(2.0)
        y_sub = y.assign_sub(2.0)
        train_op = control_flow_ops.group([x_add, y_sub])
        with monitored_session.MonitoredSession() as sess:
            sess.run(train_op)
            if context.has_barrier:
                context.wait_for_other_workers()
            else:
                self._barrier.wait()
            (x_val, y_val) = sess.run([x, y])
        self.assertEqual(x_val, 16.0)
        self.assertEqual(y_val, 14.0)
        if x_val == 16.0 and y_val == 14.0:
            with self._lock:
                self._result_correct += 1

    def _dump_worker_context(self, strategy):
        if False:
            print('Hello World!')
        'Dumps the propoerties of each worker context.\n\n    It dumps the context properties to a dict mapping from task_type to a list\n    of tuples of master_target, num_workers, is_chief and distribute_mode, where\n    the list is indexed by the task_id.\n\n    Args:\n      strategy: a `DistributionStrategy` object.\n    '
        context = distribute_coordinator_context.get_current_worker_context()
        self.assertTrue(context is not None)
        task_type = str(context.task_type)
        task_id = context.task_id or 0
        with self._lock:
            if task_type not in self._worker_context:
                self._worker_context[task_type] = []
            while len(self._worker_context[task_type]) <= task_id:
                self._worker_context[task_type].append(None)
            self._worker_context[task_type][task_id] = (context.master_target, context.num_workers, context.is_chief, context.distributed_mode)

    def _dump_strategy_property(self, strategy):
        if False:
            i = 10
            return i + 15
        context = distribute_coordinator_context.get_current_worker_context()
        self.assertTrue(context is not None)
        self.assertEqual(context._strategy.extended.experimental_should_init, strategy.extended.experimental_should_init)
        self.assertEqual(context.should_checkpoint, strategy.extended.should_checkpoint)
        self.assertEqual(context.should_save_summary, strategy.extended.should_save_summary)
        task_type = str(context.task_type)
        task_id = context.task_id or 0
        with self._lock:
            if task_type not in self._strategy_property:
                self._strategy_property[task_type] = []
            while len(self._strategy_property[task_type]) <= task_id:
                self._strategy_property[task_type].append(None)
            self._strategy_property[task_type][task_id] = (context._strategy.extended.experimental_should_init, context.should_checkpoint, context.should_save_summary)

    def _run_mock_std_server(self, session_config=None, cluster_spec=None, task_type=None, task_id=None, rpc_layer=None, environment=None):
        if False:
            while True:
                i = 10
        task_type = str(task_type)
        task_id = task_id or 0
        with self._lock:
            if task_type not in self._std_servers:
                self._std_servers[task_type] = []
            while len(self._std_servers[task_type]) <= task_id:
                self._std_servers[task_type].append(None)
            server = MockServer()
            self._std_servers[task_type][task_id] = server
        return server

class DistributeCoordinatorTestStandaloneMode(DistributeCoordinatorTestBase):

    def testInGraphStandaloneMode(self):
        if False:
            i = 10
            return i + 15
        'Test it runs in-graph replication in standalone client mode.'
        distribute_coordinator.run_distribute_coordinator(self._in_graph_worker_fn, MockStrategy(between_graph=False), cluster_spec=self._cluster_spec)
        self.assertEqual(self._result_correct, 1)

    def testBetweenGraph(self):
        if False:
            for i in range(10):
                print('nop')
        'Test it runs between-graph replication in standalone client mode.'
        distribute_coordinator.run_distribute_coordinator(self._between_graph_worker_fn, MockStrategy(between_graph=True), cluster_spec=self._cluster_spec)
        self.assertEqual(self._result_correct, NUM_WORKERS)

    @test_util.run_v1_only('MonitoredSession removed from v2')
    def testBetweenGraphWithMonitoredSession(self):
        if False:
            return 10
        'Test monitored session in standalone client mode.'
        distribute_coordinator.run_distribute_coordinator(self._between_graph_with_monitored_session, MockStrategy(between_graph=True), cluster_spec=self._cluster_spec)
        self.assertEqual(self._result_correct, NUM_WORKERS)

    def testBetweenGraphContext(self):
        if False:
            for i in range(10):
                print('nop')
        distribute_coordinator.run_distribute_coordinator(self._dump_worker_context, MockStrategy(between_graph=True), cluster_spec=self._cluster_spec)
        self.assertEqual(len(self._worker_context), 1)
        self.assertTrue(WORKER in self._worker_context)
        self.assertEqual(len(self._worker_context[WORKER]), NUM_WORKERS)
        self.assertEqual(self._worker_context[WORKER][0], (_bytes_to_str(self._workers[0].target), NUM_WORKERS, True, True))
        self.assertEqual(self._worker_context[WORKER][1], (_bytes_to_str(self._workers[1].target), NUM_WORKERS, False, True))
        self.assertEqual(self._worker_context[WORKER][2], (_bytes_to_str(self._workers[2].target), NUM_WORKERS, False, True))

    def testBetweenGraphStrategyProperties(self):
        if False:
            return 10
        distribute_coordinator.run_distribute_coordinator(self._dump_strategy_property, MockStrategy(between_graph=True, should_init=True), cluster_spec=self._cluster_spec)
        self.assertEqual(len(self._strategy_property), 1)
        self.assertTrue(WORKER in self._strategy_property)
        self.assertEqual(len(self._strategy_property[WORKER]), NUM_WORKERS)
        self.assertEqual(self._strategy_property[WORKER][0], (True, True, True))
        self.assertEqual(self._strategy_property[WORKER][1], (True, False, False))
        self.assertEqual(self._strategy_property[WORKER][2], (True, False, False))

    def testInGraphContext(self):
        if False:
            return 10
        distribute_coordinator.run_distribute_coordinator(self._dump_worker_context, MockStrategy(between_graph=False), cluster_spec=self._cluster_spec)
        self.assertEqual(len(self._worker_context), 1)
        self.assertTrue('None' in self._worker_context)
        self.assertEqual(len(self._worker_context['None']), 1)
        self.assertEqual(self._worker_context['None'][0], (_bytes_to_str(self._workers[0].target), NUM_WORKERS, True, True))

    def testLocalContext(self):
        if False:
            print('Hello World!')
        distribute_coordinator.run_distribute_coordinator(self._dump_worker_context, MockStrategy(between_graph=False), cluster_spec=None)
        self.assertEqual(len(self._worker_context), 1)
        self.assertTrue('None' in self._worker_context)
        self.assertEqual(len(self._worker_context['None']), 1)
        self.assertEqual(self._worker_context['None'][0], ('', 0, True, False))

    def testBetweenGraphContextWithChief(self):
        if False:
            return 10
        cluster_spec = copy.deepcopy(self._cluster_spec)
        cluster_spec[CHIEF] = ['fake_chief']
        distribute_coordinator.run_distribute_coordinator(self._dump_worker_context, MockStrategy(between_graph=True), cluster_spec=cluster_spec, rpc_layer='grpc')
        self.assertEqual(len(self._worker_context), 2)
        self.assertTrue(CHIEF in self._worker_context)
        self.assertTrue(WORKER in self._worker_context)
        self.assertEqual(len(self._worker_context[CHIEF]), 1)
        self.assertEqual(len(self._worker_context[WORKER]), NUM_WORKERS)
        self.assertEqual(self._worker_context[CHIEF][0], ('grpc://fake_chief', 4, True, True))
        self.assertEqual(self._worker_context[WORKER][0], (_bytes_to_str(self._workers[0].target), NUM_WORKERS + 1, False, True))
        self.assertEqual(self._worker_context[WORKER][1], (_bytes_to_str(self._workers[1].target), NUM_WORKERS + 1, False, True))
        self.assertEqual(self._worker_context[WORKER][2], (_bytes_to_str(self._workers[2].target), NUM_WORKERS + 1, False, True))

    def testInGraphContextWithEval(self):
        if False:
            i = 10
            return i + 15
        cluster_spec = copy.deepcopy(self._cluster_spec)
        cluster_spec[EVALUATOR] = ['fake_evaluator']
        distribute_coordinator.run_distribute_coordinator(self._dump_worker_context, MockStrategy(between_graph=False), cluster_spec=cluster_spec, rpc_layer=None)
        self.assertEqual(len(self._worker_context), 2)
        self.assertTrue('None' in self._worker_context)
        self.assertTrue(EVALUATOR in self._worker_context)
        self.assertEqual(len(self._worker_context['None']), 1)
        self.assertEqual(len(self._worker_context[EVALUATOR]), 1)
        self.assertEqual(self._worker_context['None'][0], (_strip_protocol(_bytes_to_str(self._workers[0].target)), 3, True, True))
        self.assertEqual(self._worker_context[EVALUATOR][0], ('', 3, True, False))

class DistributeCoordinatorTestIndependentWorkerMode(DistributeCoordinatorTestBase):

    def testInGraph(self):
        if False:
            print('Hello World!')
        cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
        threads = self._run_multiple_coordinator_in_threads(self._in_graph_worker_fn, MockStrategy(between_graph=False), cluster_spec, mode=INDEPENDENT_WORKER)
        self._join_threads([threads[WORKER][0]])
        self.assertEqual(self._result_correct, 1)

    def testBetweenGraph(self):
        if False:
            while True:
                i = 10
        cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS, num_ps=NUM_PS)
        threads = self._run_multiple_coordinator_in_threads(self._between_graph_worker_fn, MockStrategy(between_graph=True), cluster_spec, mode=INDEPENDENT_WORKER)
        self._join_threads(threads[WORKER])
        self.assertEqual(self._result_correct, NUM_WORKERS)

    @test_util.run_v1_only('MonitoredSession removed from v2')
    def testBetweenGraphWithMonitoredSession(self):
        if False:
            print('Hello World!')
        cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS, num_ps=NUM_PS)
        threads = self._run_multiple_coordinator_in_threads(self._between_graph_with_monitored_session, MockStrategy(between_graph=True), cluster_spec, mode=INDEPENDENT_WORKER)
        self._join_threads(threads[WORKER])
        self.assertEqual(self._result_correct, NUM_WORKERS)

    def testBetweenGraphContext(self):
        if False:
            for i in range(10):
                print('nop')
        cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
        with test.mock.patch.object(distribute_coordinator, '_run_std_server', self._run_mock_std_server):
            threads = self._run_multiple_coordinator_in_threads(self._dump_worker_context, MockStrategy(between_graph=True), cluster_spec, mode=INDEPENDENT_WORKER, rpc_layer=None)
            self._join_threads(threads[WORKER])
        self.assertEqual(len(self._worker_context), 1)
        self.assertTrue(WORKER in self._worker_context)
        self.assertEqual(len(self._worker_context[WORKER]), NUM_WORKERS)
        self.assertEqual(self._worker_context[WORKER][0], (_bytes_to_str(cluster_spec[WORKER][0]), NUM_WORKERS, True, True))
        self.assertEqual(self._worker_context[WORKER][1], (_bytes_to_str(cluster_spec[WORKER][1]), NUM_WORKERS, False, True))
        self.assertEqual(self._worker_context[WORKER][2], (_bytes_to_str(cluster_spec[WORKER][2]), NUM_WORKERS, False, True))
        self.assertEqual(len(self._std_servers), 1)
        self.assertTrue(WORKER in self._std_servers)
        self.assertEqual(len(self._std_servers[WORKER]), 3)
        self.assertFalse(self._std_servers[WORKER][0].joined)
        self.assertFalse(self._std_servers[WORKER][1].joined)
        self.assertFalse(self._std_servers[WORKER][2].joined)

    def testBetweenGraphStrategyProperties(self):
        if False:
            for i in range(10):
                print('nop')
        cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
        with test.mock.patch.object(distribute_coordinator, '_run_std_server', self._run_mock_std_server):
            threads = self._run_multiple_coordinator_in_threads(self._dump_strategy_property, MockStrategy(between_graph=True, should_init=True), cluster_spec, mode=INDEPENDENT_WORKER, rpc_layer=None)
            self._join_threads(threads[WORKER])
        self.assertEqual(len(self._strategy_property), 1)
        self.assertTrue(WORKER in self._strategy_property)
        self.assertEqual(len(self._strategy_property[WORKER]), NUM_WORKERS)
        self.assertEqual(self._strategy_property[WORKER][0], (True, True, True))
        self.assertEqual(self._strategy_property[WORKER][1], (True, False, False))
        self.assertEqual(self._strategy_property[WORKER][2], (True, False, False))

    def testInGraphContext(self):
        if False:
            while True:
                i = 10
        cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
        with test.mock.patch.object(distribute_coordinator, '_run_std_server', self._run_mock_std_server):
            threads = self._run_multiple_coordinator_in_threads(self._dump_worker_context, MockStrategy(between_graph=False), cluster_spec, mode=INDEPENDENT_WORKER, rpc_layer=None)
            self._join_threads(threads[WORKER])
        self.assertEqual(len(self._worker_context), 1)
        self.assertTrue('None' in self._worker_context)
        self.assertEqual(len(self._worker_context['None']), 1)
        self.assertEqual(self._worker_context['None'][0], (_bytes_to_str(cluster_spec[WORKER][0]), NUM_WORKERS, True, True))
        self.assertEqual(len(self._std_servers), 1)
        self.assertTrue(WORKER in self._std_servers)
        self.assertEqual(len(self._std_servers[WORKER]), 3)
        self.assertFalse(self._std_servers[WORKER][0].joined)
        self.assertTrue(self._std_servers[WORKER][1].joined)
        self.assertTrue(self._std_servers[WORKER][2].joined)

    def testInGraphContextWithEval(self):
        if False:
            for i in range(10):
                print('nop')
        cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS, has_eval=True)
        with test.mock.patch.object(distribute_coordinator, '_run_std_server', self._run_mock_std_server):
            threads = self._run_multiple_coordinator_in_threads(self._dump_worker_context, MockStrategy(between_graph=False), cluster_spec, mode=INDEPENDENT_WORKER, rpc_layer=None)
            self._join_threads(threads[WORKER])
            self._join_threads([threads[EVALUATOR][0]])
        self.assertEqual(len(self._worker_context), 2)
        self.assertTrue('None' in self._worker_context)
        self.assertTrue(EVALUATOR in self._worker_context)
        self.assertEqual(len(self._worker_context['None']), 1)
        self.assertEqual(len(self._worker_context[EVALUATOR]), 1)
        self.assertEqual(self._worker_context['None'][0], (_bytes_to_str(cluster_spec[WORKER][0]), 3, True, True))
        self.assertEqual(self._worker_context[EVALUATOR][0], ('', 3, True, False))
        self.assertEqual(len(self._std_servers), 1)
        self.assertTrue(WORKER in self._std_servers)
        self.assertEqual(len(self._std_servers[WORKER]), 3)
        self.assertFalse(self._std_servers[WORKER][0].joined)
        self.assertTrue(self._std_servers[WORKER][1].joined)
        self.assertTrue(self._std_servers[WORKER][2].joined)

    def testRunStdServerInGoogleEnvironment(self):
        if False:
            for i in range(10):
                print('nop')
        cluster_spec = {'worker': ['fake_worker'], 'ps': ['localhost:0']}
        tf_config = {'cluster': cluster_spec, 'environment': 'google'}
        joined = [False]

        def _fake_sleep(_):
            if False:
                for i in range(10):
                    print('nop')
            joined[0] = True
            original_sys_exit(0)

        def _thread_fn(cluster_spec):
            if False:
                print('Hello World!')
            distribute_coordinator.run_distribute_coordinator(None, MockStrategy(between_graph=True), mode=INDEPENDENT_WORKER, cluster_spec=cluster_spec, task_type='ps', task_id=0)
        with test.mock.patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}), test.mock.patch.object(time, 'sleep', _fake_sleep):
            t = threading.Thread(target=_thread_fn, args=(cluster_spec,))
            t.start()
            t.join()
        self.assertTrue(joined[0])

    def testRpcLayerEnvironmentVariable(self):
        if False:
            i = 10
            return i + 15
        cluster_spec = {'worker': ['fake_worker'], 'ps': ['fake_ps']}
        tf_config = {'cluster': cluster_spec, 'rpc_layer': 'cake'}
        rpc_layer_from_coordinator = [None]

        def _run_mock_server(cluster_spec=None, task_type=None, task_id=None, session_config=None, rpc_layer=None, environment=None):
            if False:
                i = 10
                return i + 15
            del cluster_spec, task_type, task_id, session_config, environment
            rpc_layer_from_coordinator[0] = rpc_layer
            return MockServer()
        with test.mock.patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}), test.mock.patch.object(distribute_coordinator, '_run_std_server', _run_mock_server):
            distribute_coordinator.run_distribute_coordinator(None, MockStrategy(between_graph=True), mode=INDEPENDENT_WORKER, cluster_spec=cluster_spec, task_type='ps', task_id=0)
        self.assertEqual(rpc_layer_from_coordinator[0], 'cake')

class StrategyConfigureTest(test.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._device_filters = []
        self._intra_op_parallelism_threads = None
        self._inter_op_parallelism_threads = None
        super(StrategyConfigureTest, self).setUp()

    def _dump_device_filters(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        session_config = kwargs.get('session_config', None)
        self._device_filters.extend(session_config.device_filters)
        self._intra_op_parallelism_threads = session_config.intra_op_parallelism_threads
        self._inter_op_parallelism_threads = session_config.inter_op_parallelism_threads
        return MockServer()

    def _worker_fn(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        worker_context = distribute_coordinator_context.get_current_worker_context()
        session_config = worker_context._session_config
        self._device_filters.extend(session_config.device_filters)
        self._intra_op_parallelism_threads = session_config.intra_op_parallelism_threads
        self._inter_op_parallelism_threads = session_config.inter_op_parallelism_threads
        return MockServer()

    def test_session_config_in_std_server(self):
        if False:
            return 10
        cluster_spec = {'worker': ['fake_worker'], 'ps': ['fake_ps']}
        tf_config = {'cluster': cluster_spec}
        with test.mock.patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}), test.mock.patch.object(distribute_coordinator, '_run_std_server', self._dump_device_filters):
            distribute_coordinator.run_distribute_coordinator(lambda _: None, MockStrategy(between_graph=True), mode=INDEPENDENT_WORKER, cluster_spec=cluster_spec, task_type='worker', task_id=0)
        self.assertEqual(self._intra_op_parallelism_threads, 1)
        self.assertEqual(self._inter_op_parallelism_threads, 0)

    def test_session_config_in_session_creator(self):
        if False:
            for i in range(10):
                print('nop')
        cluster_spec = {'worker': ['localhost:0']}
        tf_config = {'cluster': cluster_spec}
        distribute_coordinator._thread_local = threading.local()
        with test.mock.patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}):
            distribute_coordinator.run_distribute_coordinator(self._worker_fn, MockStrategy(between_graph=True), mode=INDEPENDENT_WORKER, cluster_spec=cluster_spec, task_type='worker', task_id=0)
        self.assertEqual(self._device_filters, ['/job:worker/task:0', '/job:ps'])
        self.assertEqual(self._intra_op_parallelism_threads, 2)
        self.assertEqual(self._inter_op_parallelism_threads, 0)

    def test_eval_strategy_configure(self):
        if False:
            for i in range(10):
                print('nop')
        cluster_spec = {'evaluator': ['localhost:0']}
        tf_config = {'cluster': cluster_spec}
        with test.mock.patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}):
            distribute_coordinator.run_distribute_coordinator(lambda _: None, MockStrategy(between_graph=False), eval_fn=self._worker_fn, eval_strategy=MockStrategy(between_graph=True), mode=INDEPENDENT_WORKER, cluster_spec=cluster_spec, task_type='evaluator', task_id=0)
        self.assertEqual(self._device_filters, ['/job:somejob'])
        self.assertEqual(self._intra_op_parallelism_threads, 0)
        self.assertEqual(self._inter_op_parallelism_threads, 2)

class RunStandardTensorflowServerTest(test.TestCase):

    def test_std_server_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        cs = {'worker': ['fake_worker'], 'ps': ['fake_ps']}
        tf_config = {'cluster': cs, 'task': {'type': 'ps', 'id': 0}}

        def _mock_run_std_server(cluster_spec=None, task_type=None, task_id=None, session_config=None, rpc_layer=None):
            if False:
                print('Hello World!')
            self.assertEqual(cluster_spec.as_dict(), cs)
            self.assertEqual(task_type, 'ps')
            self.assertEqual(task_id, 0)
            self.assertEqual(session_config.experimental.collective_group_leader, '/job:worker/replica:0/task:0')
            self.assertEqual(session_config.intra_op_parallelism_threads, 1)
            self.assertEqual(rpc_layer, 'grpc')
            return MockServer()
        with test.mock.patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}), test.mock.patch.object(distribute_coordinator, '_run_std_server', _mock_run_std_server):
            session_config = config_pb2.ConfigProto()
            session_config.intra_op_parallelism_threads = 1
            mock_server = distribute_coordinator.run_standard_tensorflow_server(session_config)
            self.assertTrue(mock_server.started)
if __name__ == '__main__':
    with test.mock.patch.object(sys, 'exit', os._exit):
        orig_init = session_manager.SessionManager.__init__

        def new_init(*args, **kwargs):
            if False:
                while True:
                    i = 10
            kwargs.pop('recovery_wait_secs', None)
            kwargs['recovery_wait_secs'] = 0.5
            orig_init(*args, **kwargs)
        session_manager.SessionManager.__init__ = new_init
        test.main()