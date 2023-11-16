"""This file contains tests that simulate peer failures.

When a peer fails during MultiWorkerMirroredStrategy training. All workers
should get Unavailable error.
"""
import os
import tensorflow as tf
from tensorflow.python.distribute import collective_all_reduce_strategy as mwms_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import test
RPC_PROTOCOL = 'grpc'
mwms_lib.CollectiveAllReduceExtended._enable_check_health = True
mwms_lib.CollectiveAllReduceExtended._check_health_interval = 3
mwms_lib.CollectiveAllReduceExtended._check_health_initial_timeout = 0
mwms_lib.CollectiveAllReduceExtended._check_health_timeout = 1

def get_attempt(strategy, attempts):
    if False:
        return 10
    task_type = strategy.cluster_resolver.task_type
    task_id = strategy.cluster_resolver.task_id
    attempts[task_type, task_id] = attempts.get((task_type, task_id), 0) + 1
    return (task_id, attempts[task_type, task_id])
quick_exit = os._exit

class PeerFailureTest(test.TestCase):

    def test_creating_variable(self):
        if False:
            i = 10
            return i + 15

        def worker_fn():
            if False:
                return 10
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
            with strategy.scope():
                tf.Variable(1.0)
                if strategy.cluster_resolver.task_id == 1:
                    quick_exit(1)
                v = tf.Variable(tf.random.uniform(()))
                return v.read_value().numpy()
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec, rpc_layer=RPC_PROTOCOL)
        mpr.start()
        with self.assertRaises((tf.errors.UnavailableError, tf.errors.DeadlineExceededError)):
            mpr.join(timeout=60)

    def test_reduce_small_tensor(self):
        if False:
            print('Hello World!')

        def worker_fn():
            if False:
                return 10
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
            value = tf.identity([1.0])
            strategy.reduce('sum', value, axis=None)
            if strategy.cluster_resolver.task_id == 1:
                quick_exit(1)
            strategy.reduce('sum', value, axis=None)
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec, rpc_layer=RPC_PROTOCOL)
        mpr.start()
        with self.assertRaises((tf.errors.UnavailableError, tf.errors.DeadlineExceededError)):
            mpr.join(timeout=60)

class PeerFailureRecoverTest(test.TestCase):

    def test_creating_variable(self):
        if False:
            return 10

        def worker_fn(attempts):
            if False:
                print('Hello World!')
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
            (task_id, attempt) = get_attempt(strategy, attempts)
            with strategy.scope():
                tf.Variable(1.0)
                if attempt == 1 and task_id == 1:
                    quick_exit(1)
                v = tf.Variable(tf.random.uniform(()))
                return v.read_value().numpy()
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        attempts = multi_process_runner.manager().dict()
        mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec, rpc_layer=RPC_PROTOCOL, args=(attempts,), auto_restart=True)
        mpr.start()
        results = mpr.join(timeout=90).return_value
        self.assertEqual(results[0], results[1])

    def test_reduce_small_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def worker_fn(attempts):
            if False:
                print('Hello World!')
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
            (task_id, attempt) = get_attempt(strategy, attempts)
            value = tf.identity([1.0])
            strategy.reduce('sum', value, axis=None)
            if attempt == 1 and task_id == 1:
                quick_exit(1)
            return strategy.reduce('sum', value, axis=None).numpy()
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        attempts = multi_process_runner.manager().dict()
        mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec, rpc_layer=RPC_PROTOCOL, args=(attempts,), auto_restart=True)
        mpr.start()
        results = mpr.join(timeout=90).return_value
        self.assertAllEqual(results, [[2.0], [2.0]])

    def test_quick_recover(self):
        if False:
            for i in range(10):
                print('nop')

        def worker_fn(attempts):
            if False:
                while True:
                    i = 10
            mwms_lib.CollectiveAllReduceExtended._check_alive_interval = 30
            mwms_lib.CollectiveAllReduceExtended._check_alive_initial_timeout = 30
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
            (task_id, attempt) = get_attempt(strategy, attempts)

            @tf.function
            def replica_fn():
                if False:
                    i = 10
                    return i + 15
                ctx = tf.distribute.get_replica_context()
                value = tf.ones((64, 64))
                ctx.all_reduce(tf.distribute.ReduceOp.SUM, [value, value])
            strategy.run(replica_fn)
            if attempt == 1 and task_id == 1:
                quick_exit(1)
            strategy.run(replica_fn)
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        attempts = multi_process_runner.manager().dict()
        mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec, rpc_layer=RPC_PROTOCOL, args=(attempts,), auto_restart=True)
        mpr.start()
        mpr.join(timeout=90)
if __name__ == '__main__':
    test_util.main()