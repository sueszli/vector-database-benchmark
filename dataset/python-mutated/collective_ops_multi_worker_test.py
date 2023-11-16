"""Tests for multi worker Collective Operations."""
import copy
import os
import threading
import time
from absl.testing import parameterized
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import collective_ops

def enable_collective_ops(cluster_resolver):
    if False:
        for i in range(10):
            print('nop')
    context.context().configure_collective_ops(collective_leader='/job:worker/replica:0/task:0')
    config_proto = copy.deepcopy(context.context().config)
    server_def = tensorflow_server_pb2.ServerDef(cluster=cluster_resolver.cluster_spec().as_cluster_def(), default_session_config=config_proto, job_name=cluster_resolver.task_type, task_index=cluster_resolver.task_id, protocol=cluster_resolver.rpc_layer or 'grpc')
    context.context().enable_collective_ops(server_def)

def enable_collective_ops_with_barrier(cluster_resolver):
    if False:
        while True:
            i = 10
    multi_process_runner.get_barrier().wait()
    enable_collective_ops(cluster_resolver)
    multi_process_runner.get_barrier().wait()
device_combination = combinations.combine(device='CPU', communication='RING', required_gpus=0) + combinations.combine(device='GPU', communication=['RING', 'NCCL'], required_gpus=1)

class CollectiveOpTest(test.TestCase):

    def testCheckHealth(self):
        if False:
            i = 10
            return i + 15

        def worker_fn():
            if False:
                i = 10
                return i + 15
            enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver())
            while True:
                try:
                    for task in ['/job:worker/replica:0/task:0', '/job:worker/replica:0/task:1']:
                        context.context().check_collective_ops_peer_health(task, timeout_in_ms=1000)
                except (errors.UnavailableError, errors.DeadlineExceededError):
                    continue
                break
            multi_process_runner.get_barrier().wait()
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec)
        mpr.start()
        mpr.join()

    def testCheckHealthPeerDown(self):
        if False:
            print('Hello World!')

        def worker_fn():
            if False:
                for i in range(10):
                    print('nop')
            enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver())
            context.context().check_collective_ops_peer_health('/job:worker/replica:0/task:1', timeout_in_ms=1000)
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec)
        mpr.start_single_process('worker', 0)
        with self.assertRaises((errors.UnavailableError, errors.DeadlineExceededError)):
            mpr.join()

    def testCheckHealthPeerRestart(self):
        if False:
            print('Hello World!')

        def worker_fn():
            if False:
                while True:
                    i = 10
            cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
            enable_collective_ops(cluster_resolver)
            collective_ops.all_reduce(constant_op.constant(1.0), group_size=2, group_key=100, instance_key=100, merge_op='Add', final_op='Id', communication_hint='ring')
            if cluster_resolver.task_type == 'worker':
                os._exit(1)
            else:
                while True:
                    time.sleep(1)
                    try:
                        context.context().check_collective_ops_peer_health('/job:worker/replica:0/task:0', timeout_in_ms=1000)
                    except errors.UnavailableError:
                        pass
                    except errors.FailedPreconditionError:
                        break
        cluster_spec = multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=1)
        mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec, auto_restart=True)
        mpr.start()
        mpr.join()

    def testCheckHealthInvalidPeer(self):
        if False:
            print('Hello World!')

        def worker_fn():
            if False:
                print('Hello World!')
            enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver())
            context.context().check_collective_ops_peer_health('localhost:12345', timeout_in_ms=1000)
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec)
        mpr.start_single_process('worker', 0)
        with self.assertRaises(errors.InvalidArgumentError):
            mpr.join()
two_worker_pool_runner = multi_process_runner.MultiProcessPoolRunner(multi_worker_test_base.create_cluster_spec(num_workers=2), initializer=lambda : enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver()))

@combinations.generate(combinations.times(combinations.combine(mode='eager', num_workers=2, runner=two_worker_pool_runner), device_combination))
class AbortCollectiveOpsTest(test.TestCase, parameterized.TestCase):

    def testAbortCommunication(self, device, communication):
        if False:
            while True:
                i = 10
        if communication == 'NCCL':
            self.skipTest('b/171358086: cannot test multi worker NCCL')
        dev0 = '/device:%s:0' % device
        cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
        enable_collective_ops_with_barrier(cluster_resolver)
        group_size = 2
        group_key = 100
        instance_key = 100
        in_tensor = constant_op.constant([1.0])
        with ops.device(dev0):
            collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key, communication_hint=communication)
        if cluster_resolver.task_id == 1:

            def abort_fn():
                if False:
                    for i in range(10):
                        print('nop')
                time.sleep(2)
                context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')
            t = threading.Thread(target=abort_fn)
            t.start()
            with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
                with ops.device(dev0):
                    collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key, communication_hint=communication)
            with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
                with ops.device(dev0):
                    collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key, communication_hint=communication)
            t.join()
        enable_collective_ops_with_barrier(cluster_resolver)
        with ops.device(dev0):
            collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key, communication_hint=communication)

    def testAbortGroupParamsResolution(self, device, communication):
        if False:
            for i in range(10):
                print('nop')
        if communication == 'NCCL':
            self.skipTest('b/171358086: cannot test multi worker NCCL')
        dev0 = '/device:%s:0' % device
        cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
        enable_collective_ops_with_barrier(cluster_resolver)
        group_size = 2
        group_key = 100
        instance_key = 100
        in_tensor = constant_op.constant([1.0])
        if cluster_resolver.task_id == 1:

            def abort_fn():
                if False:
                    print('Hello World!')
                time.sleep(2)
                context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')
            t = threading.Thread(target=abort_fn)
            t.start()
            with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
                with ops.device(dev0):
                    collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key)
            with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
                with ops.device(dev0):
                    collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key)
            t.join()
        enable_collective_ops_with_barrier(cluster_resolver)
        with ops.device(dev0):
            collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key)

    def testAbortInstanceParamsResolution(self, device, communication):
        if False:
            for i in range(10):
                print('nop')
        if communication == 'NCCL':
            self.skipTest('b/171358086: cannot test multi worker NCCL')
        dev0 = '/device:%s:0' % device
        cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
        enable_collective_ops_with_barrier(cluster_resolver)
        group_size = 2
        group_key = 100
        instance_key = 100
        in_tensor = constant_op.constant([1.0])
        with ops.device(dev0):
            collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key)
        if cluster_resolver.task_id == 1:

            def abort_fn():
                if False:
                    print('Hello World!')
                time.sleep(2)
                context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')
            t = threading.Thread(target=abort_fn)
            t.start()
            instance_key = 101
            with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
                with ops.device(dev0):
                    collective_ops.broadcast_send(in_tensor, (1,), dtypes.float32, group_size, group_key, instance_key)
            with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
                with ops.device(dev0):
                    collective_ops.broadcast_send(in_tensor, (1,), dtypes.float32, group_size, group_key, instance_key)
            t.join()
        enable_collective_ops_with_barrier(cluster_resolver)
        instance_key = 100
        with ops.device(dev0):
            if cluster_resolver.task_id == 0:
                collective_ops.broadcast_send(in_tensor, (1,), dtypes.float32, group_size, group_key, instance_key)
            else:
                collective_ops.broadcast_recv((1,), dtypes.float32, group_size, group_key, instance_key)
if __name__ == '__main__':
    multi_process_runner.test_main()