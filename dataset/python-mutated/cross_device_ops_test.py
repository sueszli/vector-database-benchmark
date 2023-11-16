"""Tests for CrossDeviceOps."""
import collections
import os
import threading
import time
from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
try:
    import dill
    _REGISTER_DECORATOR = dill.register
except ImportError:
    _REGISTER_DECORATOR = lambda fn, *_: fn
CollectiveReplicaLauncher = cross_device_utils.CollectiveReplicaLauncher
CommunicationImplementation = collective_util.CommunicationImplementation
ReduceOp = reduce_util.ReduceOp
IndexedSlicesValue = indexed_slices.IndexedSlicesValue
IndexedSlices = indexed_slices.IndexedSlices

def make_per_replica_value(value, devices):
    if False:
        for i in range(10):
            print('nop')
    'Creates a `PerReplica` object whose values reside in `devices`.\n\n  Args:\n    value: a tensor-convertible value or a `IndexedSlicesValue`, or a callable\n      that takes one argument (`device_idx`) and should return the value that is\n      going to be created on devices[device_idx].\n    devices: a list of device strings to create `PerReplica` values on.\n\n  Returns:\n    A `PerReplica` object.\n  '
    values = []
    for (device_idx, device) in enumerate(devices):
        if callable(value):
            v = value(device_idx)
        elif isinstance(value, list):
            v = value[device_idx]
        else:
            v = value
        if isinstance(v, IndexedSlicesValue):
            with ops.device(device):
                values.append(IndexedSlices(values=array_ops.identity(v.values), indices=array_ops.identity(v.indices), dense_shape=array_ops.identity(v.dense_shape)))
        else:
            with ops.device(device):
                values.append(array_ops.identity(v))
    return value_lib.PerReplica(values)

def enable_collective_ops():
    if False:
        i = 10
        return i + 15
    'Enable collectives in the current process.'
    cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
    context.context().configure_collective_ops(collective_leader="'/job:worker/replica:0/task:0'")
    config_proto = config_pb2.ConfigProto()
    config_proto.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
    server_def = tensorflow_server_pb2.ServerDef(cluster=cluster_resolver.cluster_spec().as_cluster_def(), default_session_config=config_proto, job_name=cluster_resolver.task_type, task_index=cluster_resolver.task_id, protocol=cluster_resolver.rpc_layer)
    context.context().enable_collective_ops(server_def)
    CollectiveReplicaLauncher._prefer_unique_instance_key = True
    CollectiveReplicaLauncher._prefer_ordering_token = False

class MultiProcessPoolRunner:

    def __init__(self, num_processes):
        if False:
            i = 10
            return i + 15
        cluster_spec_dict = multi_worker_test_base.create_cluster_spec(num_workers=num_processes)
        self.runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec_dict)
global_mpr_2p = MultiProcessPoolRunner(num_processes=2)
global_mpr_1p = MultiProcessPoolRunner(num_processes=1)

def get_global_mpr(num_processes):
    if False:
        print('Hello World!')
    if num_processes == 1:
        return global_mpr_1p.runner
    elif num_processes == 2:
        return global_mpr_2p.runner
    else:
        raise ValueError('get_global_mpr: num_processes must be 1 or 2, got %d' % num_processes)

class CollectiveOpsTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        global_mpr_1p.runner.run(enable_collective_ops)
        global_mpr_2p.runner.run(enable_collective_ops)

    def make_collective(self, num_processes, gpu_per_process):
        if False:
            print('Hello World!')
        'Returns collectives and other info to be used in tests.\n\n    Args:\n      num_processes: an integer indicating the number of processes that\n        participate in the collective.\n      gpu_per_process: number of GPUs (0 if no GPUs) used by each process.\n\n    Returns:\n     A tuple of (collective, devices, pid) where collective is a instance\n     of `CollectiveAllReduce`, devices are a list of local devices (str)\n     attached to the current process, and pid is the id of this process among\n     all participant processes.\n    '
        cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
        devices = ['/job:worker/replica:0/task:%d/device:CPU:0' % cluster_resolver.task_id]
        if gpu_per_process > 0:
            devices = ['/job:worker/replica:0/task:%d/device:GPU:%d' % (cluster_resolver.task_id, i) for i in range(gpu_per_process)]
        group_size = num_processes * len(devices)
        collective = cross_device_ops_lib.CollectiveAllReduce(devices=devices, group_size=group_size, options=collective_util.Options())
        return (collective, devices, cluster_resolver.task_id)

    def as_list(self, value):
        if False:
            return 10
        'An utility to convert a `Mirrored`, `Tensor` or `IndexedSlices` to a list.\n\n    The reason it exists is to provide a uniformed view of returned value of\n    "reduce" calls, especially across tf.function boundaries. Returning\n    `Mirrored` from a tf.function will only evaluate the primary value, which\n    makes collective ops of non-primary device being pruned, and will eventually\n    cause hanging.\n\n    Args:\n      value: the value to convert, can be one of `Mirrored`, `Tensor` and\n        `IndexedSlices`.\n\n    Returns:\n      A list of `Tensor` or `IndexedSlices`.\n    '
        if isinstance(value, tensor_lib.Tensor):
            return [value]
        elif isinstance(value, IndexedSlices):
            return [value]
        elif isinstance(value, value_lib.Mirrored):
            return value.values
        else:
            raise ValueError('unwrap: unsupported input type: %s' % type(value))
    RunOptions = collections.namedtuple('RunOptions', ['mode', 'num_processes', 'gpus_per_process', 'reduce_op', 'communication_options', 'prefer_unique_instance_key'])
    RunOptions.__new__.__defaults__ = (['eager', 'func_graph'], 2, 0, ReduceOp.SUM, collective_util.Options(), True)

    def reduce_and_verify(self, inputs, expect, options):
        if False:
            i = 10
            return i + 15
        'Reduce the given `inputs` and verify the output matches `expect`.\n\n    Args:\n      inputs: a list of `Tensor` or `IndexedSlices`, where i-th value will be\n        fed to i-th replica.\n      expect: a `Tensor` or `IndexedSlices`. This should be the expected value\n        for one replica.\n      options: a `RunOpotions` instance.\n    '

        def replica_fn():
            if False:
                print('Hello World!')
            CollectiveReplicaLauncher._prefer_unique_instance_key = options.prefer_unique_instance_key
            (collective, devices, pid) = self.make_collective(options.num_processes, options.gpus_per_process)

            def reduce_fn():
                if False:
                    print('Hello World!')
                value_fn = lambda device_idx: inputs[pid * len(devices) + device_idx]
                per_replica_value = make_per_replica_value(value_fn, devices)
                reduced_values = collective.reduce(options.reduce_op, per_replica_value, per_replica_value, options.communication_options)
                if options.gpus_per_process > 1:
                    self.assertIsInstance(reduced_values, value_lib.Mirrored)
                reduced_values = self.as_list(reduced_values)
                self.assertAllEqual(devices, [v.device for v in reduced_values])
                return [ops.convert_to_tensor(v) for v in reduced_values]
            per_replica_expect = [ops.convert_to_tensor(expect)] * len(devices)
            if 'eager' in options.mode:
                got = reduce_fn()
                self.assertAllClose(got, per_replica_expect)
            if 'func_graph' in options.mode:
                got = def_function.function(reduce_fn)()
                self.assertAllClose(got, per_replica_expect)
        get_global_mpr(options.num_processes).run(replica_fn)

    def batch_reduce_and_verify(self, inputs, expect, options):
        if False:
            while True:
                i = 10
        'Batch reduce the given `inputs` and verify the output matches `expect`.\n\n    Args:\n      inputs: a 2-level nested list of `Tensor` or `IndexedSlices`, where i-th\n        value will be fed to i-th replica.\n      expect: a list of `Tensor` or `IndexedSlices`. This should be the expected\n        value for one replica.\n      options: a `RunOpotions` instance.\n    '

        def replica_fn():
            if False:
                for i in range(10):
                    print('nop')
            CollectiveReplicaLauncher._prefer_unique_instance_key = options.prefer_unique_instance_key
            (collective, devices, pid) = self.make_collective(options.num_processes, options.gpus_per_process)

            def batch_reduce_fn():
                if False:
                    while True:
                        i = 10
                batch_size = len(inputs[0])
                value_dst_pairs = []
                for i in range(batch_size):

                    def value_fn(device_idx, idx=i):
                        if False:
                            while True:
                                i = 10
                        return inputs[pid * len(devices) + device_idx][idx]
                    per_replica_value = make_per_replica_value(value_fn, devices)
                    value_dst_pairs.append((per_replica_value, per_replica_value))
                reduced_values = collective.batch_reduce(options.reduce_op, value_dst_pairs, options.communication_options)
                if options.gpus_per_process > 1:
                    for v in reduced_values:
                        self.assertIsInstance(v, value_lib.Mirrored)
                reduced_values = [self.as_list(v) for v in reduced_values]
                for v in reduced_values:
                    self.assertAllEqual(devices, [t.device for t in v])
                return nest.map_structure(ops.convert_to_tensor, reduced_values)
            per_replica_expect = nest.map_structure(lambda x: [ops.convert_to_tensor(x)] * len(devices), expect)
            if 'eager' in options.mode:
                got = batch_reduce_fn()
                self.assertAllClose(got, per_replica_expect)
            if 'func_graph' in options.mode:
                got = def_function.function(batch_reduce_fn)()
                self.assertAllClose(got, per_replica_expect)
        get_global_mpr(options.num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=[1, 2], required_gpus=[0, 1, 2], implementation=[CommunicationImplementation.AUTO, CommunicationImplementation.RING, CommunicationImplementation.NCCL], reduce_op=[ReduceOp.SUM, ReduceOp.MEAN], prefer_unique_instance_key=[True, False]))
    def testReduceDense(self, num_processes, required_gpus, implementation, reduce_op, prefer_unique_instance_key):
        if False:
            while True:
                i = 10
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        options = self.RunOptions(num_processes=num_processes, gpus_per_process=required_gpus, reduce_op=reduce_op, communication_options=collective_util.Options(implementation=implementation), prefer_unique_instance_key=prefer_unique_instance_key)
        group_size = options.num_processes * (options.gpus_per_process or 1)
        inputs_data = [1.0, 2.0, 3.0, 4.0]
        inputs = inputs_data[0:group_size]
        if group_size == 1:
            expect = 1.0
        if group_size == 2:
            expect = 3.0 if reduce_op == ReduceOp.SUM else 1.5
        elif group_size == 4:
            expect = 10.0 if reduce_op == ReduceOp.SUM else 2.5
        self.reduce_and_verify(inputs, expect, options)

    @combinations.generate(combinations.combine(num_processes=[1, 2], required_gpus=[0, 1, 2], implementation=[CommunicationImplementation.AUTO, CommunicationImplementation.RING, CommunicationImplementation.NCCL], reduce_op=ReduceOp.SUM, prefer_unique_instance_key=[True, False]))
    def testReduceSparse(self, num_processes, required_gpus, implementation, reduce_op, prefer_unique_instance_key):
        if False:
            i = 10
            return i + 15
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        options = self.RunOptions(mode=['func_graph'], num_processes=num_processes, gpus_per_process=required_gpus, reduce_op=reduce_op, communication_options=collective_util.Options(implementation=implementation), prefer_unique_instance_key=prefer_unique_instance_key)
        group_size = options.num_processes * (options.gpus_per_process or 1)
        inputs_data = [IndexedSlicesValue(values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1]), IndexedSlicesValue(values=[[3.0], [4.0]], indices=[1, 2], dense_shape=[10, 1]), IndexedSlicesValue(values=[[5.0], [6.0]], indices=[7, 8], dense_shape=[10, 1]), IndexedSlicesValue(values=[[7.0], [8.0]], indices=[3, 2], dense_shape=[10, 1])]
        inputs = inputs_data[0:group_size]
        if group_size == 1:
            expect = IndexedSlices(values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1])
        elif group_size == 2:
            expect = IndexedSlices(values=[[1.0], [2.0], [3.0], [4.0]], indices=[0, 1, 1, 2], dense_shape=[10, 1])
        elif group_size == 4:
            expect = IndexedSlices(values=[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]], indices=[0, 1, 1, 2, 7, 8, 3, 2], dense_shape=[10, 1])
        self.reduce_and_verify(inputs, expect, options)

    @combinations.generate(combinations.combine(prefer_unique_instance_key=[True, False]))
    def testReduceSparseVariableLength(self, prefer_unique_instance_key):
        if False:
            while True:
                i = 10
        inputs = [IndexedSlicesValue(values=[[1.0]], indices=[0], dense_shape=[10, 1]), IndexedSlicesValue(values=[[2.0], [3.0], [4.0]], indices=[0, 1, 2], dense_shape=[10, 1])]
        expect = IndexedSlices(values=[[1.0], [2.0], [3.0], [4.0]], indices=[0, 0, 1, 2], dense_shape=[10, 1])
        self.reduce_and_verify(inputs, expect, self.RunOptions(mode=['func_graph'], num_processes=2, reduce_op=ReduceOp.SUM, prefer_unique_instance_key=prefer_unique_instance_key))

    @combinations.generate(combinations.combine(num_processes=[1, 2], required_gpus=[0, 1, 2], implementation=[CommunicationImplementation.AUTO, CommunicationImplementation.RING, CommunicationImplementation.NCCL], reduce_op=[ReduceOp.SUM, ReduceOp.MEAN], prefer_unique_instance_key=[True, False]))
    def testBatchReduceDense(self, num_processes, required_gpus, implementation, reduce_op, prefer_unique_instance_key):
        if False:
            for i in range(10):
                print('nop')
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        options = self.RunOptions(num_processes=num_processes, gpus_per_process=required_gpus, reduce_op=reduce_op, communication_options=collective_util.Options(implementation=implementation), prefer_unique_instance_key=prefer_unique_instance_key)
        group_size = options.num_processes * (options.gpus_per_process or 1)
        inputs_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        inputs = inputs_data[0:group_size]
        if group_size == 1:
            expect = [1.0, 2.0]
        if group_size == 2:
            expect = [4.0, 6.0] if reduce_op == ReduceOp.SUM else [2.0, 3.0]
        elif group_size == 4:
            expect = [16.0, 20.0] if reduce_op == ReduceOp.SUM else [4.0, 5.0]
        self.batch_reduce_and_verify(inputs, expect, options)

    @combinations.generate(combinations.combine(num_processes=[1, 2], required_gpus=[0, 1, 2], implementation=[CommunicationImplementation.AUTO, CommunicationImplementation.RING, CommunicationImplementation.NCCL], reduce_op=ReduceOp.SUM, prefer_unique_instance_key=[True, False]))
    def testBatchReduceSparse(self, num_processes, required_gpus, implementation, reduce_op, prefer_unique_instance_key):
        if False:
            while True:
                i = 10
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        options = self.RunOptions(mode=['func_graph'], num_processes=num_processes, gpus_per_process=required_gpus, reduce_op=reduce_op, communication_options=collective_util.Options(implementation=implementation), prefer_unique_instance_key=prefer_unique_instance_key)
        group_size = options.num_processes * (options.gpus_per_process or 1)
        inputs_data = ([IndexedSlicesValue(values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1]), IndexedSlicesValue(values=[[3.0], [4.0]], indices=[1, 2], dense_shape=[5, 1])], [IndexedSlicesValue(values=[[5.0], [6.0]], indices=[1, 2], dense_shape=[10, 1]), IndexedSlicesValue(values=[[7.0], [8.0]], indices=[0, 1], dense_shape=[5, 1])], [IndexedSlicesValue(values=[[9.0], [10.0]], indices=[3, 4], dense_shape=[10, 1]), IndexedSlicesValue(values=[[11.0], [12.0]], indices=[3, 4], dense_shape=[5, 1])], [IndexedSlicesValue(values=[[13.0], [14.0]], indices=[8, 9], dense_shape=[10, 1]), IndexedSlicesValue(values=[[15.0], [16.0]], indices=[3, 4], dense_shape=[5, 1])])
        inputs = inputs_data[0:group_size]
        if group_size == 1:
            expect = [IndexedSlices(values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1]), IndexedSlices(values=[[3.0], [4.0]], indices=[1, 2], dense_shape=[5, 1])]
        if group_size == 2:
            expect = [IndexedSlices(values=[[1.0], [2.0], [5.0], [6.0]], indices=[0, 1, 1, 2], dense_shape=[10, 1]), IndexedSlices(values=[[3.0], [4.0], [7.0], [8.0]], indices=[1, 2, 0, 1], dense_shape=[5, 1])]
        elif group_size == 4:
            expect = [IndexedSlices(values=[[1.0], [2.0], [5.0], [6.0], [9.0], [10.0], [13.0], [14.0]], indices=[0, 1, 1, 2, 3, 4, 8, 9], dense_shape=[10, 1]), IndexedSlices(values=[[3.0], [4.0], [7.0], [8.0], [11.0], [12.0], [15.0], [16.0]], indices=[1, 2, 0, 1, 3, 4, 3, 4], dense_shape=[5, 2])]
        self.batch_reduce_and_verify(inputs, expect, options)

    def testBatchReduceMixedDenseAndSparse(self):
        if False:
            return 10
        options = self.RunOptions(num_processes=2, gpus_per_process=0, reduce_op=ReduceOp.SUM, mode=['func_graph'])
        inputs_data = [[1.0, 2.0, IndexedSlicesValue(values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1]), IndexedSlicesValue(values=[[3.0], [4.0]], indices=[1, 2], dense_shape=[5, 1])], [3.0, 4.0, IndexedSlicesValue(values=[[5.0], [6.0]], indices=[1, 2], dense_shape=[10, 1]), IndexedSlicesValue(values=[[7.0], [8.0]], indices=[0, 1], dense_shape=[5, 1])]]
        expect = [4.0, 6.0, IndexedSlices(values=[[1.0], [2.0], [5.0], [6.0]], indices=[0, 1, 1, 2], dense_shape=[10, 1]), IndexedSlices(values=[[3.0], [4.0], [7.0], [8.0]], indices=[1, 2, 0, 1], dense_shape=[5, 1])]
        self.batch_reduce_and_verify(inputs_data, expect, options)

    @combinations.generate(combinations.combine(num_processes=[1, 2], required_gpus=[0, 1, 2], implementation=[CommunicationImplementation.AUTO, CommunicationImplementation.RING, CommunicationImplementation.NCCL], reduce_op=[ReduceOp.SUM, ReduceOp.MEAN]))
    def testAllReduceDense(self, num_processes, required_gpus, implementation, reduce_op):
        if False:
            for i in range(10):
                print('nop')
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                print('Hello World!')
            (collective, devices, _) = self.make_collective(num_processes, required_gpus)
            options = collective_util.Options(implementation=implementation)
            group_size = num_processes * (required_gpus or 1)

            @def_function.function
            def collective_all_reduce():
                if False:
                    print('Hello World!')
                results = []
                for (replica_id, device) in enumerate(devices):
                    with ops.device(device):
                        value = constant_op.constant(1.0)
                        results.append(collective._all_reduce(reduce_op, value, replica_id, options))
                return results
            got = collective_all_reduce()
            if reduce_op == ReduceOp.SUM:
                expect = [1.0 * group_size] * len(devices)
            elif reduce_op == ReduceOp.MEAN:
                expect = [1.0] * len(devices)
            self.assertAllClose(got, expect)

            @def_function.function
            def collective_batch_all_reduce():
                if False:
                    print('Hello World!')
                results = []
                for (replica_id, device) in enumerate(devices):
                    with ops.device(device):
                        value = (constant_op.constant(1.0), constant_op.constant(2.0))
                        results.append(collective._all_reduce(reduce_op, value, replica_id, options))
                return results
            got = collective_batch_all_reduce()
            if reduce_op == ReduceOp.SUM:
                expect = [(1.0 * group_size, 2.0 * group_size)] * len(devices)
            elif reduce_op == ReduceOp.MEAN:
                expect = [(1.0, 2.0)] * len(devices)
            self.assertAllClose(got, expect)
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=[1, 2], required_gpus=[0, 1, 2], implementation=[CommunicationImplementation.AUTO, CommunicationImplementation.RING, CommunicationImplementation.NCCL], reduce_op=[ReduceOp.SUM, ReduceOp.MEAN]))
    def testAllReduceSparse(self, num_processes, required_gpus, implementation, reduce_op):
        if False:
            print('Hello World!')
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                i = 10
                return i + 15
            (collective, devices, _) = self.make_collective(num_processes, required_gpus)
            options = collective_util.Options(implementation=implementation)
            group_size = num_processes * (required_gpus or 1)

            @def_function.function
            def collective_all_reduce():
                if False:
                    print('Hello World!')
                results = []
                for (replica_id, device) in enumerate(devices):
                    with ops.device(device):
                        value = IndexedSlices(values=array_ops.identity([[1.0]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1]))
                        results.append(collective._all_reduce(reduce_op, value, replica_id, options))
                return results
            got = collective_all_reduce()
            if reduce_op == ReduceOp.SUM:
                expect = [IndexedSlices([[1.0 * group_size]], [0], [5, 1])] * len(devices)
            elif reduce_op == ReduceOp.MEAN:
                expect = [IndexedSlices([[1.0]], [0], [5, 1])] * len(devices)
            self.assertAllClose(nest.map_structure(ops.convert_to_tensor, got), nest.map_structure(ops.convert_to_tensor, expect))

            @def_function.function
            def collective_batch_all_reduce():
                if False:
                    while True:
                        i = 10
                results = []
                for (replica_id, device) in enumerate(devices):
                    with ops.device(device):
                        value = (IndexedSlices(array_ops.identity([[1.0]]), array_ops.identity([0]), array_ops.identity([5, 1])), IndexedSlices(array_ops.identity([[3.0]]), array_ops.identity([2]), array_ops.identity([5, 1])))
                        results.append(collective._all_reduce(reduce_op, value, replica_id, options))
                return results
            got = collective_batch_all_reduce()
            if reduce_op == ReduceOp.SUM:
                expect = [(IndexedSlices([[1.0 * group_size]], [0], [5, 1]), IndexedSlices([[3.0 * group_size]], [2], [5, 1]))] * len(devices)
            elif reduce_op == ReduceOp.MEAN:
                expect = [(IndexedSlices([[1.0]], [0], [5, 1]), IndexedSlices([[3.0]], [2], [5, 1]))] * len(devices)
            self.assertAllClose(nest.map_structure(ops.convert_to_tensor, got), nest.map_structure(ops.convert_to_tensor, expect))
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=2, required_gpus=0, implementation=CommunicationImplementation.AUTO, reduce_op=ReduceOp.SUM))
    def testAllReduceMixedDenseAndSparse(self, num_processes, required_gpus, implementation, reduce_op):
        if False:
            i = 10
            return i + 15
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                return 10
            (collective, devices, _) = self.make_collective(num_processes, required_gpus)
            options = collective_util.Options(implementation=implementation)
            group_size = num_processes * (required_gpus or 1)

            @def_function.function
            def collective_batch_all_reduce():
                if False:
                    i = 10
                    return i + 15
                results = []
                for (replica_id, device) in enumerate(devices):
                    with ops.device(device):
                        value = (IndexedSlices(array_ops.identity([[1.0]]), array_ops.identity([0]), array_ops.identity([5, 1])), array_ops.identity(1.0), IndexedSlices(array_ops.identity([[3.0]]), array_ops.identity([2]), array_ops.identity([5, 1])), array_ops.identity(2.0))
                        results.append(collective._all_reduce(reduce_op, value, replica_id, options))
                return results
            got = collective_batch_all_reduce()
            expect = [(IndexedSlices([[1.0 * group_size]], [0], [5, 1]), 1.0 * group_size, IndexedSlices([[3.0 * group_size]], [2], [5, 1]), 2.0 * group_size)] * len(devices)
            self.assertAllClose(nest.map_structure(ops.convert_to_tensor, got), nest.map_structure(ops.convert_to_tensor, expect))
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=[1, 2], required_gpus=[0, 1, 2], axis=[0, 1, 2], func_mode=['eager', 'func_graph'], implementation=[CommunicationImplementation.AUTO, CommunicationImplementation.RING, CommunicationImplementation.NCCL], prefer_unique_instance_key=[True, False]))
    def testAllGatherSameShape(self, num_processes, required_gpus, implementation, func_mode, axis, prefer_unique_instance_key):
        if False:
            for i in range(10):
                print('nop')
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                return 10
            CollectiveReplicaLauncher._prefer_unique_instance_key = prefer_unique_instance_key
            (collective, devices, _) = self.make_collective(num_processes, required_gpus)
            options = collective_util.Options(implementation=implementation)
            value = constant_op.constant([[[1, 2], [1, 2]]], dtype=dtypes.float32)

            def gather_fn():
                if False:
                    while True:
                        i = 10
                per_replica_value = make_per_replica_value(value, devices)
                gathered_values = collective._gather(per_replica_value, per_replica_value, axis=axis, options=options)
                gathered_values = self.as_list(gathered_values)
                if not context.executing_eagerly():
                    self.assertAllEqual(devices, [v.device for v in gathered_values])
                return [ops.convert_to_tensor(v) for v in gathered_values]
            group_size = num_processes * (required_gpus or 1)
            expect = array_ops.concat([value] * group_size, axis=axis)
            per_replica_expect = [ops.convert_to_tensor(expect)] * len(devices)
            if func_mode == 'eager':
                result = gather_fn()
                self.assertAllClose(result, per_replica_expect)
            if func_mode == 'func_graph':
                result = def_function.function(gather_fn)()
                self.assertAllClose(result, per_replica_expect)
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=[1, 2], required_gpus=[0, 1, 2], implementation=[CommunicationImplementation.RING]))
    def testCollectiveV2ControlFlow(self, num_processes, required_gpus, implementation):
        if False:
            print('Hello World!')

        def replica_fn():
            if False:
                while True:
                    i = 10
            CollectiveReplicaLauncher._prefer_unique_instance_key = True
            (collective, devices, _) = self.make_collective(num_processes, required_gpus)
            options = collective_util.Options(implementation=implementation)
            value = make_per_replica_value(constant_op.constant([1.0]), devices)

            @def_function.function
            def reduce_fn():
                if False:
                    i = 10
                    return i + 15

                def cond_body():
                    if False:
                        print('Hello World!')
                    reduced = collective.reduce(reduce_util.ReduceOp.SUM, value, value, options)
                    return math_ops.add_n(self.as_list(reduced)) / len(devices)
                return cond.cond(array_ops.identity(False), cond_body, cond_body)
            num_replicas = num_processes * len(devices)
            self.assertAllEqual(reduce_fn(), [1.0 * num_replicas])
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=1, required_gpus=2, implementation=[CommunicationImplementation.RING, CommunicationImplementation.NCCL], prefer_unique_instance_key=[True, False]))
    def testMultiThreadedCollectiveLaunchNoInterleave(self, num_processes, required_gpus, implementation, prefer_unique_instance_key):
        if False:
            i = 10
            return i + 15
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                for i in range(10):
                    print('nop')
            CollectiveReplicaLauncher._prefer_unique_instance_key = prefer_unique_instance_key
            (collective, devices, _) = self.make_collective(num_processes, required_gpus)
            options = collective_util.Options(implementation=implementation)
            v0 = make_per_replica_value(1.0, devices)
            v1 = make_per_replica_value(2.0, devices)
            sequence = [v0.values[0], v1.values[0], v1.values[1], v0.values[1]]
            all_reduce = collective_ops.all_reduce

            def delayed_all_reduce(input_tensor, *args, **kwargs):
                if False:
                    return 10
                for (idx, v) in enumerate(sequence):
                    if input_tensor is v:
                        time.sleep(idx)
                        break
                return all_reduce(input_tensor, *args, **kwargs)
            with test.mock.patch.object(collective_ops, 'all_reduce', delayed_all_reduce):

                def thread_fn():
                    if False:
                        while True:
                            i = 10
                    reduced = collective.batch_reduce(reduce_util.ReduceOp.SUM, [(v0, v0), (v0, v0)], options)
                    self.assertAllEqual(reduced[0].values, [2.0, 2.0])
                    self.assertAllEqual(reduced[1].values, [2.0, 2.0])
                t = threading.Thread(target=thread_fn)
                t.start()
                reduced = collective.batch_reduce(reduce_util.ReduceOp.SUM, [(v1, v1), (v1, v1)], options)
                self.assertAllEqual(reduced[0].values, [4.0, 4.0])
                self.assertAllEqual(reduced[1].values, [4.0, 4.0])
                t.join()
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=1, required_gpus=2, implementation=[CommunicationImplementation.RING, CommunicationImplementation.NCCL], prefer_unique_instance_key=[True, False]))
    def testInputsAreFunctionArgs(self, num_processes, required_gpus, implementation, prefer_unique_instance_key):
        if False:
            for i in range(10):
                print('nop')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                return 10
            CollectiveReplicaLauncher._prefer_unique_instance_key = prefer_unique_instance_key
            (collective, devices, _) = self.make_collective(num_processes, required_gpus)
            options = collective_util.Options(implementation=implementation)

            @def_function.function
            def reduce_fn(v):
                if False:
                    for i in range(10):
                        print('nop')
                self.assertEqual(v.values[0].device, '')
                self.assertEqual(v.values[1].device, '')
                reduced = collective.batch_reduce(reduce_util.ReduceOp.SUM, [(v, v), (v, v)], options)
                self.assertEqual(reduced[0].values[0].device, devices[0])
                self.assertEqual(reduced[0].values[1].device, devices[1])
                self.assertEqual(reduced[1].values[0].device, devices[0])
                self.assertEqual(reduced[1].values[1].device, devices[1])
                return [reduced[0].values, reduced[1].values]
            v = make_per_replica_value(1.0, devices)
            reduced = reduce_fn(v)
            self.assertAllClose(reduced, [[2.0, 2.0], [2.0, 2.0]])
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=2, required_gpus=[0, 1], implementation=[CommunicationImplementation.RING, CommunicationImplementation.NCCL], prefer_unique_instance_key=[True, False]))
    def testTimeoutReduceDense(self, num_processes, implementation, required_gpus, prefer_unique_instance_key):
        if False:
            i = 10
            return i + 15
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                i = 10
                return i + 15
            CollectiveReplicaLauncher._prefer_unique_instance_key = prefer_unique_instance_key
            (collective, devices, task_id) = self.make_collective(num_processes, required_gpus)
            if task_id != 0:
                return
            v = make_per_replica_value(1.0, devices)
            options = collective_util.Options(timeout_seconds=1.0, implementation=implementation)

            @def_function.function
            def reduce_dense():
                if False:
                    return 10
                return collective.reduce(reduce_util.ReduceOp.SUM, v, v, options)
            with self.assertRaises(errors.DeadlineExceededError):
                reduce_dense()
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=2, required_gpus=[0, 1], implementation=[CommunicationImplementation.RING, CommunicationImplementation.NCCL], prefer_unique_instance_key=[True, False]))
    def testTimeoutBatchReduceDense(self, num_processes, implementation, required_gpus, prefer_unique_instance_key):
        if False:
            while True:
                i = 10
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                return 10
            CollectiveReplicaLauncher._prefer_unique_instance_key = prefer_unique_instance_key
            (collective, devices, task_id) = self.make_collective(num_processes, required_gpus)
            if task_id != 0:
                return
            v = make_per_replica_value(1.0, devices)
            options = collective_util.Options(timeout_seconds=1.0, implementation=implementation)

            @def_function.function
            def batch_reduce_dense():
                if False:
                    i = 10
                    return i + 15
                return collective.batch_reduce(reduce_util.ReduceOp.SUM, [(v, v), (v, v)], options)
            with self.assertRaises(errors.DeadlineExceededError):
                batch_reduce_dense()
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=2, required_gpus=[0, 1], implementation=[CommunicationImplementation.RING, CommunicationImplementation.NCCL], prefer_unique_instance_key=[True, False]))
    def testTimeoutReduceSparse(self, num_processes, implementation, required_gpus, prefer_unique_instance_key):
        if False:
            while True:
                i = 10
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                while True:
                    i = 10
            CollectiveReplicaLauncher._prefer_unique_instance_key = prefer_unique_instance_key
            (collective, devices, task_id) = self.make_collective(num_processes, required_gpus)
            if task_id != 0:
                return
            v = make_per_replica_value(IndexedSlicesValue(values=[[4.0, 6.0]], indices=[1], dense_shape=[5, 2]), devices)
            options = collective_util.Options(timeout_seconds=1.0, implementation=implementation)

            @def_function.function
            def reduce_sparse():
                if False:
                    return 10
                return collective.reduce(reduce_util.ReduceOp.SUM, v, v, options)
            with self.assertRaises(errors.DeadlineExceededError):
                reduce_sparse()
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=2, required_gpus=[0, 1], implementation=[CommunicationImplementation.RING, CommunicationImplementation.NCCL], prefer_unique_instance_key=[True, False]))
    def testTimeoutBatchReduceSparse(self, num_processes, required_gpus, implementation, prefer_unique_instance_key):
        if False:
            return 10
        if required_gpus == 0 and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip CPU + NCCL combination')
        if num_processes != required_gpus and implementation == CommunicationImplementation.NCCL:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')
        if num_processes != required_gpus and implementation == CommunicationImplementation.AUTO:
            self.skipTest('Skip potential NCCL combination (AUTO) with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                for i in range(10):
                    print('nop')
            CollectiveReplicaLauncher._prefer_unique_instance_key = prefer_unique_instance_key
            (collective, devices, task_id) = self.make_collective(num_processes, required_gpus)
            if task_id != 0:
                return
            v = make_per_replica_value(IndexedSlicesValue(values=[[4.0, 6.0]], indices=[1], dense_shape=[5, 2]), devices)
            options = collective_util.Options(timeout_seconds=1.0, implementation=implementation)

            @def_function.function
            def batch_reduce_sparse():
                if False:
                    i = 10
                    return i + 15
                return collective.batch_reduce(reduce_util.ReduceOp.SUM, [(v, v), (v, v)], options)
            with self.assertRaises(errors.DeadlineExceededError):
                batch_reduce_sparse()
        get_global_mpr(num_processes).run(replica_fn)

    @combinations.generate(combinations.combine(num_processes=1, required_gpus=2))
    def testNcclOrdering(self, num_processes, required_gpus):
        if False:
            for i in range(10):
                print('nop')
        if num_processes != required_gpus:
            self.skipTest('Skip NCCL combination with mismatched process and GPU count. NCCL requires physical GPUs for every process.')

        def replica_fn():
            if False:
                return 10
            CollectiveReplicaLauncher._prefer_unique_instance_key = True
            CollectiveReplicaLauncher._prefer_ordering_token = True
            (collective, devices, _) = self.make_collective(num_processes, required_gpus)
            options = collective_util.Options(implementation=CommunicationImplementation.NCCL)
            v_dense = make_per_replica_value([1.0, 1.0], devices)
            v_sparse = make_per_replica_value([IndexedSlicesValue([[4.0, 6.0], [5.0, 6.0]], [1, 3], [5, 2]), IndexedSlicesValue([[4.0, 6.0], [5.0, 6.0]], [1, 3], [5, 2])], devices)

            @def_function.function
            def nested_dense():
                if False:
                    return 10
                collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)

            @def_function.function
            def nested_sparse():
                if False:
                    i = 10
                    return i + 15
                collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)

            @def_function.function
            def f():
                if False:
                    while True:
                        i = 10
                collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)
                collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)
                collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)
                nested_dense()
                nested_sparse()
                if array_ops.identity(1.0) > array_ops.identity(2.0):
                    collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)
                else:
                    v_dense
                if array_ops.identity(1.0) > array_ops.identity(2.0):
                    v_sparse
                else:
                    collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)
                i = array_ops.identity(1)
                while i < 3:
                    collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)
                    i += 1
                i = array_ops.identity(1)
                while i < 3:
                    collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)
                    i += 1
                collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)
                collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)
            graph = f.get_concrete_function().graph
            should_be_ordered = set(['CollectiveReduceV2', 'CollectiveGatherV2', 'If', 'While', 'StatefulPartitionedCall'])
            nodes_by_device = {}
            for op in graph.get_operations():
                if op.type in should_be_ordered:
                    if op.device not in nodes_by_device:
                        nodes_by_device[op.device] = []
                    nodes_by_device[op.device].append(op)
            order = test_util.topological_sort_operations(graph.get_operations())
            for device in devices:
                device = device_util.canonicalize(device)
                operations = nodes_by_device[device] + nodes_by_device['']
                self.assertEqual(set((op.type for op in operations)), should_be_ordered)
                test_util.assert_sequential_execution(order, operations)
        get_global_mpr(num_processes).run(replica_fn)

@_REGISTER_DECORATOR(CollectiveOpsTest)
def _save_test_case(pickler, obj):
    if False:
        while True:
            i = 10

    def reconstruct(*args, **kwargs):
        if False:
            print('Hello World!')
        del args, kwargs
        return CollectiveOpsTest()
    return pickler.save_reduce(reconstruct, (), obj=obj)
if __name__ == '__main__':
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    test_util.main(config_logical_devices=False)