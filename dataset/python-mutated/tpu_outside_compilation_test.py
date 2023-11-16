"""Tests for TPU outside compilation."""
import os
import tempfile
from absl.testing import parameterized
import numpy as np
from tensorboard.plugins.histogram import summary_v2 as histogram_summary_v2
from tensorboard.plugins.image import summary_v2 as image_summary_v2
from tensorboard.plugins.scalar import summary_v2 as scalar_summary_v2
from tensorflow.core.util import event_pb2
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
FLAGS = flags.FLAGS
flags.DEFINE_bool('use_local_tpu', False, 'use local TPUs on a TPU VM instead of connecting to a GCP TPU VM or node.')
flags.DEFINE_string('tpu', '', 'Name of TPU to connect to.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')

def get_tpu_cluster_resolver():
    if False:
        for i in range(10):
            print('nop')
    if FLAGS.use_local_tpu:
        return tpu_cluster_resolver.TPUClusterResolver('local')
    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
    return resolver

def get_tpu_strategy():
    if False:
        while True:
            i = 10
    resolver = get_tpu_cluster_resolver()
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    return tpu_lib.TPUStrategyV2(resolver)

def computation_with_string_ops(x):
    if False:
        return 10
    output = string_ops.string_format('1{}', x)
    return string_ops.string_to_number(output)

def _events_from_logdir(test_case, logdir):
    if False:
        print('Hello World!')
    'Reads summary events from log directory.'
    test_case.assertTrue(gfile.Exists(logdir))
    files = gfile.ListDirectory(logdir)
    test_case.assertLen(files, 1)
    records = list(tf_record.tf_record_iterator(os.path.join(logdir, files[0])))
    result = []
    for r in records:
        event = event_pb2.Event()
        event.ParseFromString(r)
        result.append(event)
    return result

def _rewrite_func_wrapper(tf_func):
    if False:
        for i in range(10):
            print('nop')

    def tpu_fn(*args, **kwargs):
        if False:
            while True:
                i = 10
        concrete = tf_func.get_concrete_function(*list(args) + list(kwargs.values()))
        return tpu.rewrite(concrete.__call__, list(args) + list(kwargs.values()))
    return def_function.function(tpu_fn)

def _tpu_partitioned_call_wrapper(tf_func):
    if False:
        while True:
            i = 10
    'Wrap a tensorflow Function with TPUPartitionedCall.'

    def inner_func(*args, **kwargs):
        if False:
            print('Hello World!')
        concrete = tf_func.get_concrete_function(*args, **kwargs)
        op_args = list(args) + list(kwargs.values()) + concrete.captured_inputs
        return tpu_functional.TPUPartitionedCall(args=op_args, device_ordinal=tpu_ops.tpu_ordinal_selector(), Tout=[o.type for o in concrete.function_def.signature.output_arg], f=concrete)
    return def_function.function(inner_func)

class TpuOutsideCompilationTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TpuOutsideCompilationTest, self).setUp()
        config.set_soft_device_placement(False)

    def testHostNoInput(self):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy()

        def outside_fn():
            if False:
                return 10
            logging_ops.print_v2('Outside compiled')

        @def_function.function
        def train_step():
            if False:
                for i in range(10):
                    print('nop')

            def tpu_fn(x):
                if False:
                    return 10
                x2 = x + 5.0
                tpu_replication.outside_compilation(outside_fn)
                return x2 + 5.0
            return strategy.run(tpu_fn, args=(25.0,))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(35.0, shape=strategy.num_replicas_in_sync))

    def testHostInputOnly(self):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy()

        def outside_fn(x):
            if False:
                print('Hello World!')
            logging_ops.print_v2('Outside compiled', x)

        @def_function.function
        def train_step():
            if False:
                print('Hello World!')

            def tpu_fn(x):
                if False:
                    i = 10
                    return i + 15
                x2 = x + 5.0
                tpu_replication.outside_compilation(outside_fn, x2)
                return x2 + 5.0
            return strategy.run(tpu_fn, args=(25.0,))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(35.0, shape=strategy.num_replicas_in_sync))

    def testJitCompile(self):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy()

        def outside_fn(x):
            if False:
                i = 10
                return i + 15
            logging_ops.print_v2('Outside compiled', x)

        @def_function.function(jit_compile=True)
        def train_step():
            if False:
                for i in range(10):
                    print('nop')

            def tpu_fn(x):
                if False:
                    while True:
                        i = 10
                x2 = x + 5.0
                tpu_replication.outside_compilation(outside_fn, x2)
                return x2 + 5.0
            return strategy.run(tpu_fn, args=(25.0,))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(35.0, shape=strategy.num_replicas_in_sync))

    def testHostInputOutput(self):
        if False:
            return 10
        strategy = get_tpu_strategy()

        def outside_fn(x):
            if False:
                return 10
            logging_ops.print_v2('Outside compiled', x)
            return x + 6.0

        @def_function.function
        def train_step():
            if False:
                for i in range(10):
                    print('nop')

            def tpu_fn(x):
                if False:
                    print('Hello World!')
                x2 = x + 5.0
                output = tpu_replication.outside_compilation(outside_fn, x2)
                return output
            return strategy.run(tpu_fn, args=(25.0,))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(36.0, shape=strategy.num_replicas_in_sync))

    def testHostMultipleInputs(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()
        val0 = np.arange(6).reshape((2, 3)).astype(np.float32)
        val1 = np.arange(6).reshape((3, 2)).astype(np.float32)

        def outside_fn(arg0, arg1):
            if False:
                i = 10
                return i + 15
            tmp = array_ops.reshape(arg1, array_ops.shape(arg0))
            ret0 = arg0 + tmp
            ret1 = math_ops.matmul(arg0, arg1)
            ret2 = array_ops.concat([arg0, tmp], 0)
            return (ret0, ret1, ret2)

        @def_function.function
        def train_step():
            if False:
                print('Hello World!')

            def tpu_fn(x, y):
                if False:
                    i = 10
                    return i + 15
                a = x + 7.0
                b = y * 2.0
                (c, d, e) = tpu_replication.outside_compilation(outside_fn, a, b)
                return math_ops.reduce_max(c) + math_ops.reduce_min(d) + math_ops.reduce_sum(e)
            return strategy.run(tpu_fn, args=(val0, val1))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(213.0, shape=strategy.num_replicas_in_sync))

    def testMultipleClusters(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()

        def outside_fn1(x):
            if False:
                for i in range(10):
                    print('nop')
            logging_ops.print_v2('Outside compiled', x)
            return x + 6.0

        def outside_fn2(x):
            if False:
                return 10
            logging_ops.print_v2('Outside compiled', x)
            return x - 18.0

        @def_function.function
        def train_step():
            if False:
                for i in range(10):
                    print('nop')

            def tpu_fn(x):
                if False:
                    i = 10
                    return i + 15
                x2 = x + 5.0
                output1 = tpu_replication.outside_compilation(outside_fn1, x2)
                x3 = output1 + 3.0
                output2 = tpu_replication.outside_compilation(outside_fn2, x3)
                return output2
            return strategy.run(tpu_fn, args=(25.0,))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(21.0, shape=strategy.num_replicas_in_sync))

    @parameterized.parameters(True, False)
    def testOutsideCompilationControlFlowIf(self, take_true_branch):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()

        def outside_fn(x):
            if False:
                i = 10
                return i + 15
            logging_ops.print_v2('Outside compiled', x)
            return x + 6.0
        input_value = 51.0 if take_true_branch else 25.0

        @def_function.function
        def train_step():
            if False:
                i = 10
                return i + 15

            def tpu_fn(x):
                if False:
                    print('Hello World!')
                x2 = x + 5.0
                if x < 50.0:
                    return tpu_replication.outside_compilation(outside_fn, x2)
                else:
                    return x2
            return strategy.run(tpu_fn, args=(input_value,))
        output_value = 36.0
        if take_true_branch:
            output_value = 56.0
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(output_value, shape=strategy.num_replicas_in_sync))

    def testOutsideCompilationControlFlowWhile(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()

        def outside_fn(x):
            if False:
                print('Hello World!')
            logging_ops.print_v2('Outside compiled', x)
            return x + 6.0

        @def_function.function
        def train_step():
            if False:
                return 10

            def tpu_fn(x):
                if False:
                    i = 10
                    return i + 15
                x2 = x + 5.0
                while x2 < 50.0:
                    x2 = tpu_replication.outside_compilation(outside_fn, x2)
                return x2 + 4.0
            return strategy.run(tpu_fn, args=(25.0,))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(58.0, shape=strategy.num_replicas_in_sync))

    def testOutsideCompilationHostControlFlow(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that control flow on host for outside_compilation works.'
        strategy = get_tpu_strategy()

        def outside_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            n = 0
            while n < 4:
                x = x + 6.0
                n = n + 1
            return x

        @def_function.function
        def train_step():
            if False:
                while True:
                    i = 10

            def tpu_fn(x):
                if False:
                    return 10
                x2 = x + 5.0
                x2 = tpu_replication.outside_compilation(outside_fn, x2)
                return x2 + 4.0
            return strategy.run(tpu_fn, args=(25.0,))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(58.0, shape=strategy.num_replicas_in_sync))

    def testSummary(self):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy()

        def host_computation(x):
            if False:
                for i in range(10):
                    print('nop')
            scalar_summary_v2.scalar('x', x, step=0)
            return x * 2.0

        @def_function.function
        def step():
            if False:
                while True:
                    i = 10

            def computation(x):
                if False:
                    for i in range(10):
                        print('nop')
                x = x + 1.0
                y = tpu_replication.outside_compilation(host_computation, x)
                y = tpu_replication.outside_compilation(host_computation, x)
                return y + 1.0
            return strategy.run(computation, args=(2.0,))
        summary_writer = summary.create_file_writer(os.path.join(os.getenv('TEST_TMPDIR', '/tmp')), flush_millis=10000)
        with summary_writer.as_default(), summary.always_record_summaries():
            self.assertAllEqual(strategy.experimental_local_results(step()), constant_op.constant(7.0, shape=strategy.num_replicas_in_sync))

    @parameterized.parameters(True, False)
    def testSummaryInCond(self, take_true_branch):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()

        def host_computation(x):
            if False:
                return 10
            scalar_summary_v2.scalar('x', x, step=0)
            return x * 2.0

        @def_function.function
        def step(take_true_branch):
            if False:
                while True:
                    i = 10

            def computation(x):
                if False:
                    i = 10
                    return i + 15
                x = x + 1.0
                if x < 5.0:
                    y = tpu_replication.outside_compilation(host_computation, x)
                    y = tpu_replication.outside_compilation(host_computation, x)
                    x = y
                return x + 1.0
            if take_true_branch:
                return strategy.run(computation, args=(2.0,))
            else:
                return strategy.run(computation, args=(10.0,))
        summary_writer = summary.create_file_writer(os.path.join(os.getenv('TEST_TMPDIR', '/tmp')), flush_millis=10000)
        output_value = 12.0
        if take_true_branch:
            output_value = 7.0
        with summary_writer.as_default(), summary.always_record_summaries():
            self.assertAllEqual(strategy.experimental_local_results(step(take_true_branch)), constant_op.constant(output_value, shape=strategy.num_replicas_in_sync))

    def testSummaryInWhile(self):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy()

        def host_computation(x):
            if False:
                while True:
                    i = 10
            scalar_summary_v2.scalar('x', x, step=0)
            return x * 2.0

        @def_function.function
        def step():
            if False:
                i = 10
                return i + 15

            def computation(x):
                if False:
                    i = 10
                    return i + 15
                n = 0
                while n < 3:
                    x = x + 1.0
                    y = tpu_replication.outside_compilation(host_computation, x)
                    y = tpu_replication.outside_compilation(host_computation, x)
                    x = y
                    n = n + 1
                return x + 1.0
            return strategy.run(computation, args=(2.0,))
        summary_writer = summary.create_file_writer(os.path.join(os.getenv('TEST_TMPDIR', '/tmp')), flush_millis=10000)
        with summary_writer.as_default(), summary.always_record_summaries():
            self.assertAllEqual(strategy.experimental_local_results(step()), constant_op.constant(31.0, shape=strategy.num_replicas_in_sync))

    def testOutsideCompilationAtHeadAndTail(self):
        if False:
            print('Hello World!')
        'Tests that outside_compilation at head/tail of TPU computation works.'
        strategy = get_tpu_strategy()

        def host_computation(x):
            if False:
                return 10
            return x * 2.0

        @def_function.function
        def train_step():
            if False:
                while True:
                    i = 10

            def computation(x):
                if False:
                    for i in range(10):
                        print('nop')
                w = tpu_replication.outside_compilation(host_computation, x)
                y = w + 1.0
                z = tpu_replication.outside_compilation(host_computation, y)
                return z + 5.0
            return strategy.run(computation, args=(2.0,))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(15.0, shape=strategy.num_replicas_in_sync))

    def testGradientAcrossOutsideCompilation(self):
        if False:
            print('Hello World!')
        'Tests compiled gradients can contain host computations.'
        strategy = get_tpu_strategy()

        def host_computation(a):
            if False:
                for i in range(10):
                    print('nop')
            b = a * a
            c = b * b
            return c

        @def_function.function
        def train_step():
            if False:
                while True:
                    i = 10

            def computation(x, y):
                if False:
                    return 10
                a = x + 7.0
                b = tpu_replication.outside_compilation(host_computation, a)
                c = b * y
                d = gradients_impl.gradients([c], [x], colocate_gradients_with_ops=True)[0]
                return d
            return strategy.run(computation, args=(2.0, 3.0))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(8748.0, shape=strategy.num_replicas_in_sync))

    def testGradientOfGradientAcrossOutsideCompilation(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests compiled gradients of gradients can contain host computations.'
        strategy = get_tpu_strategy()

        def host_computation(a):
            if False:
                return 10
            b = a * a
            c = b * b
            return c

        @def_function.function
        def train_step():
            if False:
                return 10

            def computation(x, y):
                if False:
                    print('Hello World!')
                a = x + 7.0
                b = tpu_replication.outside_compilation(host_computation, a)
                c = b * y
                d = gradients_impl.gradients([c], [x], colocate_gradients_with_ops=True)[0]
                e = gradients_impl.gradients([d], [x], colocate_gradients_with_ops=True)[0]
                return e
            return strategy.run(computation, args=(2.0, 3.0))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(2916.0, shape=strategy.num_replicas_in_sync))

    def testColocateGradientWithOutsideCompiledOp(self):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy()

        @def_function.function
        def train_step():
            if False:
                return 10

            @def_function.function
            def tpu_fn(x):
                if False:
                    return 10
                x1 = tpu_replication.outside_compilation(math_ops.sqrt, x)
                grad = gradients_impl.gradients([x1], [x], colocate_gradients_with_ops=True)[0]
                sqrt = [op for op in ops.get_default_graph().get_operations() if op.type == 'Sqrt'][0]
                sqrt_grad = [op for op in ops.get_default_graph().get_operations() if op.type == 'SqrtGrad'][0]
                assert sqrt.get_attr(tpu_replication._OUTSIDE_COMPILATION_ATTR) == b'0'
                assert sqrt_grad.get_attr(tpu_replication._OUTSIDE_COMPILATION_ATTR) == b'0.gradients/uid'
                return grad
            return strategy.run(tpu_fn, args=(25.0,))
        self.assertAllEqual(strategy.experimental_local_results(train_step()), constant_op.constant(0.1, shape=strategy.num_replicas_in_sync))

class OutsideCompilationOnUnsupportedOpTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(OutsideCompilationOnUnsupportedOpTest, self).setUp()
        config.set_soft_device_placement(True)

    def testStringOpWithManualOutsideCompilation(self):
        if False:
            return 10
        strategy = get_tpu_strategy()

        @def_function.function
        def train_step(x):
            if False:
                return 10

            def computation(x):
                if False:
                    while True:
                        i = 10
                return tpu_replication.outside_compilation(computation_with_string_ops, x)
            return strategy.run(computation, args=(x,))
        self.assertAllEqual(strategy.experimental_local_results(train_step(0)), constant_op.constant(10, shape=strategy.num_replicas_in_sync))

    def testStringOpWithAutoOutsideCompilation(self):
        if False:
            return 10
        strategy = get_tpu_strategy()

        @def_function.function
        def train_step(x):
            if False:
                print('Hello World!')

            def computation(x):
                if False:
                    return 10
                return computation_with_string_ops(x)
            return strategy.run(computation, args=(x,))
        self.assertAllEqual(strategy.experimental_local_results(train_step(0)), constant_op.constant(10, shape=strategy.num_replicas_in_sync))

    def testImageSummary(self):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy()

        def run():
            if False:
                return 10

            @def_function.function
            def sample_sequence():
                if False:
                    while True:
                        i = 10
                bsz = 3
                max_length = 32 * 32

                def f():
                    if False:
                        for i in range(10):
                            print('nop')

                    def body(step, tokens):
                        if False:
                            i = 10
                            return i + 15
                        next_token = random_ops.random_uniform([bsz])
                        tokens = tokens.write(step, next_token)
                        return (step + 1, tokens)

                    def cond_fn(step, tokens):
                        if False:
                            print('Hello World!')
                        del tokens
                        return math_ops.less(step, max_length)
                    tokens_var = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=max_length, dynamic_size=False, clear_after_read=False, element_shape=(bsz,), name='tokens_accumulator')
                    step = constant_op.constant(0)
                    (step, tokens_var) = while_loop.while_loop(cond_fn, body, [step, tokens_var])
                    image_flat = array_ops.transpose(tokens_var.stack(), [1, 0])
                    image = array_ops.tile(array_ops.reshape(image_flat, [bsz, 32, 32, 1]), [1, 1, 1, 3])
                    image_summary_v2.image('image_sample', image, constant_op.constant(5, dtype=dtypes.int64))
                return strategy.run(f)
            sample_sequence()
        logdir = tempfile.mkdtemp()
        summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
        with summary_writer.as_default(), summary.always_record_summaries():
            run()
        events = _events_from_logdir(self, logdir)
        decoded_image = image_ops.decode_png(events[1].summary.value[0].tensor.string_val[2]).numpy()
        self.assertNotAllEqual(array_ops.zeros((3072,), dtype=dtypes.float32), list(decoded_image.flat))

    def testSummaryWithAutoOutsideCompilation(self):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy()

        def host_computation(x):
            if False:
                return 10
            scalar_summary_v2.scalar('x', x, step=0)
            return x * 2.0

        @def_function.function
        def step():
            if False:
                for i in range(10):
                    print('nop')

            def computation(x):
                if False:
                    i = 10
                    return i + 15
                x = x + 1.0
                y = host_computation(x)
                return y + 1.0
            return strategy.run(computation, args=(2.0,))
        logdir = tempfile.mkdtemp()
        summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
        with summary_writer.as_default(), summary.always_record_summaries():
            self.assertAllEqual(strategy.experimental_local_results(step()), constant_op.constant(7.0, shape=strategy.num_replicas_in_sync))
        events = _events_from_logdir(self, logdir)
        self.assertLen(events, 2)
        self.assertEqual(events[1].summary.value[0].tag, 'x')

    def testNestedFunctionScalarSummary(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()

        def host_computation(x):
            if False:
                print('Hello World!')
            scalar_summary_v2.scalar('x', x, step=0)
            return x * 2.0

        @def_function.function
        def step():
            if False:
                while True:
                    i = 10

            @def_function.function
            def computation(x):
                if False:
                    while True:
                        i = 10
                x = x + 1.0
                y = host_computation(x)
                return y + 1.0
            return strategy.run(computation, args=(2.0,))
        logdir = tempfile.mkdtemp()
        summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
        with summary_writer.as_default(), summary.always_record_summaries():
            self.assertAllEqual(strategy.experimental_local_results(step()), constant_op.constant(7.0, shape=strategy.num_replicas_in_sync))
        events = _events_from_logdir(self, logdir)
        self.assertLen(events, 2)
        self.assertEqual(events[1].summary.value[0].tag, 'x')

    def testHistogramSummaryWithAutoOutsideCompilation(self):
        if False:
            return 10
        strategy = get_tpu_strategy()

        def host_computation(x):
            if False:
                print('Hello World!')
            histogram_summary_v2.histogram('x', x, step=0)
            return x * 2.0

        @def_function.function
        def step():
            if False:
                return 10

            def computation(x):
                if False:
                    return 10
                x = x + 1.0
                y = host_computation(x)
                return y + 1.0
            return strategy.run(computation, args=(2.0,))
        logdir = tempfile.mkdtemp()
        summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
        with summary_writer.as_default(), summary.always_record_summaries():
            self.assertAllEqual(strategy.experimental_local_results(step()), constant_op.constant(7.0, shape=strategy.num_replicas_in_sync))
        events = _events_from_logdir(self, logdir)
        self.assertLen(events, 2)
        self.assertEqual(events[1].summary.value[0].tag, 'x')

    @parameterized.parameters(True, False)
    def testSummaryControlFlowIfWithAutoOutsideCompilation(self, take_true_branch):
        if False:
            return 10
        strategy = get_tpu_strategy()

        @def_function.function
        def step():
            if False:
                while True:
                    i = 10

            def computation(x):
                if False:
                    print('Hello World!')
                x = x + 1.0
                if x < 5:
                    scalar_summary_v2.scalar('x', x, step=0)
                    x = x * 2.0
                return x + 1.0
            if take_true_branch:
                return strategy.run(computation, args=(2.0,))
            else:
                return strategy.run(computation, args=(10.0,))
        logdir = tempfile.mkdtemp()
        summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
        output_value = 12.0
        if take_true_branch:
            output_value = 7.0
        with summary_writer.as_default(), summary.always_record_summaries():
            self.assertAllEqual(strategy.experimental_local_results(step()), constant_op.constant(output_value, shape=strategy.num_replicas_in_sync))
        if take_true_branch:
            events = _events_from_logdir(self, logdir)
            self.assertLen(events, 2)
            self.assertEqual(events[1].summary.value[0].tag, 'cond/x')

    def testAutoOutsideCompilationWithFunctionalNodes(self):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy()

        @def_function.function
        def train_step(a, b):
            if False:
                for i in range(10):
                    print('nop')

            def fn(a, b):
                if False:
                    return 10
                fn1 = lambda : computation_with_string_ops(a * 100)
                fn2 = lambda : computation_with_string_ops(a)
                pred = math_ops.greater_equal(a, b)
                result = array_ops.identity(cond.cond(pred, fn1, fn2), name='uncompilable_control_flow')
                return result
            return strategy.run(fn, args=(a, b))
        self.assertAllEqual(strategy.experimental_local_results(train_step(0.0, -1.0)), constant_op.constant(10, shape=strategy.num_replicas_in_sync))

    def testRandomOpsWithAutoOutsideCompilation(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()

        @def_function.function
        def train_step():
            if False:
                i = 10
                return i + 15

            def computation():
                if False:
                    while True:
                        i = 10
                return random_ops.random_normal(shape=[1, 2, 3])
            return strategy.run(computation, args=())
        self.assertAllEqual(strategy.experimental_local_results(train_step())[0].shape, [1, 2, 3])

    def testOutsideCompilationWithTPUPartitionedCallOp(self):
        if False:
            return 10
        'Tests that control flow with TPUPartitionedCall including outside_compilation works.'
        get_tpu_strategy()

        def host_computation(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 1

        @def_function.function()
        def train_step(x):
            if False:
                return 10
            x2 = x + 5.0
            logging_ops.print_v2(x2)
            x2 = tpu_replication.outside_compilation(host_computation, x2)
            return x2 + 4.0
        tpu_fn = _rewrite_func_wrapper(train_step)
        partitioned_tpu_fn = _tpu_partitioned_call_wrapper(tpu_fn)
        concrete = partitioned_tpu_fn.get_concrete_function(x=tensor.TensorSpec(shape=1, dtype=dtypes.float32, name='input_tensor'))
        self.assertIsInstance(concrete(array_ops.ones(1, dtype=dtypes.float32))[0], tensor.Tensor)
if __name__ == '__main__':
    test.main()