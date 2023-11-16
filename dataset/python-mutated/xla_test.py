"""Definition of XLA test case."""
from collections.abc import Iterator
import contextlib
import os
import random
import re
import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.xla import jit
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
FLAGS = flags.FLAGS
flags.DEFINE_string('test_device', None, 'Tensorflow device on which to place operators under test')
flags.DEFINE_string('types', None, 'Types to test. Comma-separated list.')
flags.DEFINE_string('disabled_manifest', None, 'Path to a file with a list of tests that should not run.')
flags.DEFINE_string('tf_xla_flags', None, 'Value to set the TF_XLA_FLAGS environment variable to')

def parse_disabled_manifest(manifest_content):
    if False:
        i = 10
        return i + 15
    comments_re = re.compile('#.*$')
    disabled_tests = []
    disabled_method_types = []
    for l in manifest_content.splitlines():
        stripped = comments_re.sub('', l).strip()
        if not stripped:
            continue
        entry = stripped.split(' ')
        if len(entry) == 1:
            disabled_tests.append(entry[0])
        elif len(entry) == 2:
            disabled_method_types.append((entry[0], entry[1].strip().split(',')))
        else:
            raise ValueError('Bad entry in manifest file.')
    disabled_regex = '|'.join(disabled_tests)
    method_types_filter = {}
    for (method, types) in disabled_method_types:
        method_types_filter[method] = set([dtypes.as_dtype(types_pb2.DataType.Value(name)).as_numpy_dtype for name in types])
    return (disabled_regex, method_types_filter)

class TPURewriteSession(session.Session):
    """Tensorflow session that runs tpu.rewrite() on ops on run()."""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.topology = None

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        if False:
            print('Hello World!')
        from tensorflow.python.tpu import tpu
        if self.topology is None:
            self.topology = super().run(tpu.initialize_system())
            assert self.topology is not None
        fetch_mapper = session._FetchMapper.for_fetch(fetches)
        new_fetches = []
        for fetch in fetch_mapper.unique_fetches():
            if isinstance(fetch, ops.Operation):
                fetch = tpu.rewrite(lambda fetch=fetch: fetch)
            new_fetches.append(fetch)
        rewritten_fetches = fetch_mapper.build_results(new_fetches)
        return super().run(rewritten_fetches, feed_dict, options, run_metadata)

class XLATestCase(test.TestCase):
    """XLA test cases are parameterized test cases."""

    def __init__(self, method_name='runTest'):
        if False:
            return 10
        super(XLATestCase, self).__init__(method_name)
        if 'XLA' in FLAGS.test_device:
            context.context().enable_xla_devices()
        if test_util.is_mlir_bridge_enabled():
            context.context().enable_mlir_bridge = True
        elif test_util.is_mlir_bridge_enabled() is not None:
            context.context().enable_mlir_bridge = False
        self.device = FLAGS.test_device
        self.has_custom_call = self.device == 'XLA_CPU'
        self.rewrite_ops_for_tpu = False
        self._all_tf_types = set([dtypes.as_dtype(types_pb2.DataType.Value(name)) for name in FLAGS.types.split(',')])
        self.int_tf_types = set([dtype for dtype in self._all_tf_types if dtype.is_integer])
        self._float_tf_types = set([dtype for dtype in self._all_tf_types if dtype.is_floating])
        self.complex_tf_types = set([dtype for dtype in self._all_tf_types if dtype.is_complex])
        self._numeric_tf_types = set(self.int_tf_types | self._float_tf_types | self.complex_tf_types)
        self.quantized_tf_types = set((dtype for dtype in self._all_tf_types if dtype.is_quantized))
        self._all_types = set((dtype.as_numpy_dtype for dtype in self._all_tf_types if not dtype.is_quantized))
        self._int_types = set([dtype.as_numpy_dtype for dtype in self.int_tf_types])
        self.signed_int_types = set((dtype.as_numpy_dtype for dtype in self.int_tf_types if not dtype.is_unsigned))
        self.unsigned_int_types = set((dtype.as_numpy_dtype for dtype in self.int_tf_types if dtype.is_unsigned))
        self._float_types = set([dtype.as_numpy_dtype for dtype in self._float_tf_types])
        self.complex_types = set([dtype.as_numpy_dtype for dtype in self.complex_tf_types])
        self._numeric_types = set(self._int_types | self._float_types | self.complex_types)
        self.disabled_regex = None
        self._method_types_filter = {}
        if FLAGS.disabled_manifest is not None:
            with open(FLAGS.disabled_manifest, 'r') as manifest_file:
                (disabled_regex, self._method_types_filter) = parse_disabled_manifest(manifest_file.read())
                if disabled_regex:
                    self.disabled_regex = re.compile(disabled_regex)
        if FLAGS.tf_xla_flags is not None:
            os.environ['TF_XLA_FLAGS'] = FLAGS.tf_xla_flags

    @property
    def all_tf_types(self):
        if False:
            for i in range(10):
                print('nop')
        name = '{}.{}'.format(type(self).__name__, self._testMethodName)
        tf_types = set([dtypes.as_dtype(t) for t in self._method_types_filter.get(name, set())])
        return self._all_tf_types - tf_types

    @property
    def float_types(self):
        if False:
            while True:
                i = 10
        name = '{}.{}'.format(type(self).__name__, self._testMethodName)
        return self._float_types - self._method_types_filter.get(name, set())

    @property
    def float_tf_types(self):
        if False:
            return 10
        name = '{}.{}'.format(type(self).__name__, self._testMethodName)
        return self._float_tf_types - self._method_types_filter.get(name, set())

    @property
    def int_types(self):
        if False:
            i = 10
            return i + 15
        name = '{}.{}'.format(type(self).__name__, self._testMethodName)
        return self._int_types - self._method_types_filter.get(name, set())

    @property
    def numeric_tf_types(self):
        if False:
            return 10
        name = '{}.{}'.format(type(self).__name__, self._testMethodName)
        tf_types = set([dtypes.as_dtype(t) for t in self._method_types_filter.get(name, set())])
        return self._numeric_tf_types - tf_types

    @property
    def numeric_types(self):
        if False:
            while True:
                i = 10
        name = '{}.{}'.format(type(self).__name__, self._testMethodName)
        return self._numeric_types - self._method_types_filter.get(name, set())

    @property
    def all_types(self):
        if False:
            while True:
                i = 10
        name = '{}.{}'.format(type(self).__name__, self._testMethodName)
        return self._all_types - self._method_types_filter.get(name, set())

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(XLATestCase, self).setUp()
        name = '{}.{}'.format(type(self).__name__, self._testMethodName)
        if self.disabled_regex is not None and self.disabled_regex.match(name):
            logging.info('Disabled test case: %s', name)
            self.skipTest('{} is disabled by manifest.'.format(name))
            return
        logging.info('Start test case: %s', name)
        random.seed(random_seed.DEFAULT_GRAPH_SEED)
        np.random.seed(random_seed.DEFAULT_GRAPH_SEED)

    def tearDown(self):
        if False:
            return 10
        super(XLATestCase, self).tearDown()
        logging.info('End test case: %s', self._testMethodName)

    @contextlib.contextmanager
    def session(self) -> Iterator[session.Session]:
        if False:
            print('Hello World!')
        'Custom implementation of session() for XLA tests.\n\n    We override the standard Tensorflow session() since it is too\n    specific to CPU and GPU tests. In particular, we want to disable soft\n    placement and explicitly assign ops to devices under test.\n\n    Yields:\n      A session to use when running a test case.\n    '
        graph = ops.Graph()
        config = context.context().config
        config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
        if self.rewrite_ops_for_tpu:
            session_type = TPURewriteSession
        else:
            session_type = session.Session
        with session_type(graph=graph, config=config) as sess, graph.as_default():
            yield sess

    def cached_session(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('cached_session not supported on XLATestCase, please use session')

    def test_session(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('test_session not supported on XLATestCase, please use session')

    @contextlib.contextmanager
    def device_scope(self):
        if False:
            for i in range(10):
                print('nop')
        'Scope that runs tests on `self.device`.\n\n    Yields:\n      A scope to apply to the operators under test.\n    '
        with ops.device('device:{}:0'.format(self.device)):
            yield

    def test_scope(self):
        if False:
            return 10
        'Deprecated alias of `device_scope`.\n\n    This should be avoided as the name starts with `test`, so test runners\n    treat it as a test. This interferes with class decorators that operate on\n    each test method.\n    '
        return self.device_scope()

def Benchmark(tf_bench, builder_fn, use_xla_jit, device, separate_compiled_gradients=False):
    if False:
        print('Hello World!')
    'Build a graph and run benchmarks against it, with or without XLA.\n\n  Args:\n    tf_bench: An instance of tf.test.Benchmark, used to run the benchmark.\n    builder_fn: A function that builds a graph when invoked, and returns\n        (name, fetches), where name is the name of the test, and fetches\n        is a list of tensors to fetch as output.\n    use_xla_jit: If true compile with the XLA JIT, otherwise use regular TF.\n    device: The tensorflow device to run on, e.g. "cpu", "gpu".\n    separate_compiled_gradients: If true put each gradient subgraph into a\n      separate compilation scope. This gives fine-grained control over which\n      portions of the graph will be compiled as a single unit. Compiling\n      gradients separately may yield better performance for some graphs.\n      The scope is named based on the scope of the forward computation as well\n      as the name of the gradients. As a result, the gradients will be compiled\n      in a scope that is separate from both the forward computation, and from\n      other gradients.\n  '
    with ops.Graph().as_default():
        name = None
        targets = []
        with ops.device(device):
            fetches = []
            jit_scope = jit.experimental_jit_scope
            with jit_scope(compile_ops=use_xla_jit, separate_compiled_gradients=separate_compiled_gradients):
                (name, fetches) = builder_fn()
            for fetch in fetches:
                targets.append(array_ops.identity(fetch).op)
        config = config_pb2.ConfigProto(allow_soft_placement=True)
        with session.Session(config=config) as sess:
            sess.run(variables.global_variables_initializer())
            xla = 'xla_' if use_xla_jit else ''
            tf_bench.run_op_benchmark(sess, targets, name='%s_%s%s' % (name, xla, device))