"""Tests for runtime_client."""
from google.protobuf import text_format
from tensorflow.core.framework import function_pb2
from tensorflow.core.function.runtime_client import runtime_client
from tensorflow.core.function.testing import test_pass
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class RuntimeClientTest(test.TestCase):

    def test_create_nullary(self):
        if False:
            while True:
                i = 10
        fndef = text_format.Parse("\n            signature {\n               name: 'NullaryFunction'\n               output_arg { name: 'o' type: DT_INT32 }\n             }\n             node_def {\n               name: 'retval'\n               op: 'Const'\n               attr {\n                 key: 'dtype'\n                 value { type: DT_INT32 }\n               }\n               attr {\n                 key: 'value'\n                 value {\n                   tensor {\n                     dtype: DT_INT32\n                     tensor_shape {}\n                     int_val: 1\n                   }\n                 }\n               }\n             }\n             ret { key: 'o' value: 'retval:output' }\n         ", function_pb2.FunctionDef())
        ctx = runtime_client.GlobalEagerContext()
        rt = runtime_client.Runtime(ctx)
        rt.CreateFunction(fndef)

    def test_create_function_called_by_py_runtime(self):
        if False:
            i = 10
            return i + 15
        if not tf2.enabled():
            self.skipTest('TF2 test')
        fndef = text_format.Parse("\n            signature {\n               name: 'NullaryFunction'\n               output_arg { name: 'o' type: DT_INT32 }\n             }\n             node_def {\n               name: 'retval'\n               op: 'Const'\n               attr {\n                 key: 'dtype'\n                 value { type: DT_INT32 }\n               }\n               attr {\n                 key: 'value'\n                 value {\n                   tensor {\n                     dtype: DT_INT32\n                     tensor_shape {}\n                     int_val: 1\n                   }\n                 }\n               }\n             }\n             ret { key: 'o' value: 'retval:output' }\n         ", function_pb2.FunctionDef())
        ctx = runtime_client.GlobalPythonEagerContext()
        rt = runtime_client.Runtime(ctx)
        rt.CreateFunction(fndef)
        (ret,) = execute.execute('NullaryFunction', 1, [], (), context.context())
        self.assertAllEqual(ret, 1)

    def test_get_function_proto_from_py_runtime_function(self):
        if False:
            i = 10
            return i + 15
        if not tf2.enabled():
            self.skipTest('TF2 test')

        @def_function.function
        def f():
            if False:
                print('Hello World!')
            return 1
        cf = f.get_concrete_function()
        ctx = runtime_client.GlobalPythonEagerContext()
        rt = runtime_client.Runtime(ctx)
        fndef = rt.GetFunctionProto(cf.function_def.signature.name)
        self.assertEqual(fndef.signature.name, cf.function_def.signature.name)

    def test_concrete_function_editing_proto_executed_directly(self):
        if False:
            for i in range(10):
                print('nop')
        if not tf2.enabled():
            self.skipTest('TF2 test')

        @def_function.function
        def f():
            if False:
                return 10
            return 1
        cf = f.get_concrete_function()
        ctx = runtime_client.GlobalPythonEagerContext()
        rt = runtime_client.Runtime(ctx)
        fndef = rt.GetFunctionProto(cf.function_def.signature.name)
        fndef.node_def[0].attr['value'].tensor.int_val[0] = 2
        rt.CreateFunction(fndef)
        (ret,) = execute.execute(fndef.signature.name, 1, [], (), context.context())
        self.assertAllEqual(ret, 2)

    def test_concrete_function_editing_proto(self):
        if False:
            print('Hello World!')
        if not tf2.enabled():
            self.skipTest('TF2 test')

        @def_function.function
        def f():
            if False:
                print('Hello World!')
            return 1
        cf = f.get_concrete_function()
        ctx = runtime_client.GlobalPythonEagerContext()
        rt = runtime_client.Runtime(ctx)
        fndef = rt.GetFunctionProto(cf.function_def.signature.name)
        fndef.node_def[0].attr['value'].tensor.int_val[0] = 2
        rt.CreateFunction(fndef)
        self.assertAllEqual(self.evaluate(f()), 2)

    def test_concrete_function_editing_proto_after_instantiation(self):
        if False:
            while True:
                i = 10
        if not tf2.enabled():
            self.skipTest('TF2 test')

        @def_function.function
        def f():
            if False:
                while True:
                    i = 10
            return 1
        cf = f.get_concrete_function()
        ctx = runtime_client.GlobalPythonEagerContext()
        rt = runtime_client.Runtime(ctx)
        fndef = rt.GetFunctionProto(cf.function_def.signature.name)
        fndef.node_def[0].attr['value'].tensor.int_val[0] = 2
        rt.CreateFunction(fndef)
        self.assertAllEqual(self.evaluate(f()), 2)

    def test_concrete_function_editing_via_mlir_pass_tfg_dialect(self):
        if False:
            print('Hello World!')
        if not tf2.enabled():
            self.skipTest('TF2 test')

        @def_function.function
        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.add(x, y, name='x_plus_y')
        one = constant_op.constant(1)
        cf = f.get_concrete_function(one, one)
        ctx = runtime_client.GlobalPythonEagerContext()
        rt = runtime_client.Runtime(ctx)
        rt.TransformFunction(cf.function_def.signature.name, 'test-pass')
        self.assertAllEqual(self.evaluate(f(one, one)), 1)

    def test_concrete_function_editing_via_mlir_pass_tf_dialect(self):
        if False:
            return 10
        if not tf2.enabled():
            self.skipTest('TF2 test')

        @def_function.function
        def f(x, y):
            if False:
                return 10
            return math_ops.multiply(x, y, name='x_times_y')
        one = constant_op.constant(1)
        cf = f.get_concrete_function(one, one)
        fname = cf.function_def.signature.name
        ctx = runtime_client.GlobalPythonEagerContext()
        rt = runtime_client.Runtime(ctx)
        rt.TransformFunction(fname, 'test-pass-tf-dialect', runtime_client.Runtime.Dialect.TF)
        self.assertAllEqual(f(one, one), 2)

    def test_concrete_function_editing_via_mlir_pass_mixed_dialects(self):
        if False:
            for i in range(10):
                print('nop')
        if not tf2.enabled():
            self.skipTest('TF2 test')

        @def_function.function
        def f(x, y):
            if False:
                return 10
            return math_ops.add(x, y, name='x_plus_y')
        one = constant_op.constant(1)
        cf = f.get_concrete_function(one, one)
        ctx = runtime_client.GlobalPythonEagerContext()
        rt = runtime_client.Runtime(ctx)
        fname = cf.function_def.signature.name
        rt.TransformFunction(fname, 'test-pass')
        self.assertAllEqual(f(one, one), 1)
        rt.TransformFunction(fname, 'test-pass-tf-dialect', runtime_client.Runtime.Dialect.TF)
        self.assertAllEqual(f(one, one), 2)

class RuntimeClientMultiWorkersTest(test.TestCase):

    @test_util.run_v2_only
    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        (workers, _) = test_util.create_local_cluster(2, 0)
        remote.connect_to_remote_host([workers[0].target, workers[1].target])
        self.device0 = '/job:worker/replica:0/task:0/device:CPU:0'
        self.device1 = '/job:worker/replica:0/task:1/device:CPU:0'

    @test_util.run_v2_only
    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        ops.device(None).__enter__()
        context._reset_context()

    @test_util.run_v2_only
    def test_transform_function_in_remote_contexts(self):
        if False:
            return 10
        'Tests if function_defs in remote contexts could be transformed.'

        @def_function.function
        def add(x, y):
            if False:
                return 10
            return math_ops.add(x, y, name='x_plus_y')
        inputs = [1.0, 2.0]
        with ops.device(self.device0):
            result = add(*inputs)
        self.assertAllEqual(result, 3.0)
        self.assertEqual(result.device, self.device0)
        with ops.device(self.device1):
            result = add(*inputs)
        self.assertAllEqual(result, 3.0)
        self.assertEqual(result.device, self.device1)
        cf = add.get_concrete_function(*inputs)
        function_name = cf.function_def.signature.name
        ctx = runtime_client.GlobalPythonEagerContext()
        rt = runtime_client.Runtime(ctx)
        rt.TransformFunction(function_name, 'test-pass')
        fndef = rt.GetFunctionProto(function_name)
        rt.CreateFunction(fndef)
        with ops.device(self.device0):
            result = add(*inputs)
        self.assertAllEqual(result, 2.0)
        self.assertEqual(result.device, self.device0)
        with ops.device(self.device1):
            result = add(*inputs)
        self.assertAllEqual(result, 2.0)
        self.assertEqual(result.device, self.device1)
if __name__ == '__main__':
    context.set_soft_device_placement(False)
    test_pass.RegisterTestPass()
    test.main()