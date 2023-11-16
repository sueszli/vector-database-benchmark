"""Utilities to test TF-TensorRT integration."""
import gc
import os
import re
import tempfile
from unittest import mock
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
from tensorflow.compiler.tf2tensorrt.utils.trt_engine_instance_pb2 import TRTEngineInstance
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.compiler.tensorrt.test import test_utils
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util.lazy_loader import LazyLoader
_SAVED_MODEL_SIGNATURE_KEY = 'mypredict'
gen_trt_ops = LazyLoader('gen_trt_ops', globals(), 'tensorflow.compiler.tf2tensorrt.ops.gen_trt_ops')

class TrtConvertTest(test_util.TensorFlowTestCase, parameterized.TestCase):
    """Class to test Tensorflow-TensorRT integration python API."""
    _TRT_MAX_WORKSPACE_SIZE_BYTES = trt_convert.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES

    def mkdtemp(self):
        if False:
            i = 10
            return i + 15
        return tempfile.mkdtemp(dir=self.get_temp_dir())

    def testTRTEngineInstanceAvailable(self):
        if False:
            for i in range(10):
                print('nop')
        assert hasattr(TRTEngineInstance(), 'serialized_engine')

    def _GetConfigProto(self, rewriter_config=None):
        if False:
            return 10
        'Get ConfigProto for session creation.'
        config = config_pb2.ConfigProto(gpu_options=config_pb2.GPUOptions(allow_growth=True))
        if rewriter_config:
            config.graph_options.rewrite_options.CopyFrom(rewriter_config)
        return config

    @classmethod
    def _GetGraph(cls, inp1, inp2, var):
        if False:
            for i in range(10):
                print('nop')
        'Get the graph for testing.'
        add = inp1 + var
        mul = inp1 * add
        add = mul + add
        add = add + inp2
        out = array_ops.identity(add, name='output')
        return out

    def _GetShapeOpModel(self):
        if False:
            for i in range(10):
                print('nop')

        class ShapeOpModel(autotrackable.AutoTrackable):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.v = None

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[None, None], dtype=dtypes.float32)])
            def run(self, x):
                if False:
                    while True:
                        i = 10
                q = x + 1
                q_shape = array_ops.shape(q)
                q = math_ops.cumsum(q_shape)
                q = q * 2
                return array_ops.identity(q, name='output')
        return ShapeOpModel()

    def _GetModelForV2(self):
        if False:
            while True:
                i = 10

        class SimpleModel(autotrackable.AutoTrackable):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.v = None

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32), tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32)])
            def run(self, inp1, inp2):
                if False:
                    for i in range(10):
                        print('nop')
                if self.v is None:
                    self.v = variables.Variable([[[1.0]]], dtype=dtypes.float32)
                return TrtConvertTest._GetGraph(inp1, inp2, self.v)
        return SimpleModel()

    def _GetGraphForV1(self, device):
        if False:
            while True:
                i = 10

        def _GraphFn():
            if False:
                while True:
                    i = 10
            inp1 = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 1, 1], name='input1')
            inp2 = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 1, 1], name='input2')
            var = variables.Variable([[[1.0]]], dtype=dtypes.float32, name='v1')
            out = TrtConvertTest._GetGraph(inp1, inp2, var)
            return (g, var, inp1, inp2, out)
        g = ops.Graph()
        with g.as_default():
            if device:
                with g.device(device):
                    return _GraphFn()
            return _GraphFn()

    def _GetGraphDefForV1(self, device):
        if False:
            return 10
        'Get the graph def for testing.'
        (g, var, _, _, _) = self._GetGraphForV1(device)
        with self.session(graph=g, config=self._GetConfigProto()) as sess:
            sess.run(var.initializer)
            graph_def = convert_to_constants.convert_variables_to_constants(sess, g.as_graph_def(add_shapes=True), ['output'])
        node_name_to_op = {node.name: node.op for node in graph_def.node}
        self.assertEqual({'v1': 'Const', 'add/ReadVariableOp': 'Identity', 'input1': 'Placeholder', 'input2': 'Placeholder', 'add': 'AddV2', 'mul': 'Mul', 'add_1': 'AddV2', 'add_2': 'AddV2', 'output': 'Identity'}, node_name_to_op)
        return graph_def

    def _WriteInputSavedModelForV1(self, input_saved_model_dir, device):
        if False:
            i = 10
            return i + 15
        'Write the saved model as an input for testing.'
        (g, var, inp1, inp2, out) = self._GetGraphForV1(device)
        signature_def = signature_def_utils.build_signature_def(inputs={'myinput1': utils.build_tensor_info(inp1), 'myinput2': utils.build_tensor_info(inp2)}, outputs={'myoutput': utils.build_tensor_info(out)}, method_name=signature_constants.PREDICT_METHOD_NAME)
        saved_model_builder = builder.SavedModelBuilder(input_saved_model_dir)
        with self.session(graph=g, config=self._GetConfigProto()) as sess:
            sess.run(var.initializer)
            saved_model_builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map={_SAVED_MODEL_SIGNATURE_KEY: signature_def})
        saved_model_builder.save()

    def _ConvertGraphV1(self, output_saved_model_dir=None, need_calibration=False, max_batch_size=1, minimum_segment_size=3, is_dynamic_op=False, maximum_cached_engines=1, device=None):
        if False:
            i = 10
            return i + 15
        'Helper method to convert a GraphDef or SavedModel using TF-TRT.'
        input_saved_model_dir = None
        if output_saved_model_dir:
            input_saved_model_dir = self.mkdtemp()
            self._WriteInputSavedModelForV1(input_saved_model_dir, device)
        if need_calibration:
            is_dynamic_op = True
        if is_dynamic_op:
            max_batch_size = None
        converter = trt_convert.TrtGraphConverter(input_saved_model_dir=input_saved_model_dir, input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY, input_graph_def=None if input_saved_model_dir else self._GetGraphDefForV1(device), nodes_denylist=None if input_saved_model_dir else ['output'], max_batch_size=max_batch_size, max_workspace_size_bytes=TrtConvertTest._TRT_MAX_WORKSPACE_SIZE_BYTES, precision_mode=trt_convert.TrtPrecisionMode.INT8 if need_calibration else trt_convert.TrtPrecisionMode.FP32, minimum_segment_size=minimum_segment_size, is_dynamic_op=is_dynamic_op, maximum_cached_engines=maximum_cached_engines)
        output_graph_def = converter.convert()
        if need_calibration:

            class CalibrationData(object):

                def __init__(self):
                    if False:
                        return 10
                    self._data = 0

                def next(self):
                    if False:
                        print('Hello World!')
                    self._data += 1
                    return {'input1:0': [[[self._data]]], 'input2:0': [[[self._data]]]}
            output_graph_def = converter.calibrate(fetch_names=['output:0'], num_runs=10, feed_dict_fn=CalibrationData().next)
        if output_saved_model_dir is not None:
            converter.save(output_saved_model_dir=output_saved_model_dir)
        return output_graph_def

    def _MayRemoveGraphSequenceNumber(self, name):
        if False:
            i = 10
            return i + 15
        prefix = re.search('TRTEngineOp_\\d{3,}_', name)
        if prefix and name.startswith(prefix.group(0)):
            parts = name.split('_', maxsplit=2)
            assert len(parts) == 3
            return parts[0] + '_' + parts[2]
        return name

    def _GetUniqueTRTEngineOp(self, graph_def):
        if False:
            i = 10
            return i + 15
        trt_engine_nodes = [node for node in graph_def.node if node.op == 'TRTEngineOp']
        assert len(trt_engine_nodes) == 1
        return trt_engine_nodes[0]

    def _TestTrtGraphConverter(self, device, output_saved_model_dir=None, need_calibration=False, is_dynamic_op=False):
        if False:
            print('Hello World!')
        'General method to test trt_convert.TrtGraphConverter().'
        output_graph_def = self._ConvertGraphV1(output_saved_model_dir=output_saved_model_dir, need_calibration=need_calibration, is_dynamic_op=is_dynamic_op, device=device)
        graph_defs_to_verify = [output_graph_def]
        if output_saved_model_dir:
            saved_model_graph_def = saved_model_utils.get_meta_graph_def(output_saved_model_dir, tag_constants.SERVING).graph_def
            self.assertIsInstance(saved_model_graph_def, graph_pb2.GraphDef)
            graph_defs_to_verify.append(saved_model_graph_def)
        for graph_def in graph_defs_to_verify:
            node_name_to_op = {self._MayRemoveGraphSequenceNumber(node.name): node.op for node in graph_def.node}
            if device is not None and device.startswith('/CPU:'):
                self.assertEqual({'add': 'AddV2', 'v1': 'Const', 'add_1': 'AddV2', 'add_2': 'AddV2', 'input1': 'Placeholder', 'input2': 'Placeholder', 'mul': 'Mul', 'output': 'Identity'}, node_name_to_op)
            else:
                self.assertEqual({'input1': 'Placeholder', 'input2': 'Placeholder', 'TRTEngineOp_000': 'TRTEngineOp', 'output': 'Identity'}, node_name_to_op)
            if need_calibration:
                trt_engine_nodes = [node for node in graph_def.node if node.op == 'TRTEngineOp']
                if device is not None and device.startswith('/CPU:'):
                    self.assertEmpty(trt_engine_nodes)
                    return
                self.assertNotEmpty(trt_engine_nodes)
                for node in trt_engine_nodes:
                    self.assertTrue(len(node.attr['calibration_data'].s))
                with ops.Graph().as_default():
                    importer.import_graph_def(graph_def, name='')
                    with self.session(config=self._GetConfigProto()) as sess:
                        for test_data in range(10):
                            self.assertEqual((test_data + 1.0) ** 2 + test_data, sess.run('output:0', feed_dict={'input1:0': [[[test_data]]], 'input2:0': [[[test_data]]]}))

    @parameterized.named_parameters([('NoDeviceAssignment', None), ('GPU', '/GPU:0'), ('CPU', '/CPU:0')])
    @test_util.deprecated_graph_mode_only
    def testTrtGraphConverter_OfflineConversion(self, device):
        if False:
            while True:
                i = 10
        'Test case for trt_convert.TrtGraphConverter().'
        for need_calibration in [False, True]:
            self._TestTrtGraphConverter(device)
            self._TestTrtGraphConverter(device, output_saved_model_dir=self.mkdtemp(), need_calibration=need_calibration)

    @parameterized.named_parameters([('NoDeviceAssignment', None), ('GPU', '/device:GPU:0'), ('CPU', '/device:CPU:0')])
    @test_util.deprecated_graph_mode_only
    def testTrtGraphConverter_OnlineConversion(self, device):
        if False:
            print('Hello World!')
        'Test case for TF-TRT conversion using Grappler directly.'
        conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt_convert.TrtPrecisionMode.FP32)
        config = self._GetConfigProto(rewriter_config=trt_convert.get_tensorrt_rewriter_config(conversion_params, is_dynamic_op=False, max_batch_size=1, is_v2=False))
        with ops.Graph().as_default():
            inp1 = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 1, 1], name='input1')
            inp2 = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 1, 1], name='input2')
            if device:
                with ops.device(device):
                    TrtConvertTest._GetGraph(inp1, inp2, inp1)
            else:
                TrtConvertTest._GetGraph(inp1, inp2, inp1)
            with self.session(config=config) as sess:
                self._TestRun(sess, batch_size=1)

    def _CreateConverterV2(self, input_saved_model_dir, input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY, max_workspace_size_bytes=10 << 20, precision_mode=trt_convert.TrtPrecisionMode.FP32, maximum_cached_engines=2, allow_build_at_runtime=True):
        if False:
            while True:
                i = 10
        return trt_convert.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir, input_saved_model_signature_key=input_saved_model_signature_key, max_workspace_size_bytes=max_workspace_size_bytes, precision_mode=precision_mode, maximum_cached_engines=maximum_cached_engines, allow_build_at_runtime=allow_build_at_runtime)

    def _CheckTrtOps(self, concrete_func, check_fn=None, num_engines=1):
        if False:
            return 10
        graph_def = concrete_func.graph.as_graph_def()
        trt_op_names = []
        for node in graph_def.node:
            if node.op == 'TRTEngineOp':
                trt_op_names.append(self._MayRemoveGraphSequenceNumber(node.name))
                if check_fn:
                    check_fn(node)
        for func in graph_def.library.function:
            for node in func.node_def:
                if node.op == 'TRTEngineOp':
                    trt_op_names.append(self._MayRemoveGraphSequenceNumber(node.name))
                    if check_fn:
                        check_fn(node)
        self.assertLen(trt_op_names, num_engines)

    def _RandomInput(self, shape, dtype=np.float32):
        if False:
            return 10
        inp1 = np.random.random_sample(shape).astype(dtype)
        inp2 = np.random.random_sample(shape).astype(dtype)
        return (inp1, inp2)

    def _GetAssetFile(self, output_saved_model_dir, trt_engine_name):
        if False:
            i = 10
            return i + 15
        asset_file = os.path.join(output_saved_model_dir, 'assets/trt-serialized-engine.' + trt_engine_name)
        return asset_file

    def _BuildGraphWithInputGenerator(self, InputFunc, np_input=None):
        if False:
            print('Hello World!')
        root = self._GetShapeOpModel()
        expected_output = None if np_input is None else root.run(np_input)
        input_saved_model_dir = self.mkdtemp()
        save.save(root, input_saved_model_dir, signatures=root.run)
        conv_params = trt_convert.TrtConversionParams(minimum_segment_size=2)
        converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir, use_dynamic_shape=True, **conv_params._asdict())
        converter.convert()
        converter.build(InputFunc)
        output_saved_model_dir = self.mkdtemp()
        converter.save(output_saved_model_dir)
        del converter
        return (output_saved_model_dir, expected_output)

    def _BuildGraphWithInputGeneratorTwoInputs(self, InputFunc, np_input=None):
        if False:
            i = 10
            return i + 15
        input_saved_model_dir = self.mkdtemp()
        root = self._GetModelForV2()
        if np_input is None:
            expected_output = None
        else:
            expected_output = root.run(np_input[0], np_input[1])
        save.save(root, input_saved_model_dir, {_SAVED_MODEL_SIGNATURE_KEY: root.run})
        converter = self._CreateConverterV2(input_saved_model_dir)
        converter.convert()
        self._CheckTrtOps(converter._converted_func)
        trt_engine_name = self._GetUniqueTRTEngineOp(converter._converted_graph_def).name
        output_saved_model_dir = self.mkdtemp()
        converter.save(output_saved_model_dir)
        unexpected_asset_file = self._GetAssetFile(output_saved_model_dir, trt_engine_name)
        self.assertFalse(os.path.exists(unexpected_asset_file))
        converter.build(input_fn=InputFunc)
        output_saved_model_dir = self.mkdtemp()
        converter.save(output_saved_model_dir)
        expected_asset_file = self._GetAssetFile(output_saved_model_dir, trt_engine_name)
        self.assertTrue(os.path.exists(expected_asset_file))
        self.assertTrue(os.path.getsize(expected_asset_file))
        del converter
        return (output_saved_model_dir, expected_output)

    @test_util.run_v2_only
    def testTrtGraphBuild(self):
        if False:
            while True:
                i = 10
        'Testing the construction of a graph with an input data generator\n\n       that takes one or two input parameters passed in different formats.\n    '
        np_input = np.random.random_sample([5, 3]).astype(np.float32)

        def _Func_1():
            if False:
                while True:
                    i = 10
            yield (np_input,)

        def _Func_2():
            if False:
                while True:
                    i = 10
            yield [np_input]

        def _Func_3():
            if False:
                print('Hello World!')
            yield np_input

        def _Func_4():
            if False:
                print('Hello World!')
            yield {'x': np_input}

        def _Func_5():
            if False:
                print('Hello World!')
            yield np_input
            yield [np_input]
            yield np_input
            yield {'x': np_input}

        def _Func_6():
            if False:
                for i in range(10):
                    print('nop')
            for shape in [(1, 128), (16, 128), (256, 128)]:
                yield np.random.random_sample(shape).astype(np.float32)
        for input_fn in [_Func_1, _Func_2, _Func_3, _Func_4, _Func_5, _Func_6]:
            self._BuildGraphWithInputGenerator(input_fn)
        (np_input1, np_input2) = self._RandomInput([4, 1, 1])

        def _Func_A():
            if False:
                for i in range(10):
                    print('nop')
            yield (np_input1, np_input2)

        def _Func_B():
            if False:
                while True:
                    i = 10
            yield [np_input1, np_input2]

        def _Func_C():
            if False:
                for i in range(10):
                    print('nop')
            yield {'inp1': np_input1, 'inp2': np_input2}

        def _Func_D():
            if False:
                print('Hello World!')
            yield (np_input1, np_input2)
            yield [np_input1, np_input2]
            yield {'inp1': np_input1, 'inp2': np_input2}

        def _Func_E():
            if False:
                while True:
                    i = 10
            for shape in [[4, 1, 1], [4, 2, 1], [4, 4, 1]]:
                yield self._RandomInput(shape)
        for input_fn in [_Func_A, _Func_B, _Func_C, _Func_D, _Func_E]:
            self._BuildGraphWithInputGeneratorTwoInputs(input_fn)

    @test_util.run_v2_only
    def testTrtGraphConverter_DynamicConversion_v2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test case for trt_convert.TrtGraphConverter().'
        (np_input1, np_input2) = self._RandomInput([4, 1, 1])

        def _InputFn():
            if False:
                print('Hello World!')
            yield (np_input1, np_input2)
        np_inputs = [np_input1, np_input2]
        (output_saved_model_dir, expected_output) = self._BuildGraphWithInputGeneratorTwoInputs(_InputFn, np_inputs)
        gc.collect()
        root_with_trt = load.load(output_saved_model_dir)
        converted_signature = root_with_trt.signatures[_SAVED_MODEL_SIGNATURE_KEY]
        self._CheckTrtOps(converted_signature)
        output_with_trt = converted_signature(inp1=ops.convert_to_tensor(np_input1), inp2=ops.convert_to_tensor(np_input2))
        self.assertAllClose(expected_output, list(output_with_trt.values())[0], atol=1e-06, rtol=1e-06)
        del root_with_trt
        gc.collect()

    @test_util.run_v2_only
    def testTrtGraphConverter_ShapeOp_Int32InputOutput_v2(self):
        if False:
            for i in range(10):
                print('nop')
        'Testing ShapeOp and int32 values as engine input and output.'
        np_input = np.random.random_sample([5, 3]).astype(np.float32)

        def _InputFunc():
            if False:
                return 10
            yield (np_input,)
        (output_saved_model_dir, expected_output) = self._BuildGraphWithInputGenerator(_InputFunc, np_input)
        root_with_trt = load.load(output_saved_model_dir)
        converted_signature = root_with_trt.signatures['serving_default']
        self._CheckTrtOps(converted_signature, num_engines=2)
        output_with_trt = converted_signature(x=ops.convert_to_tensor(np_input))
        self.assertAllClose(expected_output, list(output_with_trt.values())[0])

    @test_util.run_v2_only
    def testTrtGraphConverter_Int8Conversion_v2(self):
        if False:
            print('Hello World!')
        (np_input1, np_input2) = self._RandomInput([4, 1, 1])
        input_saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
        root = self._GetModelForV2()
        expected_output = root.run(np_input1, np_input2)
        save.save(root, input_saved_model_dir, {_SAVED_MODEL_SIGNATURE_KEY: root.run})
        converter = self._CreateConverterV2(input_saved_model_dir, precision_mode=trt_convert.TrtPrecisionMode.INT8, maximum_cached_engines=3)

        def _CalibrationInputFn():
            if False:
                i = 10
                return i + 15
            yield (np_input1, np_input2)
        converter.convert(calibration_input_fn=_CalibrationInputFn)
        trt_engine_name = self._GetUniqueTRTEngineOp(converter._converted_graph_def).name

        def _CheckFn(node):
            if False:
                return 10
            self.assertTrue(len(node.attr['calibration_data'].s), node.name)
        self._CheckTrtOps(converter._converted_func, _CheckFn)

        def _InputFn():
            if False:
                for i in range(10):
                    print('nop')
            yield self._RandomInput([5, 1, 1])
        converter.build(input_fn=_InputFn)
        output_saved_model_dir = self.mkdtemp()
        converter.save(output_saved_model_dir)
        expected_asset_file = self._GetAssetFile(output_saved_model_dir, trt_engine_name)
        self.assertTrue(os.path.exists(expected_asset_file))
        self.assertTrue(os.path.getsize(expected_asset_file))
        del converter
        gc.collect()
        root_with_trt = load.load(output_saved_model_dir)
        converted_signature = root_with_trt.signatures[_SAVED_MODEL_SIGNATURE_KEY]
        self._CheckTrtOps(converted_signature, _CheckFn)
        output_with_trt = converted_signature(inp1=ops.convert_to_tensor(np_input1), inp2=ops.convert_to_tensor(np_input2))
        self.assertEqual(1, len(output_with_trt))
        self.assertAllClose(expected_output, list(output_with_trt.values())[0], atol=1e-06, rtol=1e-06)
        (np_input1, np_input2) = self._RandomInput([6, 1, 1])
        converted_signature(inp1=ops.convert_to_tensor(np_input1), inp2=ops.convert_to_tensor(np_input2))
        del root_with_trt
        gc.collect()

    @test_util.run_v2_only
    def testTrtGraphConverter_RemoveNativeSegments(self):
        if False:
            for i in range(10):
                print('nop')
        'Test case for trt_convert._remove_native_segment().'
        np_input = np.random.random_sample([5, 3]).astype(np.float32)
        input_saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
        root = self._GetShapeOpModel()
        expected_output = root.run(np_input)
        save.save(root, input_saved_model_dir, signatures=root.run)
        converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir, precision_mode=trt_convert.TrtPrecisionMode.FP32, allow_build_at_runtime=False, minimum_segment_size=1)

        def _input_fn():
            if False:
                i = 10
                return i + 15
            yield (np_input,)
        graph_func = converter.convert()
        converter.build(_input_fn)
        output_saved_model_dir2 = self.mkdtemp()
        with test_utils.experimental_feature_scope('remove_native_segments'):
            converter.save(output_saved_model_dir2)
        saved_model_loaded = load.load(output_saved_model_dir2)
        graph_func_after = saved_model_loaded.signatures['serving_default']
        actual_output = graph_func_after(x=np_input)['output_0']
        self.assertAllClose(expected_output, actual_output, atol=1e-06, rtol=1e-06)
        del graph_func
        del root
        gc.collect()

    @test_util.run_v2_only
    def testTrtGraphConverter_DestroyEngineCache(self):
        if False:
            while True:
                i = 10
        'Test case for trt_convert.TrtGraphConverter().'
        (np_input1, np_input2) = self._RandomInput([4, 1, 1])
        input_saved_model_dir = self.mkdtemp()
        root = self._GetModelForV2()
        save.save(root, input_saved_model_dir, {_SAVED_MODEL_SIGNATURE_KEY: root.run})
        converter = self._CreateConverterV2(input_saved_model_dir)
        converter.convert()
        trt_engine_name = self._GetUniqueTRTEngineOp(converter._converted_graph_def).name

        def _InputFn():
            if False:
                print('Hello World!')
            yield (np_input1, np_input2)
        converter.build(input_fn=_InputFn)
        output_saved_model_dir = self.mkdtemp()
        converter.save(output_saved_model_dir)

        def _DestroyCache():
            if False:
                i = 10
                return i + 15
            with ops.device('GPU:0'):
                handle = gen_trt_ops.create_trt_resource_handle(resource_name=trt_engine_name)
                gen_resource_variable_ops.destroy_resource_op(handle, ignore_lookup_error=False)
        with self.assertRaisesRegex(errors.NotFoundError, 'Resource .* does not exist.'):
            _DestroyCache()
        root = load.load(output_saved_model_dir)
        _DestroyCache()
        with self.assertRaisesRegex(errors.NotFoundError, 'Resource .* does not exist.'):
            _DestroyCache()
        root = load.load(output_saved_model_dir)
        del root
        gc.collect()
        with self.assertRaisesRegex(errors.NotFoundError, 'Resource .* does not exist.'):
            _DestroyCache()

    def _CompareSavedModel(self, model_class):
        if False:
            for i in range(10):
                print('nop')
        signature_key = 'serving_default'

        def _GetModelPaths(model_class):
            if False:
                for i in range(10):
                    print('nop')
            input_saved_model_dir = self.mkdtemp()
            root = model_class()
            save.save(root, input_saved_model_dir)
            converter = self._CreateConverterV2(input_saved_model_dir, input_saved_model_signature_key=signature_key)
            converter.convert()
            output_saved_model_dir = self.mkdtemp()
            converter.save(output_saved_model_dir)
            return (input_saved_model_dir, output_saved_model_dir)

        def _GetSignatureDef(export_dir):
            if False:
                i = 10
                return i + 15
            saved_model_proto = loader_impl.parse_saved_model(export_dir)
            self.assertEqual(1, len(saved_model_proto.meta_graphs))
            meta_graph = saved_model_proto.meta_graphs[0]
            self.assertIn(signature_key, meta_graph.signature_def)
            return meta_graph.signature_def[signature_key]

        def _CompareSignatureDef(original_def, converted_def, is_input):
            if False:
                while True:
                    i = 10
            endpoints = original_def.inputs if is_input else original_def.outputs
            converted_endpoints = converted_def.inputs if is_input else converted_def.outputs
            self.assertEqual(set(endpoints.keys()), set(converted_endpoints.keys()))
            for key in endpoints:
                original_input = endpoints[key]
                converted_input = converted_endpoints[key]
                self.assertEqual(original_input.name, converted_input.name)
                self.assertEqual(original_input.dtype, converted_input.dtype)
                self.assertEqual(tensor_shape.TensorShape(original_input.tensor_shape).as_list(), tensor_shape.TensorShape(converted_input.tensor_shape).as_list())

        def _GetStructuredOutputs(export_dir):
            if False:
                print('Hello World!')
            root = load.load(export_dir)
            return root.signatures[signature_key].structured_outputs
        (saved_model_path, converted_saved_model_path) = _GetModelPaths(model_class)
        original_def = _GetSignatureDef(saved_model_path)
        converted_def = _GetSignatureDef(converted_saved_model_path)
        self.assertEqual(original_def.method_name, converted_def.method_name)
        _CompareSignatureDef(original_def, converted_def, True)
        _CompareSignatureDef(original_def, converted_def, False)
        self.assertEqual(_GetStructuredOutputs(saved_model_path), _GetStructuredOutputs(converted_saved_model_path))

    @test_util.run_v2_only
    def testRetainSignatureInfo_NoInputs(self):
        if False:
            return 10

        class _Model(autotrackable.AutoTrackable):

            @def_function.function(input_signature=[])
            def run(self):
                if False:
                    while True:
                        i = 10
                return array_ops.constant(1.0)
        self._CompareSavedModel(_Model)

    @test_util.run_v2_only
    def testRetainSignatureInfo_OneInput(self):
        if False:
            while True:
                i = 10

        class _Model(autotrackable.AutoTrackable):

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32)])
            def run(self, inp):
                if False:
                    while True:
                        i = 10
                return inp + inp * inp
        self._CompareSavedModel(_Model)

    @test_util.run_v2_only
    def testRetainSignatureInfo_TwoInputs(self):
        if False:
            i = 10
            return i + 15

        class _Model(autotrackable.AutoTrackable):

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32), tensor_spec.TensorSpec(shape=[None, 2], dtype=dtypes.float32)])
            def run(self, inp1, inp2):
                if False:
                    for i in range(10):
                        print('nop')
                return inp1 + inp2 * inp2
        self._CompareSavedModel(_Model)

    @test_util.run_v2_only
    def testRetainSignatureInfo_OneOutputSignatureKey(self):
        if False:
            print('Hello World!')

        class _Model(autotrackable.AutoTrackable):

            @def_function.function(input_signature=[])
            def run(self):
                if False:
                    while True:
                        i = 10
                return {'my_output': array_ops.constant(1.0)}
        self._CompareSavedModel(_Model)

    @test_util.run_v2_only
    def testRetainSignatureInfo_TwoOutputSignatureKeys(self):
        if False:
            for i in range(10):
                print('nop')

        class _Model(autotrackable.AutoTrackable):

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32)])
            def run(self, inp):
                if False:
                    while True:
                        i = 10
                return {'output_b': array_ops.constant(1.0), 'output_a': inp + inp * inp}
        self._CompareSavedModel(_Model)

    def _TestRun(self, sess, batch_size):
        if False:
            print('Hello World!')
        result = sess.run('output:0', feed_dict={'input1:0': [[[1.0]]] * batch_size, 'input2:0': [[[1.0]]] * batch_size})
        self.assertAllEqual([[[5.0]]] * batch_size, result)

    @parameterized.named_parameters([('LargeSegmentSize', 7), ('NoMainGraphConversionSegmentSize', -1)])
    @test_util.deprecated_graph_mode_only
    def testTrtGraphConverter_MinimumSegmentSize(self, minimum_segment_size):
        if False:
            for i in range(10):
                print('nop')
        output_graph_def = self._ConvertGraphV1(minimum_segment_size=minimum_segment_size)
        node_name_to_op = {node.name: node.op for node in output_graph_def.node}
        self.assertEqual({'v1': 'Const', 'input1': 'Placeholder', 'input2': 'Placeholder', 'add': 'AddV2', 'mul': 'Mul', 'add_1': 'AddV2', 'add_2': 'AddV2', 'output': 'Identity'}, node_name_to_op)

    @test_util.deprecated_graph_mode_only
    def testTrtGraphConverter_DynamicOp(self):
        if False:
            return 10
        output_saved_model_dir = self.mkdtemp()
        output_graph_def = self._ConvertGraphV1(output_saved_model_dir=output_saved_model_dir, is_dynamic_op=True, maximum_cached_engines=2)
        with ops.Graph().as_default():
            importer.import_graph_def(output_graph_def, name='')
            with self.session(config=self._GetConfigProto()) as sess:
                self._TestRun(sess, 1)
                self._TestRun(sess, 2)
                self._TestRun(sess, 3)
        with ops.Graph().as_default():
            with self.session(config=self._GetConfigProto()) as sess:
                loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
                self._TestRun(sess, 1)
                self._TestRun(sess, 2)
                self._TestRun(sess, 3)

    @test_util.deprecated_graph_mode_only
    def testTrtGraphConverter_StaticOp(self):
        if False:
            for i in range(10):
                print('nop')
        output_saved_model_dir = self.mkdtemp()
        output_graph_def = self._ConvertGraphV1(output_saved_model_dir=output_saved_model_dir, maximum_cached_engines=1)
        with ops.Graph().as_default():
            importer.import_graph_def(output_graph_def, name='')
            with self.session(config=self._GetConfigProto()) as sess:
                self._TestRun(sess, 1)
                self._TestRun(sess, 2)
        with ops.Graph().as_default():
            with self.session(config=self._GetConfigProto()) as sess:
                loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
                self._TestRun(sess, 1)
                self._TestRun(sess, 2)

    @test_util.run_v2_only
    def testTrtGraphConverter_AllowEngineNativeSegmentExecution(self):
        if False:
            for i in range(10):
                print('nop')
        (np_input1, np_input2) = self._RandomInput([4, 1, 1])
        input_saved_model_dir = self.mkdtemp()
        root = self._GetModelForV2()
        save.save(root, input_saved_model_dir, {_SAVED_MODEL_SIGNATURE_KEY: root.run})

        def _InputFn():
            if False:
                for i in range(10):
                    print('nop')
            yield (np_input1, np_input2)
        converter = self._CreateConverterV2(input_saved_model_dir, max_workspace_size_bytes=1 << 20)
        converter.convert()
        os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'False'
        os.environ['TF_TRT_ABORT_CUDA_ENGINE_BUILD'] = 'True'
        with self.assertRaisesRegex(errors.AbortedError, 'User disallowed engine native segment execution'):
            try:
                converter.build(input_fn=_InputFn)
            finally:
                os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'True'
                os.environ['TF_TRT_ABORT_CUDA_ENGINE_BUILD'] = 'False'
        converter.build(input_fn=_InputFn)

    @parameterized.parameters((True, True), (True, False), (False, True), (False, False))
    @test_util.run_v2_only
    def testTrtGraphConverter_AllowBuildAtRuntime(self, build_offline, allow_build_at_runtime):
        if False:
            return 10
        if not is_tensorrt_enabled():
            return
        input_saved_model_dir = self.mkdtemp()
        root = self._GetModelForV2()
        save.save(root, input_saved_model_dir, {_SAVED_MODEL_SIGNATURE_KEY: root.run})
        np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
        np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))

        def _InputFn():
            if False:
                for i in range(10):
                    print('nop')
            yield (np_input1, np_input2)
        converter = self._CreateConverterV2(input_saved_model_dir, allow_build_at_runtime=allow_build_at_runtime)
        converter.convert()
        if build_offline:
            converter.build(input_fn=_InputFn)
        output_saved_model_dir = self.mkdtemp()
        converter.save(output_saved_model_dir)
        saved_model_loaded = load.load(output_saved_model_dir, tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[_SAVED_MODEL_SIGNATURE_KEY]

        def _CheckFn(node):
            if False:
                return 10
            self.assertEqual(node.attr['_allow_build_at_runtime'].b, allow_build_at_runtime)
        self._CheckTrtOps(graph_func, _CheckFn)
        if not build_offline and (not allow_build_at_runtime):
            with self.assertRaisesRegex(errors.AbortedError, 'User disallowed engine native segment execution'):
                try:
                    os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'False'
                    graph_func(inp1=np_input1, inp2=np_input2)
                finally:
                    os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'True'
        else:
            output = graph_func(inp1=np_input1, inp2=np_input2)['output_0']
            self.assertEqual(output.shape, (4, 1, 1))
            self.assertAllClose(np.asarray([5.0, 5.0, 5.0, 5.0]).reshape([4, 1, 1]), output)

    @test_util.run_v2_only
    def testBackwardCompatibility(self):
        if False:
            i = 10
            return i + 15
        'Load and execute a model that was saved in TF2.0.'
        model_dir = test.test_src_dir_path('python/compiler/tensorrt/test/testdata/tftrt_2.0_saved_model')
        saved_model_loaded = load.load(model_dir, tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
        np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
        output = graph_func(input1=np_input1, input2=np_input2)['output_0']
        self.assertEqual(output.shape, (4, 1, 1))
        self.assertAllClose(np.asarray([5.0, 5.0, 5.0, 5.0]).reshape([4, 1, 1]), output)

    @parameterized.named_parameters([('SaveGPUSpecificEngine', True), ('WithoutSaveGPUSpecificEngine', False)])
    @test_util.run_v2_only
    def testTrtGraphConverter_SaveGPUSpecificEngine(self, save_engine_flag):
        if False:
            while True:
                i = 10
        'Test case for trt_convert.TrtGraphConverter().'
        (np_input1, np_input2) = self._RandomInput([4, 1, 1])
        input_saved_model_dir = self.mkdtemp()
        root = self._GetModelForV2()
        save.save(root, input_saved_model_dir, {_SAVED_MODEL_SIGNATURE_KEY: root.run})
        converter = self._CreateConverterV2(input_saved_model_dir, precision_mode=trt_convert.TrtPrecisionMode.INT8)

        def CalibrationFn():
            if False:
                while True:
                    i = 10
            yield (np_input1, np_input2)
        converter.convert(calibration_input_fn=CalibrationFn)
        self._CheckTrtOps(converter._converted_func)
        trt_engine_name = self._GetUniqueTRTEngineOp(converter._converted_graph_def).name
        output_saved_model_dir = self.mkdtemp()
        converter.save(output_saved_model_dir, save_gpu_specific_engines=save_engine_flag)
        expected_asset_file = self._GetAssetFile(output_saved_model_dir, trt_engine_name)
        self.assertTrue(os.path.exists(expected_asset_file))
        if save_engine_flag:
            self.assertTrue(os.path.getsize(expected_asset_file))
        else:
            self.assertFalse(os.path.getsize(expected_asset_file))
        del converter
        gc.collect()

    @test_util.run_v2_only
    def testTrtGraphConverterV2_SaveWithOptions(self):
        if False:
            for i in range(10):
                print('nop')
        'Test to make sure that save method respects options kwarg.'
        input_saved_model_dir = self.mkdtemp()
        root = self._GetModelForV2()
        save.save(root, input_saved_model_dir, {_SAVED_MODEL_SIGNATURE_KEY: root.run})
        converter = self._CreateConverterV2(input_saved_model_dir)
        converter.convert()
        with mock.patch.object(trt_convert, 'save') as mock_save:
            mock_save.save = mock.MagicMock()
            output_saved_model_dir = self.mkdtemp()
            options = save_options.SaveOptions(save_debug_info=True)
            converter.save(output_saved_model_dir, options=options)
            mock_save.save.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY, options=options)

    @parameterized.named_parameters([('NoDeviceAssignment', None), ('GPU1', 'GPU:1')])
    @test_util.run_v2_only
    def testTrtGraphConverter_DevicePlacement(self, device_id):
        if False:
            print('Hello World!')
        'Test case for trt_convert.TrtGraphConverter().'
        gpus = config.list_physical_devices('GPU')
        if len(gpus) < 2:
            self.skipTest('Expected at least 2 GPUs but found {} GPUs'.format(len(gpus)))
        np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
        np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
        input_saved_model_dir = self.mkdtemp()
        root = self._GetModelForV2()
        save.save(root, input_saved_model_dir, {_SAVED_MODEL_SIGNATURE_KEY: root.run})
        converter = self._CreateConverterV2(input_saved_model_dir, precision_mode=trt_convert.TrtPrecisionMode.FP32)
        converted_model = None
        with ops.device(device_id):
            converted_model = converter.convert()
        self._CheckTrtOps(converter._converted_func)
        actual_device_id = self._GetUniqueTRTEngineOp(converter._converted_graph_def).device
        expected_device_id = None
        if device_id is not None:
            expected_device_id = device_id
        else:
            expected_device_id = 'GPU:0'
        self.assertTrue(expected_device_id.lower() in actual_device_id.lower())
        del converter
        gc.collect()

    @test_util.run_v2_only
    def testTrtGraphConverter_DevicePlacementOnCPU(self):
        if False:
            while True:
                i = 10
        'Test case for trt_convert.TrtGraphConverter().'
        np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
        np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
        input_saved_model_dir = self.mkdtemp()
        root = self._GetModelForV2()
        save.save(root, input_saved_model_dir, {_SAVED_MODEL_SIGNATURE_KEY: root.run})
        converter = self._CreateConverterV2(input_saved_model_dir, precision_mode=trt_convert.TrtPrecisionMode.FP32)
        converted_model = None
        with self.assertRaisesRegex(ValueError, 'Specified device is not a GPU'):
            with ops.device('CPU'):
                converted_model = converter.convert()
        del converter
        gc.collect()

    def _TestVariableHelper(self, variable_op, tf_model_name, tftrt_model_name, output_name):
        if False:
            print('Hello World!')
        'Helper with the common code of variable converter tests.'
        model_dir = test.test_src_dir_path('python/compiler/tensorrt/test/testdata/' + tf_model_name)
        trt_model_dir = os.path.join(self.mkdtemp(), tftrt_model_name)
        conv_params = trt_convert.TrtConversionParams(precision_mode='FP16', minimum_segment_size=3, max_workspace_size_bytes=10 << 20, maximum_cached_engines=1)
        with test_utils.experimental_feature_scope('disable_graph_freezing'):
            converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir=model_dir, conversion_params=conv_params, use_dynamic_shape=True, dynamic_shape_profile_strategy='Optimal')
        converter.convert()
        input_shapes = [[(4, 1, 1), (4, 1, 1)]]

        def _InputFn():
            if False:
                print('Hello World!')
            for shapes in input_shapes:
                yield [np.ones(shape=shape).astype(np.float32) for shape in shapes]
        converter.build(_InputFn)
        converter.save(trt_model_dir)
        saved_model_loaded = load.load(trt_model_dir, tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        graph_def = graph_func.graph.as_graph_def()
        engines = []
        for lib_function in graph_def.library.function:
            if re.search('TRTEngineOp_\\d+_\\d+_native_segment', lib_function.signature.name):
                node_ops = [node.op for node in lib_function.node_def]
                engines.append(node_ops)
        self.assertLen(engines, 1)
        self.assertEqual(engines[0].count(variable_op), 2)
        np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
        np_input2 = ops.convert_to_tensor(2.0 * np.ones([4, 1, 1]).astype(np.float32))
        output = graph_func(input1=np_input1, input2=np_input2)[output_name]
        self.assertEqual(output.shape, (4, 1, 1))
        self.assertAllClose(np.asarray([42.0, 42.0, 42.0, 42.0]).reshape([4, 1, 1]), output)

    @test_util.run_v2_only
    def testVariableV2(self):
        if False:
            return 10
        'Test conversion of VariableV2 nodes.'
        self._TestVariableHelper('VariableV2', 'tf_variablev2_saved_model', 'tftrt_variablev2_saved_model', 'output')

    @test_util.run_v2_only
    def testReadVariableOp(self):
        if False:
            i = 10
            return i + 15
        'Test conversion of ReadVariableOp nodes.'
        self._TestVariableHelper('ReadVariableOp', 'tf_readvariableop_saved_model', 'tftrt_readvariableop_saved_model', 'output_0')
if __name__ == '__main__' and is_tensorrt_enabled():
    test.main()