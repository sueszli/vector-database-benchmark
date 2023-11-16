"""Unit tests for tensorflow_hub.module."""
import tensorflow as tf
from tensorflow_hub import module
from tensorflow_hub import module_impl
from tensorflow_hub import module_spec
from tensorflow_hub import tensor_info

class TestConvertInputsOutputs(tf.test.TestCase):

    def testSingleInput(self):
        if False:
            return 10
        inputs_info = {'x': tensor_info.ParsedTensorInfo(tf.float32, tf.TensorShape([None]), is_sparse=False)}

        def _check(dict_inputs):
            if False:
                i = 10
                return i + 15
            self.assertEqual(len(dict_inputs), 1)
            self.assertEqual(dict_inputs['x'].dtype, tf.float32)
            self.assertTrue(dict_inputs['x'].shape.is_compatible_with([None]))
        _check(module._convert_dict_inputs([1, 2], inputs_info))
        _check(module._convert_dict_inputs({'x': [1, 2]}, inputs_info))
        with self.assertRaisesRegexp(TypeError, "missing \\['x'\\]"):
            module._convert_dict_inputs(None, inputs_info)
        with self.assertRaisesRegexp(TypeError, "extra given \\['y'\\]"):
            module._convert_dict_inputs({'x': [1, 2], 'y': [1, 2]}, inputs_info)

    def testNoInputs(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(module._convert_dict_inputs(None, {}), {})
        self.assertEqual(module._convert_dict_inputs({}, {}), {})
        with self.assertRaisesRegexp(TypeError, 'expects no inputs'):
            module._convert_dict_inputs([None], {})
        with self.assertRaisesRegexp(TypeError, 'expects no inputs'):
            module._convert_dict_inputs(1, {})
        with self.assertRaisesRegexp(TypeError, "extra given \\['x'\\]"):
            module._convert_dict_inputs({'x': 1}, {})

    def testMultipleInputs(self):
        if False:
            for i in range(10):
                print('nop')
        inputs_info = {'x': tensor_info.ParsedTensorInfo(tf.float32, tf.TensorShape([None]), is_sparse=False), 'y': tensor_info.ParsedTensorInfo(tf.float32, tf.TensorShape([None]), is_sparse=False)}

        def _check(dict_inputs):
            if False:
                return 10
            self.assertEqual(len(dict_inputs), 2)
            for key in ('x', 'y'):
                self.assertEqual(dict_inputs[key].dtype, tf.float32)
                self.assertTrue(dict_inputs[key].shape.is_compatible_with([None]))
        _check(module._convert_dict_inputs({'x': [1, 2], 'y': [1, 2]}, inputs_info))
        with self.assertRaisesRegexp(TypeError, "missing \\['x', 'y'\\]"):
            module._convert_dict_inputs(None, inputs_info)
        with self.assertRaisesRegexp(TypeError, "missing \\['x', 'y'\\]"):
            module._convert_dict_inputs({}, inputs_info)
        with self.assertRaisesRegexp(TypeError, "missing \\['x', 'y'\\]"):
            module._convert_dict_inputs({'z': 1}, inputs_info)
        with self.assertRaisesRegexp(TypeError, 'Signature expects multiple inputs. Use a dict.'):
            module._convert_dict_inputs(1, inputs_info)

    def testOutputWithDefault(self):
        if False:
            return 10
        outputs = {'default': 'result', 'extra': 'dbg info'}
        self.assertEquals(module._prepare_outputs(outputs, as_dict=False), 'result')
        self.assertEquals(module._prepare_outputs(outputs, as_dict=True), outputs)

    def testDictOutput(self):
        if False:
            i = 10
            return i + 15
        outputs = {'x': 1, 'y': 2}
        self.assertEquals(module._prepare_outputs(outputs, as_dict=True), outputs)
        with self.assertRaisesRegexp(TypeError, 'Use as_dict=True.'):
            self.assertEquals(module._prepare_outputs(outputs, as_dict=False), outputs)

class GetStateScopeTest(tf.test.TestCase):

    def testGetStateScope(self):
        if False:
            return 10
        with tf.Graph().as_default():
            self.assertEqual(module._try_get_state_scope('a'), 'a/')
            self.assertEqual(module._try_get_state_scope('a'), 'a_1/')

    def testGetStateScope_UsesVariableScope(self):
        if False:
            i = 10
            return i + 15
        with tf.Graph().as_default():
            self.assertEqual(module._try_get_state_scope('a'), 'a/')
            with tf.compat.v1.variable_scope(None, default_name='a') as vs:
                self.assertEqual(vs.name, 'a_1')

    def testGetStateScope_UsesNameScope(self):
        if False:
            for i in range(10):
                print('nop')
        with tf.Graph().as_default():
            self.assertEqual(module._try_get_state_scope('a'), 'a/')
            with tf.compat.v1.name_scope('a') as ns:
                self.assertEqual(ns, 'a_1/')

    def testGetStateScope_UnusedNameScope(self):
        if False:
            print('Hello World!')
        with tf.Graph().as_default():
            self.assertEqual(module._try_get_state_scope('a', False), 'a/')
            with tf.compat.v1.name_scope('a') as ns:
                self.assertEqual(ns, 'a/')
            self.assertEqual(module._try_get_state_scope('a', False), 'a_1/')
            with tf.compat.v1.name_scope('a') as ns:
                self.assertEqual(ns, 'a_1/')

    def testGetStateScope_AlreadyUsedNameScope(self):
        if False:
            print('Hello World!')
        with tf.Graph().as_default():
            with tf.compat.v1.name_scope('a'):
                pass
            with self.assertRaisesRegexp(RuntimeError, 'name_scope was already taken'):
                module._try_get_state_scope('a', False)

    def testGetStateScopeWithActiveScopes(self):
        if False:
            while True:
                i = 10
        with tf.Graph().as_default():
            with tf.compat.v1.name_scope('foo'):
                abs_scope = module._try_get_state_scope('a', False)
                self.assertEqual(abs_scope, 'a/')
                with tf.compat.v1.name_scope(abs_scope) as ns:
                    self.assertEqual(ns, 'a/')
        with tf.Graph().as_default():
            with tf.compat.v1.variable_scope('vs'):
                self.assertEqual(module._try_get_state_scope('a', False), 'vs/a/')
                with tf.compat.v1.name_scope(name='a') as ns:
                    self.assertEqual(ns, 'vs/a/')
        with tf.Graph().as_default():
            with tf.compat.v1.name_scope('foo'):
                with tf.compat.v1.variable_scope('vs'):
                    self.assertEquals(module._try_get_state_scope('a', False), 'vs/a/')

class _ModuleSpec(module_spec.ModuleSpec):

    def get_tags(self):
        if False:
            return 10
        return [set(), set(['special'])]

    def get_signature_names(self, tags=None):
        if False:
            print('Hello World!')
        if tags == set(['special']):
            return iter(['default', 'extra', 'sparse', 'ragged'])
        else:
            return iter(['default'])

    def get_input_info_dict(self, signature=None, tags=None):
        if False:
            i = 10
            return i + 15
        if signature == 'ragged' and tags == set(['special']):
            result = {'x': tensor_info.ParsedTensorInfo.from_type_spec(type_spec=tf.RaggedTensorSpec(shape=[None, None, None, 3], dtype=tf.float32, ragged_rank=2))}
        else:
            result = {'x': tensor_info.ParsedTensorInfo(tf.float32, tf.TensorShape([None]), is_sparse=signature == 'sparse' and tags == set(['special']))}
        if tags == set(['special']) and signature == 'extra':
            result['y'] = result['x']
        return result

    def get_output_info_dict(self, signature=None, tags=None):
        if False:
            i = 10
            return i + 15
        result = {'default': tensor_info.ParsedTensorInfo(tf.float32, tf.TensorShape([None]), is_sparse=False)}
        if tags == set(['special']) and signature == 'extra':
            result['z'] = result['default']
        return result

    def _create_impl(self, name, trainable, tags):
        if False:
            return 10
        return _ModuleImpl(name, trainable)

    def _get_attached_bytes(self, key, tags):
        if False:
            return 10
        del key, tags
        return None

class _ModuleImpl(module_impl.ModuleImpl):

    def __init__(self, name, trainable):
        if False:
            return 10
        super().__init__()
        with tf.compat.v1.variable_scope(name):
            pass

    def create_apply_graph(self, signature, input_tensors, name):
        if False:
            print('Hello World!')
        with tf.compat.v1.name_scope(name):
            if signature == 'sparse':
                input_tensors = {key: tf.compat.v1.sparse_tensor_to_dense(value) for (key, value) in input_tensors.items()}
            result = {'default': 2 * input_tensors['x']}
            if signature == 'extra':
                result['z'] = 2 * input_tensors['x'] + 3 * input_tensors['y']
            return result

    def export(self, path, session):
        if False:
            return 10
        raise NotImplementedError()

    @property
    def variable_map(self):
        if False:
            return 10
        raise NotImplementedError()

class ModuleTest(tf.test.TestCase):

    def testModuleSingleInput(self):
        if False:
            i = 10
            return i + 15
        with tf.Graph().as_default():
            m = module.Module(_ModuleSpec())
            result = m([1, 2])
            with tf.compat.v1.Session() as session:
                self.assertAllEqual(session.run(result), [2, 4])

    def testModuleDictInput(self):
        if False:
            return 10
        with tf.Graph().as_default():
            m = module.Module(_ModuleSpec())
            result = m({'x': [1, 2]})
            with tf.compat.v1.Session() as session:
                self.assertAllEqual(session.run(result), [2, 4])

    def testModuleDictOutput(self):
        if False:
            i = 10
            return i + 15
        with tf.Graph().as_default():
            m = module.Module(_ModuleSpec())
            result = m([1, 2], as_dict=True)
            self.assertIsInstance(result, dict)
            self.assertAllEqual(list(result.keys()), ['default'])

    def testModuleInNestedScope(self):
        if False:
            print('Hello World!')
        with tf.Graph().as_default():
            with tf.compat.v1.variable_scope('foo'):
                m = module.Module(_ModuleSpec())
                result = m([1, 2])
            with tf.compat.v1.Session() as session:
                self.assertAllEqual(session.run(result), [2, 4])

    def testModuleInterfaceGettersDefaultSignatureAndTags(self):
        if False:
            i = 10
            return i + 15
        with tf.Graph().as_default():
            m = module.Module(_ModuleSpec())
            self.assertItemsEqual(m.get_signature_names(), ['default'])
            self.assertItemsEqual(m.get_input_info_dict().keys(), ['x'])
            self.assertItemsEqual(m.get_output_info_dict().keys(), ['default'])

    def testModuleInterfaceGettersExplicitSignatureAndTags(self):
        if False:
            i = 10
            return i + 15
        'Tests that tags from Module(...) apply to module.get_*().'
        with tf.Graph().as_default():
            m = module.Module(_ModuleSpec(), tags={'special'})
            self.assertItemsEqual(m.get_signature_names(), ['default', 'extra', 'sparse', 'ragged'])
            self.assertItemsEqual(m.get_input_info_dict(signature='extra').keys(), ['x', 'y'])
            self.assertItemsEqual(m.get_output_info_dict(signature='extra').keys(), ['z', 'default'])

class EvalFunctionForModuleTest(tf.test.TestCase):
    """Tests for hub.eval_function_for_module(...).

  This tests that hub.eval_function_for_module parses input variables,
  signatures and tags correctly and that it returns the correct output.
  End-to-end tests with the native module are done in native_module_test.py.
  """

    def testSingleInput(self):
        if False:
            for i in range(10):
                print('nop')
        with module.eval_function_for_module(_ModuleSpec()) as f:
            self.assertAllEqual(f([1, 2]), [2, 4])

    def testSparseInput(self):
        if False:
            for i in range(10):
                print('nop')
        with module.eval_function_for_module(_ModuleSpec(), tags={'special'}) as f:
            self.assertAllEqual(f(tf.compat.v1.SparseTensorValue([[0]], [1], [2]), signature='sparse'), [2, 0])

    def testDictInput(self):
        if False:
            print('Hello World!')
        with module.eval_function_for_module(_ModuleSpec()) as f:
            self.assertAllEqual(f({'x': [1, 2]}), [2, 4])

    def testDictOutput(self):
        if False:
            i = 10
            return i + 15
        with module.eval_function_for_module(_ModuleSpec()) as f:
            result = f({'x': [1, 2]}, as_dict=True)
        self.assertTrue(isinstance(result, dict))
        self.assertAllEqual(list(result.keys()), ['default'])

    def testSignature(self):
        if False:
            print('Hello World!')
        with module.eval_function_for_module(_ModuleSpec()) as f:
            self.assertAllEqual(f([1, 2]), [2, 4])

    def testExplicitSignatureAndTags(self):
        if False:
            print('Hello World!')
        with module.eval_function_for_module(_ModuleSpec(), tags={'special'}) as f:
            result = f(dict(x=[1], y=[2]), signature='extra', as_dict=True)
            self.assertAllEqual(result['default'], [2])
            self.assertAllEqual(result['z'], [8])
if __name__ == '__main__':
    tf.test.main()