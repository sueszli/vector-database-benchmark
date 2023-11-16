"""Tests for saveable_object_util."""
import os
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util

class _VarSaveable(saveable_object.SaveableObject):

    def __init__(self, var, slice_spec, name):
        if False:
            print('Hello World!')
        specs = [saveable_object.SaveSpec(var.read_value(), slice_spec, name)]
        super().__init__(var, specs, name)

    def restore(self, restored_tensors, restored_shapes):
        if False:
            for i in range(10):
                print('nop')
        return self.op.assign(restored_tensors[0])

def _create_converted_trackable(obj):
    if False:
        print('Hello World!')
    saveable_factories = saveable_object_util.saveable_objects_from_trackable(obj)
    saveables = [factory(name) for (name, factory) in saveable_factories.items()]
    return saveable_object_util.SaveableCompatibilityConverter(obj, saveables)

class SaveableCompatibilityConverterTest(test.TestCase):

    def test_convert_no_saveable(self):
        if False:
            while True:
                i = 10
        t = base.Trackable()
        converter = _create_converted_trackable(t)
        self.assertEmpty(converter._serialize_to_tensors())
        converter._restore_from_tensors({})
        with self.assertRaisesRegex(ValueError, 'Could not restore object'):
            converter._restore_from_tensors({'': 0})

    def test_convert_single_saveable(self):
        if False:
            for i in range(10):
                print('nop')

        class MyTrackable(base.Trackable):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.a = variables.Variable(5.0)

            def _gather_saveables_for_checkpoint(self):
                if False:
                    while True:
                        i = 10
                return {'a': lambda name: _VarSaveable(self.a, '', name)}
        t = MyTrackable()
        converter = _create_converted_trackable(t)
        serialized_tensors = converter._serialize_to_tensors()
        self.assertLen(serialized_tensors, 1)
        self.assertIn('a', serialized_tensors)
        self.assertEqual(5, self.evaluate(serialized_tensors['a']))
        with self.assertRaisesRegex(ValueError, 'Could not restore object'):
            converter._restore_from_tensors({})
        with self.assertRaisesRegex(ValueError, 'Could not restore object'):
            converter._restore_from_tensors({'not_a': 1.0})
        self.assertEqual(5, self.evaluate(t.a))
        converter._restore_from_tensors({'a': 123.0})
        self.assertEqual(123, self.evaluate(t.a))

    def test_convert_single_saveable_renamed(self):
        if False:
            return 10

        class MyTrackable(base.Trackable):

            def __init__(self):
                if False:
                    return 10
                self.a = variables.Variable(15.0)

            def _gather_saveables_for_checkpoint(self):
                if False:
                    while True:
                        i = 10
                return {'a': lambda name: _VarSaveable(self.a, '', name + '-value')}
        t = MyTrackable()
        converter = _create_converted_trackable(t)
        serialized_tensors = converter._serialize_to_tensors()
        self.assertLen(serialized_tensors, 1)
        self.assertEqual(15, self.evaluate(serialized_tensors['a-value']))
        with self.assertRaisesRegex(ValueError, 'Could not restore object'):
            converter._restore_from_tensors({'a': 1.0})
        self.assertEqual(15, self.evaluate(t.a))
        converter._restore_from_tensors({'a-value': 456.0})
        self.assertEqual(456, self.evaluate(t.a))

class _MultiSpecSaveable(saveable_object.SaveableObject):

    def __init__(self, obj, name):
        if False:
            i = 10
            return i + 15
        self.obj = obj
        specs = [saveable_object.SaveSpec(obj.a, '', name + '-a'), saveable_object.SaveSpec(obj.b, '', name + '-b')]
        super(_MultiSpecSaveable, self).__init__(None, specs, name)

    def restore(self, restored_tensors, restored_shapes):
        if False:
            while True:
                i = 10
        del restored_shapes
        self.obj.a.assign(restored_tensors[0])
        self.obj.b.assign(restored_tensors[1])

class State(resource.TrackableResource):

    def __init__(self, initial_value):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._initial_value = initial_value
        self._initialize()

    def _create_resource(self):
        if False:
            while True:
                i = 10
        return gen_resource_variable_ops.var_handle_op(shape=[], dtype=dtypes.float32, shared_name=context.anonymous_name(), name='StateVar', container='')

    def _initialize(self):
        if False:
            while True:
                i = 10
        gen_resource_variable_ops.assign_variable_op(self.resource_handle, self._initial_value)

    def _destroy_resource(self):
        if False:
            for i in range(10):
                print('nop')
        gen_resource_variable_ops.destroy_resource_op(self.resource_handle, ignore_lookup_error=True)

    def read(self):
        if False:
            while True:
                i = 10
        return gen_resource_variable_ops.read_variable_op(self.resource_handle, dtypes.float32)

    def assign(self, value):
        if False:
            return 10
        gen_resource_variable_ops.assign_variable_op(self.resource_handle, value)

class _StateSaveable(saveable_object.SaveableObject):

    def __init__(self, obj, name):
        if False:
            return 10
        spec = saveable_object.SaveSpec(obj.read(), '', name)
        self.obj = obj
        super(_StateSaveable, self).__init__(obj, [spec], name)

    def restore(self, restored_tensors, restored_shapes):
        if False:
            for i in range(10):
                print('nop')
        del restored_shapes
        self.obj.assign(restored_tensors[0])

class SaveableState(State):

    def _gather_saveables_for_checkpoint(self):
        if False:
            i = 10
            return i + 15
        return {'value': lambda name: _StateSaveable(self, name)}

class TrackableState(State):

    def _serialize_to_tensors(self):
        if False:
            while True:
                i = 10
        return {'value': self.read()}

    def _restore_from_tensors(self, restored_tensors):
        if False:
            i = 10
            return i + 15
        return self.assign(restored_tensors['value'])

class SaveableCompatibilityEndToEndTest(test.TestCase):

    def test_checkpoint_comparison(self):
        if False:
            print('Hello World!')
        saveable_state = SaveableState(5.0)
        trackable_state = TrackableState(10.0)
        self.assertEqual(5, self.evaluate(saveable_state.read()))
        self.assertEqual(10, self.evaluate(trackable_state.read()))
        ckpt_path = os.path.join(self.get_temp_dir(), 'ckpt')
        checkpoint.Checkpoint(a=saveable_state, b=trackable_state).write(ckpt_path)
        status = checkpoint.Checkpoint(b=saveable_state, a=trackable_state).read(ckpt_path)
        status.assert_consumed()
        self.assertEqual(10, self.evaluate(saveable_state.read()))
        self.assertEqual(5, self.evaluate(trackable_state.read()))
        to_convert = SaveableState(0.0)
        converted_saveable_state = _create_converted_trackable(to_convert)
        checkpoint.Checkpoint(a=converted_saveable_state).read(ckpt_path).assert_existing_objects_matched().expect_partial()
        self.assertEqual(5, self.evaluate(to_convert.read()))
        checkpoint.Checkpoint(b=converted_saveable_state).read(ckpt_path).assert_existing_objects_matched().expect_partial()
        self.assertEqual(10, self.evaluate(to_convert.read()))

class HasSerializeToTensorTest(test.TestCase):

    def test_has_serialize_to_tensor(self):
        if False:
            i = 10
            return i + 15

        class ReturnsTrue(base.Trackable):

            def _serialize_to_tensors(self):
                if False:
                    i = 10
                    return i + 15
                return {}

        class ReturnsFalse(base.Trackable):
            pass

        class SubclassReturnsFalse(ReturnsTrue):

            def _gather_saveables_for_checkpoint(self):
                if False:
                    return 10
                return {}
        self.assertTrue(saveable_object_util.trackable_has_serialize_to_tensor(ReturnsTrue()))
        self.assertFalse(saveable_object_util.trackable_has_serialize_to_tensor(ReturnsFalse()))
        self.assertFalse(saveable_object_util.trackable_has_serialize_to_tensor(SubclassReturnsFalse()))
if __name__ == '__main__':
    test.main()