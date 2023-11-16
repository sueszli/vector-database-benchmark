"""Tests the registration functions.

For integration tests that use save and load functions, see
registration_saving_test.py.
"""
from absl.testing import parameterized
from tensorflow.python.eager import test
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base

@registration.register_serializable()
class RegisteredClass(base.Trackable):
    pass

@registration.register_serializable(name='Subclass')
class RegisteredSubclass(RegisteredClass):
    pass

@registration.register_serializable(package='testing')
class CustomPackage(base.Trackable):
    pass

@registration.register_serializable(package='testing', name='name')
class CustomPackageAndName(base.Trackable):
    pass

class SerializableRegistrationTest(test.TestCase, parameterized.TestCase):

    @parameterized.parameters([(RegisteredClass, 'Custom.RegisteredClass'), (RegisteredSubclass, 'Custom.Subclass'), (CustomPackage, 'testing.CustomPackage'), (CustomPackageAndName, 'testing.name')])
    def test_registration(self, expected_cls, expected_name):
        if False:
            while True:
                i = 10
        obj = expected_cls()
        self.assertEqual(registration.get_registered_class_name(obj), expected_name)
        self.assertIs(registration.get_registered_class(expected_name), expected_cls)

    def test_get_invalid_name(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(registration.get_registered_class('invalid name'))

    def test_get_unregistered_class(self):
        if False:
            i = 10
            return i + 15

        class NotRegistered(base.Trackable):
            pass
        no_register = NotRegistered
        self.assertIsNone(registration.get_registered_class_name(no_register))

    def test_duplicate_registration(self):
        if False:
            while True:
                i = 10

        @registration.register_serializable()
        class Duplicate(base.Trackable):
            pass
        dup = Duplicate()
        self.assertEqual(registration.get_registered_class_name(dup), 'Custom.Duplicate')
        registration.register_serializable(package='duplicate')(Duplicate)
        self.assertEqual(registration.get_registered_class_name(dup), 'duplicate.Duplicate')
        self.assertIs(registration.get_registered_class('Custom.Duplicate'), Duplicate)
        self.assertIs(registration.get_registered_class('duplicate.Duplicate'), Duplicate)
        with self.assertRaisesRegex(ValueError, 'already been registered'):
            registration.register_serializable(package='testing', name='CustomPackage')(Duplicate)

    def test_register_non_class_fails(self):
        if False:
            print('Hello World!')
        obj = RegisteredClass()
        with self.assertRaisesRegex(TypeError, 'must be a class'):
            registration.register_serializable()(obj)

    def test_register_bad_predicate_fails(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, 'must be callable'):
            registration.register_serializable(predicate=0)(RegisteredClass)

    def test_predicate(self):
        if False:
            i = 10
            return i + 15

        class Predicate(base.Trackable):

            def __init__(self, register_this):
                if False:
                    for i in range(10):
                        print('nop')
                self.register_this = register_this
        registration.register_serializable(name='RegisterThisOnlyTrue', predicate=lambda x: isinstance(x, Predicate) and x.register_this)(Predicate)
        a = Predicate(True)
        b = Predicate(False)
        self.assertEqual(registration.get_registered_class_name(a), 'Custom.RegisterThisOnlyTrue')
        self.assertIsNone(registration.get_registered_class_name(b))
        registration.register_serializable(name='RegisterAllPredicate', predicate=lambda x: isinstance(x, Predicate))(Predicate)
        self.assertEqual(registration.get_registered_class_name(a), 'Custom.RegisterAllPredicate')
        self.assertEqual(registration.get_registered_class_name(b), 'Custom.RegisterAllPredicate')

class CheckpointSaverRegistrationTest(test.TestCase):

    def test_invalid_registration(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypeError, 'must be string'):
            registration.register_checkpoint_saver(package=None, name='test', predicate=lambda : None, save_fn=lambda : None, restore_fn=lambda : None)
        with self.assertRaisesRegex(TypeError, 'must be string'):
            registration.register_checkpoint_saver(name=None, predicate=lambda : None, save_fn=lambda : None, restore_fn=lambda : None)
        with self.assertRaisesRegex(ValueError, 'Invalid registered checkpoint saver.'):
            registration.register_checkpoint_saver(package='package', name='t/est', predicate=lambda : None, save_fn=lambda : None, restore_fn=lambda : None)
        with self.assertRaisesRegex(ValueError, 'Invalid registered checkpoint saver.'):
            registration.register_checkpoint_saver(package='package', name='t/est', predicate=lambda : None, save_fn=lambda : None, restore_fn=lambda : None)
        with self.assertRaisesRegex(TypeError, 'The predicate registered to a checkpoint saver must be callable'):
            registration.register_checkpoint_saver(name='test', predicate=None, save_fn=lambda : None, restore_fn=lambda : None)
        with self.assertRaisesRegex(TypeError, 'The save_fn must be callable'):
            registration.register_checkpoint_saver(name='test', predicate=lambda : None, save_fn=None, restore_fn=lambda : None)
        with self.assertRaisesRegex(TypeError, 'The restore_fn must be callable'):
            registration.register_checkpoint_saver(name='test', predicate=lambda : None, save_fn=lambda : None, restore_fn=None)

    def test_registration(self):
        if False:
            for i in range(10):
                print('nop')
        registration.register_checkpoint_saver(package='Testing', name='test_predicate', predicate=lambda x: hasattr(x, 'check_attr'), save_fn=lambda : 'save', restore_fn=lambda : 'restore')
        x = base.Trackable()
        self.assertIsNone(registration.get_registered_saver_name(x))
        x.check_attr = 1
        saver_name = registration.get_registered_saver_name(x)
        self.assertEqual(saver_name, 'Testing.test_predicate')
        self.assertEqual(registration.get_save_function(saver_name)(), 'save')
        self.assertEqual(registration.get_restore_function(saver_name)(), 'restore')
        registration.validate_restore_function(x, 'Testing.test_predicate')
        with self.assertRaisesRegex(ValueError, 'saver cannot be found'):
            registration.validate_restore_function(x, 'Invalid.name')
        x2 = base.Trackable()
        with self.assertRaisesRegex(ValueError, 'saver cannot be used'):
            registration.validate_restore_function(x2, 'Testing.test_predicate')
if __name__ == '__main__':
    test.main()