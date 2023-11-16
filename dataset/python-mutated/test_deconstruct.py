from django.test import SimpleTestCase
from django.utils.deconstruct import deconstructible
from django.utils.version import get_docs_version

@deconstructible()
class DeconstructibleClass:
    pass

class DeconstructibleChildClass(DeconstructibleClass):
    pass

@deconstructible(path='utils_tests.deconstructible_classes.DeconstructibleWithPathClass')
class DeconstructibleWithPathClass:
    pass

class DeconstructibleWithPathChildClass(DeconstructibleWithPathClass):
    pass

@deconstructible(path='utils_tests.deconstructible_classes.DeconstructibleInvalidPathClass')
class DeconstructibleInvalidPathClass:
    pass

class DeconstructibleInvalidPathChildClass(DeconstructibleInvalidPathClass):
    pass

class DeconstructibleTests(SimpleTestCase):

    def test_deconstruct(self):
        if False:
            for i in range(10):
                print('nop')
        obj = DeconstructibleClass('arg', key='value')
        (path, args, kwargs) = obj.deconstruct()
        self.assertEqual(path, 'utils_tests.test_deconstruct.DeconstructibleClass')
        self.assertEqual(args, ('arg',))
        self.assertEqual(kwargs, {'key': 'value'})

    def test_deconstruct_with_path(self):
        if False:
            return 10
        obj = DeconstructibleWithPathClass('arg', key='value')
        (path, args, kwargs) = obj.deconstruct()
        self.assertEqual(path, 'utils_tests.deconstructible_classes.DeconstructibleWithPathClass')
        self.assertEqual(args, ('arg',))
        self.assertEqual(kwargs, {'key': 'value'})

    def test_deconstruct_child(self):
        if False:
            print('Hello World!')
        obj = DeconstructibleChildClass('arg', key='value')
        (path, args, kwargs) = obj.deconstruct()
        self.assertEqual(path, 'utils_tests.test_deconstruct.DeconstructibleChildClass')
        self.assertEqual(args, ('arg',))
        self.assertEqual(kwargs, {'key': 'value'})

    def test_deconstruct_child_with_path(self):
        if False:
            for i in range(10):
                print('nop')
        obj = DeconstructibleWithPathChildClass('arg', key='value')
        (path, args, kwargs) = obj.deconstruct()
        self.assertEqual(path, 'utils_tests.test_deconstruct.DeconstructibleWithPathChildClass')
        self.assertEqual(args, ('arg',))
        self.assertEqual(kwargs, {'key': 'value'})

    def test_invalid_path(self):
        if False:
            while True:
                i = 10
        obj = DeconstructibleInvalidPathClass()
        docs_version = get_docs_version()
        msg = f'Could not find object DeconstructibleInvalidPathClass in utils_tests.deconstructible_classes.\nPlease note that you cannot serialize things like inner classes. Please move the object into the main module body to use migrations.\nFor more information, see https://docs.djangoproject.com/en/{docs_version}/topics/migrations/#serializing-values'
        with self.assertRaisesMessage(ValueError, msg):
            obj.deconstruct()

    def test_parent_invalid_path(self):
        if False:
            return 10
        obj = DeconstructibleInvalidPathChildClass('arg', key='value')
        (path, args, kwargs) = obj.deconstruct()
        self.assertEqual(path, 'utils_tests.test_deconstruct.DeconstructibleInvalidPathChildClass')
        self.assertEqual(args, ('arg',))
        self.assertEqual(kwargs, {'key': 'value'})