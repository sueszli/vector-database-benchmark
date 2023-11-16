from abc import abstractmethod, ABCMeta
from unittest import TestCase
from six import with_metaclass
from zipline.utils.final import FinalMeta, final
from zipline.utils.metautils import compose_types

class FinalMetaTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')

        class ClassWithFinal(with_metaclass(FinalMeta, object)):
            a = final('ClassWithFinal: a')
            b = 'ClassWithFinal: b'

            @final
            def f(self):
                if False:
                    while True:
                        i = 10
                return 'ClassWithFinal: f'

            def g(self):
                if False:
                    i = 10
                    return i + 15
                return 'ClassWithFinal: g'
        cls.class_ = ClassWithFinal

    def test_subclass_no_override(self):
        if False:
            print('Hello World!')
        '\n        Tests that it is valid to create a subclass that does not override\n        any methods.\n        '

        class SubClass(self.class_):
            pass

    def test_subclass_no_final_override(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that it is valid to create a subclass that does not override\n        and final methods.\n        '

        class SubClass(self.class_):
            b = 'SubClass: b'

            def g(self):
                if False:
                    return 10
                return 'SubClass: g'

    def test_override_final_no_decorator(self):
        if False:
            while True:
                i = 10
        '\n        Tests that attempting to create a subclass that overrides a final\n        method will raise a `TypeError`.\n        '
        with self.assertRaises(TypeError):

            class SubClass(self.class_):

                def f(self):
                    if False:
                        i = 10
                        return i + 15
                    return 'SubClass: f'

    def test_override_final_attribute(self):
        if False:
            print('Hello World!')
        '\n        Tests that attempting to create a subclass that overrides a final\n        attribute will raise a `TypeError`.\n        '
        with self.assertRaises(TypeError):

            class SubClass(self.class_):
                a = 'SubClass: a'

    def test_override_final_with_decorator(self):
        if False:
            print('Hello World!')
        '\n        Tests that attempting to create a subclass that overrides a final\n        method will raise a `TypeError` even if you mark the new version as\n        final.\n        '
        with self.assertRaises(TypeError):

            class SubClass(self.class_):

                @final
                def f(self):
                    if False:
                        while True:
                            i = 10
                    return 'SubClass: f'

    def test_override_final_attribute_with_final(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that attempting to create a subclass that overrides a final\n        attribute will raise a `TypeError` even if you mark the new version as\n        final.\n        '
        with self.assertRaises(TypeError):

            class SubClass(self.class_):
                a = final('SubClass: a')

    def test_override_on_class_object(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests overriding final methods and attributes on the class object\n        itself.\n        '

        class SubClass(self.class_):
            pass
        with self.assertRaises(TypeError):
            SubClass.f = lambda self: 'SubClass: f'
        with self.assertRaises(TypeError):
            SubClass.a = 'SubClass: a'

    def test_override_on_instance(self):
        if False:
            while True:
                i = 10
        '\n        Tests overriding final methods on instances of a class.\n        '

        class SubClass(self.class_):

            def h(self):
                if False:
                    i = 10
                    return i + 15
                pass
        s = SubClass()
        with self.assertRaises(TypeError):
            s.f = lambda self: 'SubClass: f'
        with self.assertRaises(TypeError):
            s.a = lambda self: 'SubClass: a'

    def test_override_on_super(self):
        if False:
            while True:
                i = 10
        '\n        Tests overriding on the class that has the @final methods in it.\n        '
        old_a = self.class_.a
        old_f = self.class_.f
        try:
            with self.assertRaises(TypeError):
                self.class_.f = lambda *args: None
        except Exception:
            self.class_.f = old_f
            raise
        try:
            with self.assertRaises(TypeError):
                self.class_.a = 'SubClass: a'
        except Exception:
            self.class_.a = old_a
            raise

    def test_override___setattr___on_subclass(self):
        if False:
            while True:
                i = 10
        '\n        Tests an attempt to override __setattr__ which is implicitly final.\n        '
        with self.assertRaises(TypeError):

            class SubClass(self.class_):

                def __setattr__(self, name, value):
                    if False:
                        i = 10
                        return i + 15
                    object.__setattr__(self, name, value)

    def test_override___setattr___on_instance(self):
        if False:
            while True:
                i = 10
        '\n        Tests overriding __setattr__ on an instance.\n        '

        class SubClass(self.class_):
            pass
        s = SubClass()
        with self.assertRaises(TypeError):
            s.__setattr__ = lambda a, b: None

class FinalABCMetaTestCase(FinalMetaTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        FinalABCMeta = compose_types(FinalMeta, ABCMeta)

        class ABCWithFinal(with_metaclass(FinalABCMeta, object)):
            a = final('ABCWithFinal: a')
            b = 'ABCWithFinal: b'

            @final
            def f(self):
                if False:
                    while True:
                        i = 10
                return 'ABCWithFinal: f'

            def g(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'ABCWithFinal: g'

            @abstractmethod
            def h(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise NotImplementedError('h')
        cls.class_ = ABCWithFinal

    def test_cannot_instantiate_subclass(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that you cannot create an instance of a subclass\n        that does not implement the abstractmethod h.\n        '

        class AbstractSubClass(self.class_):
            pass
        with self.assertRaises(TypeError):
            AbstractSubClass()

    def test_override_on_instance(self):
        if False:
            for i in range(10):
                print('nop')

        class SubClass(self.class_):

            def h(self):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Pass the abstract tests by creating this method.\n                '
                pass
        s = SubClass()
        with self.assertRaises(TypeError):
            s.f = lambda self: 'SubClass: f'

    def test_override___setattr___on_instance(self):
        if False:
            print('Hello World!')
        '\n        Tests overriding __setattr__ on an instance.\n        '

        class SubClass(self.class_):

            def h(self):
                if False:
                    print('Hello World!')
                pass
        s = SubClass()
        with self.assertRaises(TypeError):
            s.__setattr__ = lambda a, b: None

    def test_subclass_setattr(self):
        if False:
            return 10
        "\n        Tests that subclasses don't destroy the __setattr__.\n        "

        class ClassWithFinal(with_metaclass(FinalMeta, object)):

            @final
            def f(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'ClassWithFinal: f'

        class SubClass(ClassWithFinal):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.a = 'a'
        SubClass()
        self.assertEqual(SubClass().a, 'a')
        self.assertEqual(SubClass().f(), 'ClassWithFinal: f')

    def test_final_classmethod(self):
        if False:
            i = 10
            return i + 15

        class ClassWithClassMethod(with_metaclass(FinalMeta, object)):
            count = 0

            @final
            @classmethod
            def f(cls):
                if False:
                    return 10
                cls.count += 1
                return cls.count
        with self.assertRaises(TypeError):

            class ClassOverridingClassMethod(ClassWithClassMethod):

                @classmethod
                def f(cls):
                    if False:
                        i = 10
                        return i + 15
                    return 'Oh Noes!'
        with self.assertRaises(TypeError):
            ClassWithClassMethod.f = lambda cls: 0
        self.assertEqual(ClassWithClassMethod.f(), 1)
        self.assertEqual(ClassWithClassMethod.f(), 2)
        self.assertEqual(ClassWithClassMethod.f(), 3)
        instance = ClassWithClassMethod()
        with self.assertRaises(TypeError):
            instance.f = lambda cls: 0
        self.assertEqual(ClassWithClassMethod.f(), 4)
        self.assertEqual(ClassWithClassMethod.f(), 5)
        self.assertEqual(ClassWithClassMethod.f(), 6)