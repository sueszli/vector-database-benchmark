from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import assert_equal, assert_is, assert_is_instance, assert_is_subclass, assert_true
from zipline.utils.metautils import compose_types, with_metaclasses

class C(object):

    @staticmethod
    def f():
        if False:
            for i in range(10):
                print('nop')
        return 'C.f'

    def delegate(self):
        if False:
            while True:
                i = 10
        return ('C.delegate', super(C, self).delegate())

class D(object):

    @staticmethod
    def f():
        if False:
            i = 10
            return i + 15
        return 'D.f'

    @staticmethod
    def g():
        if False:
            print('Hello World!')
        return 'D.g'

    def delegate(self):
        if False:
            while True:
                i = 10
        return 'D.delegate'

class ComposeTypesTestCase(ZiplineTestCase):

    def test_identity(self):
        if False:
            for i in range(10):
                print('nop')
        assert_is(compose_types(C), C, msg='compose_types of a single class should be identity')

    def test_compose(self):
        if False:
            while True:
                i = 10
        composed = compose_types(C, D)
        assert_is_subclass(composed, C)
        assert_is_subclass(composed, D)

    def test_compose_mro(self):
        if False:
            i = 10
            return i + 15
        composed = compose_types(C, D)
        assert_equal(composed.f(), C.f())
        assert_equal(composed.g(), D.g())
        assert_equal(composed().delegate(), ('C.delegate', 'D.delegate'))

class M(type):

    def __new__(mcls, name, bases, dict_):
        if False:
            i = 10
            return i + 15
        dict_['M'] = True
        return super(M, mcls).__new__(mcls, name, bases, dict_)

class N(type):

    def __new__(mcls, name, bases, dict_):
        if False:
            print('Hello World!')
        dict_['N'] = True
        return super(N, mcls).__new__(mcls, name, bases, dict_)

class WithMetaclassesTestCase(ZiplineTestCase):

    def test_with_metaclasses_no_subclasses(self):
        if False:
            while True:
                i = 10

        class E(with_metaclasses((M, N))):
            pass
        assert_true(E.M)
        assert_true(E.N)
        assert_is_instance(E, M)
        assert_is_instance(E, N)

    def test_with_metaclasses_with_subclasses(self):
        if False:
            return 10

        class E(with_metaclasses((M, N), C, D)):
            pass
        assert_true(E.M)
        assert_true(E.N)
        assert_is_instance(E, M)
        assert_is_instance(E, N)
        assert_is_subclass(E, C)
        assert_is_subclass(E, D)