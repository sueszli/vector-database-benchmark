from zipline.extensions import Registry
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import assert_raises_str, assert_true

class FakeInterface(object):
    pass

class RegistrationManagerTestCase(ZiplineTestCase):

    def test_load_not_registered(self):
        if False:
            for i in range(10):
                print('nop')
        rm = Registry(FakeInterface)
        msg = "no FakeInterface factory registered under name 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')
        rm.register('c', FakeInterface)
        rm.register('b', FakeInterface)
        rm.register('a', FakeInterface)
        msg = "no FakeInterface factory registered under name 'ayy-lmao', options are: ['a', 'b', 'c']"
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')

    def test_register_decorator(self):
        if False:
            i = 10
            return i + 15
        rm = Registry(FakeInterface)

        @rm.register('ayy-lmao')
        class ProperDummyInterface(FakeInterface):
            pass

        def check_registered():
            if False:
                return 10
            assert_true(rm.is_registered('ayy-lmao'), "Class ProperDummyInterface wasn't properly registered undername 'ayy-lmao'")
            self.assertIsInstance(rm.load('ayy-lmao'), ProperDummyInterface)
        check_registered()
        m = "FakeInterface factory with name 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, m):

            @rm.register('ayy-lmao')
            class Fake(object):
                pass
        check_registered()
        rm.unregister('ayy-lmao')
        msg = "no FakeInterface factory registered under name 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')
        msg = "FakeInterface factory 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            rm.unregister('ayy-lmao')

    def test_register_non_decorator(self):
        if False:
            return 10
        rm = Registry(FakeInterface)

        class ProperDummyInterface(FakeInterface):
            pass
        rm.register('ayy-lmao', ProperDummyInterface)

        def check_registered():
            if False:
                i = 10
                return i + 15
            assert_true(rm.is_registered('ayy-lmao'), "Class ProperDummyInterface wasn't properly registered undername 'ayy-lmao'")
            self.assertIsInstance(rm.load('ayy-lmao'), ProperDummyInterface)
        check_registered()

        class Fake(object):
            pass
        m = "FakeInterface factory with name 'ayy-lmao' is already registered"
        with assert_raises_str(ValueError, m):
            rm.register('ayy-lmao', Fake)
        check_registered()
        rm.unregister('ayy-lmao')
        msg = "no FakeInterface factory registered under name 'ayy-lmao', options are: []"
        with assert_raises_str(ValueError, msg):
            rm.load('ayy-lmao')
        msg = "FakeInterface factory 'ayy-lmao' was not already registered"
        with assert_raises_str(ValueError, msg):
            rm.unregister('ayy-lmao')