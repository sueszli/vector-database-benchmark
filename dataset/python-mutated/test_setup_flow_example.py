def setup_module(module):
    if False:
        print('Hello World!')
    module.TestStateFullThing.classcount = 0

class TestStateFullThing:

    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.classcount += 1

    def teardown_class(cls):
        if False:
            while True:
                i = 10
        cls.classcount -= 1

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        self.id = eval(method.__name__[5:])

    def test_42(self):
        if False:
            i = 10
            return i + 15
        assert self.classcount == 1
        assert self.id == 42

    def test_23(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.classcount == 1
        assert self.id == 23

def teardown_module(module):
    if False:
        i = 10
        return i + 15
    assert module.TestStateFullThing.classcount == 0
' For this example the control flow happens as follows::\n    import test_setup_flow_example\n    setup_module(test_setup_flow_example)\n       setup_class(TestStateFullThing)\n           instance = TestStateFullThing()\n           setup_method(instance, instance.test_42)\n              instance.test_42()\n           setup_method(instance, instance.test_23)\n              instance.test_23()\n       teardown_class(TestStateFullThing)\n    teardown_module(test_setup_flow_example)\n\nNote that ``setup_class(TestStateFullThing)`` is called and not\n``TestStateFullThing.setup_class()`` which would require you\nto insert ``setup_class = classmethod(setup_class)`` to make\nyour setup function callable.\n'