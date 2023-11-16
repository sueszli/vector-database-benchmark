import util

class ExampleClass(object):

    def __init__(self):
        if False:
            return 10
        self.called = []
        self.onChanged = util.Event()

    def increment(self, title):
        if False:
            print('Hello World!')
        self.called.append(title)

class TestEvent:

    def testEvent(self):
        if False:
            while True:
                i = 10
        test_obj = ExampleClass()
        test_obj.onChanged.append(lambda : test_obj.increment('Called #1'))
        test_obj.onChanged.append(lambda : test_obj.increment('Called #2'))
        test_obj.onChanged.once(lambda : test_obj.increment('Once'))
        assert test_obj.called == []
        test_obj.onChanged()
        assert test_obj.called == ['Called #1', 'Called #2', 'Once']
        test_obj.onChanged()
        test_obj.onChanged()
        assert test_obj.called == ['Called #1', 'Called #2', 'Once', 'Called #1', 'Called #2', 'Called #1', 'Called #2']

    def testOnce(self):
        if False:
            print('Hello World!')
        test_obj = ExampleClass()
        test_obj.onChanged.once(lambda : test_obj.increment('Once test #1'))
        assert test_obj.called == []
        test_obj.onChanged()
        assert test_obj.called == ['Once test #1']
        test_obj.onChanged()
        test_obj.onChanged()
        assert test_obj.called == ['Once test #1']

    def testOnceMultiple(self):
        if False:
            i = 10
            return i + 15
        test_obj = ExampleClass()
        test_obj.onChanged.once(lambda : test_obj.increment('Once test #1'))
        test_obj.onChanged.once(lambda : test_obj.increment('Once test #2'))
        test_obj.onChanged.once(lambda : test_obj.increment('Once test #3'))
        assert test_obj.called == []
        test_obj.onChanged()
        assert test_obj.called == ['Once test #1', 'Once test #2', 'Once test #3']
        test_obj.onChanged()
        test_obj.onChanged()
        assert test_obj.called == ['Once test #1', 'Once test #2', 'Once test #3']

    def testOnceNamed(self):
        if False:
            return 10
        test_obj = ExampleClass()
        test_obj.onChanged.once(lambda : test_obj.increment('Once test #1/1'), 'type 1')
        test_obj.onChanged.once(lambda : test_obj.increment('Once test #1/2'), 'type 1')
        test_obj.onChanged.once(lambda : test_obj.increment('Once test #2'), 'type 2')
        assert test_obj.called == []
        test_obj.onChanged()
        assert test_obj.called == ['Once test #1/1', 'Once test #2']
        test_obj.onChanged()
        test_obj.onChanged()
        assert test_obj.called == ['Once test #1/1', 'Once test #2']