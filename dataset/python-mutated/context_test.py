from caffe2.python import context, test_util
from threading import Thread

class MyContext(context.Managed):
    pass

class DefaultMyContext(context.DefaultManaged):
    pass

class ChildMyContext(MyContext):
    pass

class TestContext(test_util.TestCase):

    def use_my_context(self):
        if False:
            print('Hello World!')
        try:
            for _ in range(100):
                with MyContext() as a:
                    for _ in range(100):
                        self.assertTrue(MyContext.current() == a)
        except Exception as e:
            self._exceptions.append(e)

    def testMultiThreaded(self):
        if False:
            for i in range(10):
                print('nop')
        threads = []
        self._exceptions = []
        for _ in range(8):
            thread = Thread(target=self.use_my_context)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
        for e in self._exceptions:
            raise e

    @MyContext()
    def testDecorator(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNotNone(MyContext.current())

    def testNonDefaultCurrent(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AssertionError):
            MyContext.current()
        ctx = MyContext()
        self.assertEqual(MyContext.current(value=ctx), ctx)
        self.assertIsNone(MyContext.current(required=False))

    def testDefaultCurrent(self):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(DefaultMyContext.current(), DefaultMyContext)

    def testNestedContexts(self):
        if False:
            while True:
                i = 10
        with MyContext() as ctx1:
            with DefaultMyContext() as ctx2:
                self.assertEqual(DefaultMyContext.current(), ctx2)
                self.assertEqual(MyContext.current(), ctx1)

    def testChildClasses(self):
        if False:
            while True:
                i = 10
        with ChildMyContext() as ctx:
            self.assertEqual(ChildMyContext.current(), ctx)
            self.assertEqual(MyContext.current(), ctx)