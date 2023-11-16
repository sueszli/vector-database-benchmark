import threading
from bzrlib import cethread, tests

class TestCatchingExceptionThread(tests.TestCase):

    def test_start_and_join_smoke_test(self):
        if False:
            i = 10
            return i + 15

        def do_nothing():
            if False:
                print('Hello World!')
            pass
        tt = cethread.CatchingExceptionThread(target=do_nothing)
        tt.start()
        tt.join()

    def test_exception_is_re_raised(self):
        if False:
            print('Hello World!')

        class MyException(Exception):
            pass

        def raise_my_exception():
            if False:
                for i in range(10):
                    print('nop')
            raise MyException()
        tt = cethread.CatchingExceptionThread(target=raise_my_exception)
        tt.start()
        self.assertRaises(MyException, tt.join)

    def test_join_around_exception(self):
        if False:
            for i in range(10):
                print('nop')
        resume = threading.Event()

        class MyException(Exception):
            pass

        def raise_my_exception():
            if False:
                print('Hello World!')
            resume.wait()
            raise MyException()
        tt = cethread.CatchingExceptionThread(target=raise_my_exception)
        tt.start()
        tt.join(timeout=0)
        self.assertIs(None, tt.exception)
        resume.set()
        self.assertRaises(MyException, tt.join)

    def test_sync_event(self):
        if False:
            print('Hello World!')
        control = threading.Event()
        in_thread = threading.Event()

        class MyException(Exception):
            pass

        def raise_my_exception():
            if False:
                while True:
                    i = 10
            control.wait()
            raise MyException()
        tt = cethread.CatchingExceptionThread(target=raise_my_exception, sync_event=in_thread)
        tt.start()
        tt.join(timeout=0)
        self.assertIs(None, tt.exception)
        self.assertIs(in_thread, tt.sync_event)
        control.set()
        self.assertRaises(MyException, tt.join)
        self.assertEqual(True, tt.sync_event.isSet())

    def test_switch_and_set(self):
        if False:
            for i in range(10):
                print('nop')
        'Caller can precisely control a thread.'
        control1 = threading.Event()
        control2 = threading.Event()
        control3 = threading.Event()

        class TestThread(cethread.CatchingExceptionThread):

            def __init__(self):
                if False:
                    return 10
                super(TestThread, self).__init__(target=self.step_by_step)
                self.current_step = 'starting'
                self.step1 = threading.Event()
                self.set_sync_event(self.step1)
                self.step2 = threading.Event()
                self.final = threading.Event()

            def step_by_step(self):
                if False:
                    print('Hello World!')
                control1.wait()
                self.current_step = 'step1'
                self.switch_and_set(self.step2)
                control2.wait()
                self.current_step = 'step2'
                self.switch_and_set(self.final)
                control3.wait()
                self.current_step = 'done'
        tt = TestThread()
        tt.start()
        self.assertEqual('starting', tt.current_step)
        control1.set()
        tt.step1.wait()
        self.assertEqual('step1', tt.current_step)
        control2.set()
        tt.step2.wait()
        self.assertEqual('step2', tt.current_step)
        control3.set()
        tt.join()
        self.assertEqual('done', tt.current_step)

    def test_exception_while_switch_and_set(self):
        if False:
            return 10
        control1 = threading.Event()

        class MyException(Exception):
            pass

        class TestThread(cethread.CatchingExceptionThread):

            def __init__(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                self.step1 = threading.Event()
                self.step2 = threading.Event()
                super(TestThread, self).__init__(target=self.step_by_step, sync_event=self.step1)
                self.current_step = 'starting'
                self.set_sync_event(self.step1)

            def step_by_step(self):
                if False:
                    for i in range(10):
                        print('nop')
                control1.wait()
                self.current_step = 'step1'
                self.switch_and_set(self.step2)

            def set_sync_event(self, event):
                if False:
                    print('Hello World!')
                if event is self.step2:
                    raise MyException()
                super(TestThread, self).set_sync_event(event)
        tt = TestThread()
        tt.start()
        self.assertEqual('starting', tt.current_step)
        control1.set()
        tt.step1.wait()
        self.assertRaises(MyException, tt.pending_exception)
        self.assertIs(tt.step1, tt.sync_event)
        self.assertTrue(tt.step1.isSet())