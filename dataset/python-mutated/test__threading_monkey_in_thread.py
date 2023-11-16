from __future__ import print_function
import threading
from gevent import monkey
import gevent.testing as greentest

class Test(greentest.TestCase):

    @greentest.ignores_leakcheck
    def test_patch_in_thread(self):
        if False:
            while True:
                i = 10
        all_warnings = []
        try:
            get_ident = threading.get_ident
        except AttributeError:
            get_ident = threading._get_ident

        def process_warnings(warnings):
            if False:
                for i in range(10):
                    print('nop')
            all_warnings.extend(warnings)
        monkey._process_warnings = process_warnings
        current = threading.current_thread()
        current_id = get_ident()

        def target():
            if False:
                print('Hello World!')
            tcurrent = threading.current_thread()
            monkey.patch_all()
            tcurrent2 = threading.current_thread()
            self.assertIsNot(tcurrent, current)
            self.assertIsNot(tcurrent, tcurrent2)
        thread = threading.Thread(target=target)
        thread.start()
        try:
            thread.join()
        except:
            greentest.reraiseFlakyTestRaceCondition()
        self.assertNotIsInstance(current, threading._DummyThread)
        self.assertIsInstance(current, monkey.get_original('threading', 'Thread'))
        if greentest.PY3:
            self.assertEqual(all_warnings, ['Monkey-patching outside the main native thread. Some APIs will not be available. Expect a KeyError to be printed at shutdown.', 'Monkey-patching not on the main thread; threading.main_thread().join() will hang from a greenlet'])
        else:
            self.assertEqual(all_warnings, ['Monkey-patching outside the main native thread. Some APIs will not be available. Expect a KeyError to be printed at shutdown.'])
        del threading._active[current_id]
        threading._active[(getattr(threading, 'get_ident', None) or threading._get_ident)()] = current
if __name__ == '__main__':
    greentest.main()