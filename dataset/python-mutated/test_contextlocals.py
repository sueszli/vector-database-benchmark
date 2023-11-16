"""
Some objects are context-local, meaning that they have different values depending on the context they are accessed from. A context is currently defined as a thread.
"""
import unittest
import bottle
import threading

def run_thread(func):
    if False:
        print('Hello World!')
    t = threading.Thread(target=func)
    t.start()
    t.join()

class TestThreadLocals(unittest.TestCase):

    def test_request(self):
        if False:
            i = 10
            return i + 15
        e1 = {'PATH_INFO': '/t1'}
        e2 = {'PATH_INFO': '/t2'}

        def run():
            if False:
                for i in range(10):
                    print('nop')
            bottle.request.bind(e2)
            self.assertEqual(bottle.request.path, '/t2')
        bottle.request.bind(e1)
        self.assertEqual(bottle.request.path, '/t1')
        run_thread(run)
        self.assertEqual(bottle.request.path, '/t1')

    def test_response(self):
        if False:
            print('Hello World!')

        def run():
            if False:
                return 10
            bottle.response.bind()
            bottle.response.content_type = 'test/thread'
            self.assertEqual(bottle.response.headers['Content-Type'], 'test/thread')
        bottle.response.bind()
        bottle.response.content_type = 'test/main'
        self.assertEqual(bottle.response.headers['Content-Type'], 'test/main')
        run_thread(run)
        self.assertEqual(bottle.response.headers['Content-Type'], 'test/main')