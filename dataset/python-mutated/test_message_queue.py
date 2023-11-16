import os
import six
import time
import unittest
from pyspider.libs import utils
from six.moves import queue as Queue

class TestMessageQueue(object):

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def test_10_put(self):
        if False:
            return 10
        self.assertEqual(self.q1.qsize(), 0)
        self.assertEqual(self.q2.qsize(), 0)
        self.q1.put('TEST_DATA1', timeout=3)
        self.q1.put('TEST_DATA2_中文', timeout=3)
        time.sleep(0.01)
        self.assertEqual(self.q1.qsize(), 2)
        self.assertEqual(self.q2.qsize(), 2)

    def test_20_get(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.q1.get(timeout=0.01), 'TEST_DATA1')
        self.assertEqual(self.q2.get_nowait(), 'TEST_DATA2_中文')
        with self.assertRaises(Queue.Empty):
            self.q2.get(timeout=0.01)
        with self.assertRaises(Queue.Empty):
            self.q2.get_nowait()

    def test_30_full(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.q1.qsize(), 0)
        self.assertEqual(self.q2.qsize(), 0)
        for i in range(2):
            self.q1.put_nowait('TEST_DATA%d' % i)
        for i in range(3):
            self.q2.put('TEST_DATA%d' % i)
        with self.assertRaises(Queue.Full):
            self.q1.put('TEST_DATA6', timeout=0.01)
        with self.assertRaises(Queue.Full):
            self.q1.put_nowait('TEST_DATA6')

    def test_40_multiple_threading_error(self):
        if False:
            for i in range(10):
                print('nop')

        def put(q):
            if False:
                i = 10
                return i + 15
            for i in range(100):
                q.put('DATA_%d' % i)

        def get(q):
            if False:
                for i in range(10):
                    print('nop')
            for i in range(100):
                q.get()
        t = utils.run_in_thread(put, self.q3)
        get(self.q3)
        t.join()

class BuiltinQueue(TestMessageQueue, unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            print('Hello World!')
        from pyspider.message_queue import connect_message_queue
        with utils.timeout(3):
            self.q1 = self.q2 = connect_message_queue('test_queue', maxsize=5)
            self.q3 = connect_message_queue('test_queue_for_threading_test')

@unittest.skipIf(os.environ.get('IGNORE_RABBITMQ') or os.environ.get('IGNORE_ALL'), 'no rabbitmq server for test.')
class TestPikaRabbitMQ(TestMessageQueue, unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        from pyspider.message_queue import rabbitmq
        with utils.timeout(3):
            self.q1 = rabbitmq.PikaQueue('test_queue', maxsize=5, lazy_limit=False)
            self.q2 = rabbitmq.PikaQueue('test_queue', amqp_url='amqp://localhost:5672/%2F', maxsize=5, lazy_limit=False)
            self.q3 = rabbitmq.PikaQueue('test_queue_for_threading_test', amqp_url='amqp://guest:guest@localhost:5672/', lazy_limit=False)
        self.q2.delete()
        self.q2.reconnect()
        self.q3.delete()
        self.q3.reconnect()

    @classmethod
    def tearDownClass(self):
        if False:
            for i in range(10):
                print('nop')
        self.q2.delete()
        self.q3.delete()
        del self.q1
        del self.q2
        del self.q3

    def test_30_full(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.q1.qsize(), 0)
        self.assertEqual(self.q2.qsize(), 0)
        for i in range(2):
            self.q1.put_nowait('TEST_DATA%d' % i)
        for i in range(3):
            self.q2.put('TEST_DATA%d' % i)
        print(self.q1.__dict__)
        print(self.q1.qsize())
        with self.assertRaises(Queue.Full):
            self.q1.put_nowait('TEST_DATA6')
        print(self.q1.__dict__)
        print(self.q1.qsize())
        with self.assertRaises(Queue.Full):
            self.q1.put('TEST_DATA6', timeout=0.01)

@unittest.skipIf(six.PY3, 'Python 3 now using Pika')
@unittest.skipIf(os.environ.get('IGNORE_RABBITMQ') or os.environ.get('IGNORE_ALL'), 'no rabbitmq server for test.')
class TestAmqpRabbitMQ(TestMessageQueue, unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        from pyspider.message_queue import connect_message_queue
        with utils.timeout(3):
            self.q1 = connect_message_queue('test_queue', 'amqp://localhost:5672/', maxsize=5, lazy_limit=False)
            self.q2 = connect_message_queue('test_queue', 'amqp://localhost:5672/%2F', maxsize=5, lazy_limit=False)
            self.q3 = connect_message_queue('test_queue_for_threading_test', 'amqp://guest:guest@localhost:5672/', lazy_limit=False)
        self.q2.delete()
        self.q2.reconnect()
        self.q3.delete()
        self.q3.reconnect()

    @classmethod
    def tearDownClass(self):
        if False:
            i = 10
            return i + 15
        self.q2.delete()
        self.q3.delete()
        del self.q1
        del self.q2
        del self.q3

    def test_30_full(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.q1.qsize(), 0)
        self.assertEqual(self.q2.qsize(), 0)
        for i in range(2):
            self.q1.put_nowait('TEST_DATA%d' % i)
        for i in range(3):
            self.q2.put('TEST_DATA%d' % i)
        print(self.q1.__dict__)
        print(self.q1.qsize())
        with self.assertRaises(Queue.Full):
            self.q1.put('TEST_DATA6', timeout=0.01)
        print(self.q1.__dict__)
        print(self.q1.qsize())
        with self.assertRaises(Queue.Full):
            self.q1.put_nowait('TEST_DATA6')

@unittest.skipIf(os.environ.get('IGNORE_REDIS') or os.environ.get('IGNORE_ALL'), 'no redis server for test.')
class TestRedisQueue(TestMessageQueue, unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            print('Hello World!')
        from pyspider.message_queue import connect_message_queue
        from pyspider.message_queue import redis_queue
        with utils.timeout(3):
            self.q1 = redis_queue.RedisQueue('test_queue', maxsize=5, lazy_limit=False)
            self.q2 = redis_queue.RedisQueue('test_queue', maxsize=5, lazy_limit=False)
            self.q3 = connect_message_queue('test_queue_for_threading_test', 'redis://localhost:6379/')
            while not self.q1.empty():
                self.q1.get()
            while not self.q2.empty():
                self.q2.get()
            while not self.q3.empty():
                self.q3.get()

    @classmethod
    def tearDownClass(self):
        if False:
            i = 10
            return i + 15
        while not self.q1.empty():
            self.q1.get()
        while not self.q2.empty():
            self.q2.get()
        while not self.q3.empty():
            self.q3.get()

class TestKombuQueue(TestMessageQueue, unittest.TestCase):
    kombu_url = 'kombu+memory://'

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        from pyspider.message_queue import connect_message_queue
        with utils.timeout(3):
            self.q1 = connect_message_queue('test_queue', self.kombu_url, maxsize=5, lazy_limit=False)
            self.q2 = connect_message_queue('test_queue', self.kombu_url, maxsize=5, lazy_limit=False)
            self.q3 = connect_message_queue('test_queue_for_threading_test', self.kombu_url, lazy_limit=False)
            while not self.q1.empty():
                self.q1.get()
            while not self.q2.empty():
                self.q2.get()
            while not self.q3.empty():
                self.q3.get()

    @classmethod
    def tearDownClass(self):
        if False:
            while True:
                i = 10
        while not self.q1.empty():
            self.q1.get()
        self.q1.delete()
        while not self.q2.empty():
            self.q2.get()
        self.q2.delete()
        while not self.q3.empty():
            self.q3.get()
        self.q3.delete()

@unittest.skip('test cannot pass, get is buffered')
@unittest.skipIf(os.environ.get('IGNORE_RABBITMQ') or os.environ.get('IGNORE_ALL'), 'no rabbitmq server for test.')
class TestKombuAmpqQueue(TestKombuQueue):
    kombu_url = 'kombu+amqp://'

@unittest.skip('test cannot pass, put is buffered')
@unittest.skipIf(os.environ.get('IGNORE_REDIS') or os.environ.get('IGNORE_ALL'), 'no redis server for test.')
class TestKombuRedisQueue(TestKombuQueue):
    kombu_url = 'kombu+redis://'

@unittest.skipIf(os.environ.get('IGNORE_MONGODB') or os.environ.get('IGNORE_ALL'), 'no mongodb server for test.')
class TestKombuMongoDBQueue(TestKombuQueue):
    kombu_url = 'kombu+mongodb://'