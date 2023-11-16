import unittest
from io import StringIO
from time import sleep, time
from unittest import mock
from twisted.trial.unittest import SkipTest
from scrapy.utils import trackref

class Foo(trackref.object_ref):
    pass

class Bar(trackref.object_ref):
    pass

class TrackrefTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        trackref.live_refs.clear()

    def test_format_live_refs(self):
        if False:
            print('Hello World!')
        o1 = Foo()
        o2 = Bar()
        o3 = Foo()
        self.assertEqual(trackref.format_live_refs(), 'Live References\n\nBar                                 1   oldest: 0s ago\nFoo                                 2   oldest: 0s ago\n')
        self.assertEqual(trackref.format_live_refs(ignore=Foo), 'Live References\n\nBar                                 1   oldest: 0s ago\n')

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_print_live_refs_empty(self, stdout):
        if False:
            print('Hello World!')
        trackref.print_live_refs()
        self.assertEqual(stdout.getvalue(), 'Live References\n\n\n')

    @mock.patch('sys.stdout', new_callable=StringIO)
    def test_print_live_refs_with_objects(self, stdout):
        if False:
            while True:
                i = 10
        o1 = Foo()
        trackref.print_live_refs()
        self.assertEqual(stdout.getvalue(), 'Live References\n\nFoo                                 1   oldest: 0s ago\n\n')

    def test_get_oldest(self):
        if False:
            while True:
                i = 10
        o1 = Foo()
        o1_time = time()
        o2 = Bar()
        o3_time = time()
        if o3_time <= o1_time:
            sleep(0.01)
            o3_time = time()
        if o3_time <= o1_time:
            raise SkipTest('time.time is not precise enough')
        o3 = Foo()
        self.assertIs(trackref.get_oldest('Foo'), o1)
        self.assertIs(trackref.get_oldest('Bar'), o2)
        self.assertIsNone(trackref.get_oldest('XXX'))

    def test_iter_all(self):
        if False:
            while True:
                i = 10
        o1 = Foo()
        o2 = Bar()
        o3 = Foo()
        self.assertEqual(set(trackref.iter_all('Foo')), {o1, o3})