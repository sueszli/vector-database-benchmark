import datetime
from helpers import unittest
import luigi
import luigi.notifications
from luigi.mock import MockTarget
from luigi.util import inherits
luigi.notifications.DEBUG = True

class A(luigi.Task):
    task_namespace = 'wrap'

    def output(self):
        if False:
            while True:
                i = 10
        return MockTarget('/tmp/a.txt')

    def run(self):
        if False:
            return 10
        f = self.output().open('w')
        print('hello, world', file=f)
        f.close()

class B(luigi.Task):
    date = luigi.DateParameter()

    def output(self):
        if False:
            i = 10
            return i + 15
        return MockTarget(self.date.strftime('/tmp/b-%Y-%m-%d.txt'))

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        f = self.output().open('w')
        print('goodbye, space', file=f)
        f.close()

def XMLWrapper(cls):
    if False:
        print('Hello World!')

    @inherits(cls)
    class XMLWrapperCls(luigi.Task):

        def requires(self):
            if False:
                return 10
            return self.clone_parent()

        def run(self):
            if False:
                for i in range(10):
                    print('nop')
            f = self.input().open('r')
            g = self.output().open('w')
            print('<?xml version="1.0" ?>', file=g)
            for line in f:
                print('<dummy-xml>' + line.strip() + '</dummy-xml>', file=g)
            g.close()
    return XMLWrapperCls

class AXML(XMLWrapper(A)):

    def output(self):
        if False:
            print('Hello World!')
        return MockTarget('/tmp/a.xml')

class BXML(XMLWrapper(B)):

    def output(self):
        if False:
            return 10
        return MockTarget(self.date.strftime('/tmp/b-%Y-%m-%d.xml'))

class WrapperTest(unittest.TestCase):
    """ This test illustrates how a task class can wrap another task class by modifying its behavior.

    See instance_wrap_test.py for an example of how instances can wrap each other. """
    workers = 1

    def setUp(self):
        if False:
            return 10
        MockTarget.fs.clear()

    def test_a(self):
        if False:
            for i in range(10):
                print('nop')
        luigi.build([AXML()], local_scheduler=True, no_lock=True, workers=self.workers)
        self.assertEqual(MockTarget.fs.get_data('/tmp/a.xml'), b'<?xml version="1.0" ?>\n<dummy-xml>hello, world</dummy-xml>\n')

    def test_b(self):
        if False:
            i = 10
            return i + 15
        luigi.build([BXML(datetime.date(2012, 1, 1))], local_scheduler=True, no_lock=True, workers=self.workers)
        self.assertEqual(MockTarget.fs.get_data('/tmp/b-2012-01-01.xml'), b'<?xml version="1.0" ?>\n<dummy-xml>goodbye, space</dummy-xml>\n')

class WrapperWithMultipleWorkersTest(WrapperTest):
    workers = 7