import datetime
from helpers import unittest
import luigi
import luigi.interface
from luigi.mock import MockTarget

class Popularity(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today() - datetime.timedelta(1))

    def output(self):
        if False:
            i = 10
            return i + 15
        return MockTarget('/tmp/popularity/%s.txt' % self.date.strftime('%Y-%m-%d'))

    def requires(self):
        if False:
            return 10
        return Popularity(self.date - datetime.timedelta(1))

    def run(self):
        if False:
            return 10
        f = self.output().open('w')
        for line in self.input().open('r'):
            print(int(line.strip()) + 1, file=f)
        f.close()

class RecursionTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        MockTarget.fs.get_all_data()['/tmp/popularity/2009-01-01.txt'] = b'0\n'

    def test_invoke(self):
        if False:
            i = 10
            return i + 15
        luigi.build([Popularity(datetime.date(2009, 1, 5))], local_scheduler=True)
        self.assertEqual(MockTarget.fs.get_data('/tmp/popularity/2009-01-05.txt'), b'4\n')