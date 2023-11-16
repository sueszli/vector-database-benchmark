import luigi
from luigi.contrib import scalding
import mock
import os
import random
import shutil
import tempfile
import unittest
import pytest

class MyScaldingTask(scalding.ScaldingJobTask):
    scala_source = luigi.Parameter()

    def source(self):
        if False:
            for i in range(10):
                print('nop')
        return self.scala_source

@pytest.mark.contrib
class ScaldingTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.scalding_home = os.path.join(tempfile.gettempdir(), 'scalding-%09d' % random.randint(0, 999999999))
        os.mkdir(self.scalding_home)
        self.lib_dir = os.path.join(self.scalding_home, 'lib')
        os.mkdir(self.lib_dir)
        os.mkdir(os.path.join(self.scalding_home, 'provided'))
        os.mkdir(os.path.join(self.scalding_home, 'libjars'))
        f = open(os.path.join(self.lib_dir, 'scalding-core-foo'), 'w')
        f.close()
        self.scala_source = os.path.join(self.scalding_home, 'my_source.scala')
        f = open(self.scala_source, 'w')
        f.write('class foo extends Job')
        f.close()
        os.environ['SCALDING_HOME'] = self.scalding_home

    def tearDown(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.scalding_home)

    @mock.patch('subprocess.check_call')
    @mock.patch('luigi.contrib.hadoop.run_and_track_hadoop_job')
    def test_scalding(self, check_call, track_job):
        if False:
            i = 10
            return i + 15
        success = luigi.run(['MyScaldingTask', '--scala-source', self.scala_source, '--local-scheduler', '--no-lock'])
        self.assertTrue(success)