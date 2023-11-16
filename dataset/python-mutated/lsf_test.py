"""
LSF Unit Test
=============

Test runner for the LSF wrapper. The test is based on the one used for the SGE
wrappers
"""
import subprocess
import os
import os.path
from glob import glob
import unittest
import logging
from mock import patch
import luigi
from luigi.contrib.lsf import LSFJobTask
import pytest
DEFAULT_HOME = ''
LOGGER = logging.getLogger('luigi-interface')

def on_lsf_master():
    if False:
        for i in range(10):
            print('nop')
    try:
        subprocess.check_call('bjobs', shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

class TestJobTask(LSFJobTask):
    """Simple SGE job: write a test file to NSF shared drive and waits a minute"""
    i = luigi.Parameter()

    def work(self):
        if False:
            for i in range(10):
                print('nop')
        LOGGER.info('Running test job...')
        with open(self.output().path, 'w') as f:
            f.write('this is a test\n')

    def output(self):
        if False:
            while True:
                i = 10
        return luigi.LocalTarget(os.path.join(DEFAULT_HOME, 'test_lsf_file_' + str(self.i)))

@pytest.mark.contrib
class TestSGEJob(unittest.TestCase):
    """Test from SGE master node"""

    @patch('subprocess.Popen')
    @patch('subprocess.Popen.communicate')
    def test_run_job(self, mock_open, mock_communicate):
        if False:
            for i in range(10):
                print('nop')
        if on_lsf_master():
            outfile = os.path.join(DEFAULT_HOME, 'testfile_1')
            tasks = [TestJobTask(i=str(i), n_cpu_flag=1) for i in range(3)]
            luigi.build(tasks, local_scheduler=True, workers=3)
            self.assertTrue(os.path.exists(outfile))

    @patch('subprocess.Popen')
    @patch('subprocess.Popen.communicate')
    def test_run_job_with_dump(self, mock_open, mock_communicate):
        if False:
            while True:
                i = 10
        mock_open.side_effect = ['Job <1000001> is submitted to queue <queue-name>.', '']
        task = TestJobTask(i=str(1), n_cpu_flag=1, shared_tmp_dir='/tmp')
        luigi.build([task], local_scheduler=True)
        self.assertEqual(mock_open.call_count, 0)

    def tearDown(self):
        if False:
            return 10
        for fpath in glob(os.path.join(DEFAULT_HOME, 'test_lsf_file_*')):
            try:
                os.remove(fpath)
            except OSError:
                pass
if __name__ == '__main__':
    unittest.main()