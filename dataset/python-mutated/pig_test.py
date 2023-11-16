import subprocess
import tempfile
import luigi
from helpers import unittest
from luigi.contrib.pig import PigJobError, PigJobTask
from mock import patch
import pytest

class SimpleTestJob(PigJobTask):

    def output(self):
        if False:
            print('Hello World!')
        return luigi.LocalTarget('simple-output')

    def pig_script_path(self):
        if False:
            for i in range(10):
                print('nop')
        return 'my_simple_pig_script.pig'

class ComplexTestJob(PigJobTask):

    def output(self):
        if False:
            while True:
                i = 10
        return luigi.LocalTarget('complex-output')

    def pig_script_path(self):
        if False:
            i = 10
            return i + 15
        return 'my_complex_pig_script.pig'

    def pig_env_vars(self):
        if False:
            for i in range(10):
                print('nop')
        return {'PIG_CLASSPATH': '/your/path'}

    def pig_properties(self):
        if False:
            print('Hello World!')
        return {'pig.additional.jars': '/path/to/your/jar'}

    def pig_parameters(self):
        if False:
            i = 10
            return i + 15
        return {'YOUR_PARAM_NAME': 'Your param value'}

    def pig_options(self):
        if False:
            while True:
                i = 10
        return ['-x', 'local']

@pytest.mark.apache
class SimplePigTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    @patch('subprocess.Popen')
    def test_run__success(self, mock):
        if False:
            while True:
                i = 10
        arglist_result = []
        p = subprocess.Popen
        subprocess.Popen = _get_fake_Popen(arglist_result, 0)
        try:
            job = SimpleTestJob()
            job.run()
            self.assertEqual([['/usr/share/pig/bin/pig', '-f', 'my_simple_pig_script.pig']], arglist_result)
        finally:
            subprocess.Popen = p

    @patch('subprocess.Popen')
    def test_run__fail(self, mock):
        if False:
            print('Hello World!')
        arglist_result = []
        p = subprocess.Popen
        subprocess.Popen = _get_fake_Popen(arglist_result, 1)
        try:
            job = SimpleTestJob()
            job.run()
            self.assertEqual([['/usr/share/pig/bin/pig', '-f', 'my_simple_pig_script.pig']], arglist_result)
        except PigJobError as e:
            p = e
            self.assertEqual('stderr', p.err)
        else:
            self.fail('Should have thrown PigJobError')
        finally:
            subprocess.Popen = p

@pytest.mark.apache
class ComplexPigTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass

    @patch('subprocess.Popen')
    def test_run__success(self, mock):
        if False:
            while True:
                i = 10
        arglist_result = []
        p = subprocess.Popen
        subprocess.Popen = _get_fake_Popen(arglist_result, 0)
        with tempfile.NamedTemporaryFile(delete=False) as param_file_mock, tempfile.NamedTemporaryFile(delete=False) as prop_file_mock, patch('luigi.contrib.pig.tempfile.NamedTemporaryFile', side_effect=[param_file_mock, prop_file_mock]):
            try:
                job = ComplexTestJob()
                job.run()
                self.assertEqual([['/usr/share/pig/bin/pig', '-x', 'local', '-param_file', param_file_mock.name, '-propertyFile', prop_file_mock.name, '-f', 'my_complex_pig_script.pig']], arglist_result)
                with open(param_file_mock.name) as pparams_file:
                    pparams = pparams_file.readlines()
                    self.assertEqual(1, len(pparams))
                    self.assertEqual('YOUR_PARAM_NAME=Your param value\n', pparams[0])
                with open(prop_file_mock.name) as pprops_file:
                    pprops = pprops_file.readlines()
                    self.assertEqual(1, len(pprops))
                    self.assertEqual('pig.additional.jars=/path/to/your/jar\n', pprops[0])
            finally:
                subprocess.Popen = p

    @patch('subprocess.Popen')
    def test_run__fail(self, mock):
        if False:
            while True:
                i = 10
        arglist_result = []
        p = subprocess.Popen
        subprocess.Popen = _get_fake_Popen(arglist_result, 1)
        with tempfile.NamedTemporaryFile(delete=False) as param_file_mock, tempfile.NamedTemporaryFile(delete=False) as prop_file_mock, patch('luigi.contrib.pig.tempfile.NamedTemporaryFile', side_effect=[param_file_mock, prop_file_mock]):
            try:
                job = ComplexTestJob()
                job.run()
            except PigJobError as e:
                p = e
                self.assertEqual('stderr', p.err)
                self.assertEqual([['/usr/share/pig/bin/pig', '-x', 'local', '-param_file', param_file_mock.name, '-propertyFile', prop_file_mock.name, '-f', 'my_complex_pig_script.pig']], arglist_result)
                with open(param_file_mock.name) as pparams_file:
                    pparams = pparams_file.readlines()
                    self.assertEqual(1, len(pparams))
                    self.assertEqual('YOUR_PARAM_NAME=Your param value\n', pparams[0])
                with open(prop_file_mock.name) as pprops_file:
                    pprops = pprops_file.readlines()
                    self.assertEqual(1, len(pprops))
                    self.assertEqual('pig.additional.jars=/path/to/your/jar\n', pprops[0])
            else:
                self.fail('Should have thrown PigJobError')
            finally:
                subprocess.Popen = p

def _get_fake_Popen(arglist_result, return_code, *args, **kwargs):
    if False:
        print('Hello World!')

    def Popen_fake(arglist, shell=None, stdout=None, stderr=None, env=None, close_fds=True):
        if False:
            return 10
        arglist_result.append(arglist)

        class P:
            number_of_process_polls = 5

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self._process_polls_left = self.number_of_process_polls

            def wait(self):
                if False:
                    while True:
                        i = 10
                pass

            def poll(self):
                if False:
                    i = 10
                    return i + 15
                if self._process_polls_left:
                    self._process_polls_left -= 1
                    return None
                return 0

            def communicate(self):
                if False:
                    print('Hello World!')
                return 'end'

            def env(self):
                if False:
                    print('Hello World!')
                return self.env
        p = P()
        p.returncode = return_code
        p.stderr = tempfile.TemporaryFile()
        p.stdout = tempfile.TemporaryFile()
        p.stdout.write(b'stdout')
        p.stderr.write(b'stderr')
        p.stdout.seek(0)
        p.stderr.seek(0)
        return p
    return Popen_fake