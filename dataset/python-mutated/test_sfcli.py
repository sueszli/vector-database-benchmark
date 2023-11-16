import subprocess
import sys
import unittest

class TestSfcli(unittest.TestCase):
    """
    Test TestSfcli
    """

    def execute(self, command):
        if False:
            for i in range(10):
                print('nop')
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = proc.communicate()
        return (out, err, proc.returncode)

    def test_help_arg_should_print_help_and_exit(self):
        if False:
            i = 10
            return i + 15
        (out, err, code) = self.execute([sys.executable, 'sfcli.py', '-h'])
        self.assertIn(b'show this help message and exit', out)
        self.assertEqual(b'', err)
        self.assertEqual(0, code)