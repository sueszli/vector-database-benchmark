import os
import signal
import unittest
from test import support
from test.support import script_helper

@unittest.skipUnless(os.name == 'posix', 'only supported on Unix')
class EINTRTests(unittest.TestCase):

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'requires setitimer()')
    def test_all(self):
        if False:
            i = 10
            return i + 15
        script = support.findfile('_test_eintr.py')
        script_helper.run_test_script(script)
if __name__ == '__main__':
    unittest.main()