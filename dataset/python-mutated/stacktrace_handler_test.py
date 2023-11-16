"""Test to make sure stack trace is generated in case of test failures."""
import argparse
import os
import signal
import subprocess
import sys
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
FLAGS = None
_CHILD_FLAG_HELP = 'Boolean. Set to true if this is the child process.'

class StacktraceHandlerTest(test.TestCase):

    def testChildProcessKillsItself(self):
        if False:
            while True:
                i = 10
        if FLAGS.child:
            os.kill(os.getpid(), signal.SIGABRT)

    def testGeneratesStacktrace(self):
        if False:
            i = 10
            return i + 15
        if FLAGS.child:
            return
        if sys.executable:
            child_process = subprocess.Popen([sys.executable, sys.argv[0], '--child=True'], cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            child_process = subprocess.Popen([sys.argv[0], '--child=True'], cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (child_stdout, child_stderr) = child_process.communicate()
        child_output = child_stdout + child_stderr
        child_process.wait()
        logging.info('Output from the child process:')
        logging.info(child_output)
        self.assertIn(b'PyEval_EvalFrame', child_output)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=bool, default=False, help=_CHILD_FLAG_HELP)
    (FLAGS, unparsed) = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unparsed
    test.main()