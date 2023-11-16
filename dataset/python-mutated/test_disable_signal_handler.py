import signal
import subprocess
import unittest
SignalsToTest = {signal.SIGTERM, signal.SIGBUS, signal.SIGABRT, signal.SIGSEGV, signal.SIGILL, signal.SIGFPE}

class TestSignOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10
        try:
            for sig in SignalsToTest:
                output = subprocess.check_output(['python', '-c', f'import paddle; import signal,os; paddle.disable_signal_handler(); os.kill(os.getpid(), {sig})'], stderr=subprocess.STDOUT)
        except Exception as e:
            stdout_message = str(e.output)
            if 'paddle::framework::SignalHandle' in stdout_message:
                raise Exception('Paddle signal handler not disabled')
if __name__ == '__main__':
    unittest.main()