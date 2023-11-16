"""Test QtInProcessKernel"""
import unittest
from qtconsole.inprocess import QtInProcessKernelManager

class InProcessTests(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        'Open an in-process kernel.'
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel_client = self.kernel_manager.client()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        'Shutdown the in-process kernel. '
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()

    def test_execute(self):
        if False:
            for i in range(10):
                print('nop')
        'Test execution of shell commands.'
        assert not self.kernel_client.iopub_channel.closed()
        self.kernel_client.execute('a=1')
        assert self.kernel_manager.kernel.shell.user_ns.get('a') == 1