"""
Tests for L{twisted.runner.procmontap}.
"""
from twisted.python.usage import UsageError
from twisted.runner import procmontap as tap
from twisted.runner.procmon import ProcessMonitor
from twisted.trial import unittest

class ProcessMonitorTapTests(unittest.TestCase):
    """
    Tests for L{twisted.runner.procmontap}'s option parsing and makeService
    method.
    """

    def test_commandLineRequired(self) -> None:
        if False:
            print('Hello World!')
        '\n        The command line arguments must be provided.\n        '
        opt = tap.Options()
        self.assertRaises(UsageError, opt.parseOptions, [])

    def test_threshold(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The threshold option is recognised as a parameter and coerced to\n        float.\n        '
        opt = tap.Options()
        opt.parseOptions(['--threshold', '7.5', 'foo'])
        self.assertEqual(opt['threshold'], 7.5)

    def test_killTime(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The killtime option is recognised as a parameter and coerced to float.\n        '
        opt = tap.Options()
        opt.parseOptions(['--killtime', '7.5', 'foo'])
        self.assertEqual(opt['killtime'], 7.5)

    def test_minRestartDelay(self) -> None:
        if False:
            return 10
        '\n        The minrestartdelay option is recognised as a parameter and coerced to\n        float.\n        '
        opt = tap.Options()
        opt.parseOptions(['--minrestartdelay', '7.5', 'foo'])
        self.assertEqual(opt['minrestartdelay'], 7.5)

    def test_maxRestartDelay(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        The maxrestartdelay option is recognised as a parameter and coerced to\n        float.\n        '
        opt = tap.Options()
        opt.parseOptions(['--maxrestartdelay', '7.5', 'foo'])
        self.assertEqual(opt['maxrestartdelay'], 7.5)

    def test_parameterDefaults(self) -> None:
        if False:
            print('Hello World!')
        '\n        The parameters all have default values\n        '
        opt = tap.Options()
        opt.parseOptions(['foo'])
        self.assertEqual(opt['threshold'], 1)
        self.assertEqual(opt['killtime'], 5)
        self.assertEqual(opt['minrestartdelay'], 1)
        self.assertEqual(opt['maxrestartdelay'], 3600)

    def test_makeService(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The command line gets added as a process to the ProcessMontor.\n        '
        opt = tap.Options()
        opt.parseOptions(['ping', '-c', '3', '8.8.8.8'])
        s = tap.makeService(opt)
        self.assertIsInstance(s, ProcessMonitor)
        self.assertIn('ping -c 3 8.8.8.8', s.processes)