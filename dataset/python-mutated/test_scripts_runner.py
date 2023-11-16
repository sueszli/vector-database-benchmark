from __future__ import absolute_import
from __future__ import print_function
import os
import sys
from twisted.python import log
from twisted.python import usage
from twisted.trial import unittest
from buildbot_worker.scripts import runner
from buildbot_worker.test.util import misc
try:
    from unittest import mock
except ImportError:
    import mock

class OptionsMixin(object):

    def assertOptions(self, opts, exp):
        if False:
            return 10
        got = {k: opts[k] for k in exp}
        if got != exp:
            msg = []
            for k in exp:
                if opts[k] != exp[k]:
                    msg.append(' {0}: expected {1!r}, got {2!r}'.format(k, exp[k], opts[k]))
            self.fail('did not get expected options\n' + '\n'.join(msg))

class BaseDirTestsMixin(object):
    """
    Common tests for Options classes with 'basedir' parameter
    """
    GETCWD_PATH = 'test-dir'
    ABSPATH_PREFIX = 'test-prefix-'
    MY_BASEDIR = 'my-basedir'
    options_class = None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch(os, 'getcwd', lambda : self.GETCWD_PATH)
        self.patch(os.path, 'abspath', lambda path: self.ABSPATH_PREFIX + path)

    def parse(self, *args):
        if False:
            for i in range(10):
                print('nop')
        assert self.options_class is not None
        opts = self.options_class()
        opts.parseOptions(args)
        return opts

    def test_defaults(self):
        if False:
            return 10
        opts = self.parse()
        self.assertEqual(opts['basedir'], self.ABSPATH_PREFIX + self.GETCWD_PATH, 'unexpected basedir path')

    def test_basedir_arg(self):
        if False:
            print('Hello World!')
        opts = self.parse(self.MY_BASEDIR)
        self.assertEqual(opts['basedir'], self.ABSPATH_PREFIX + self.MY_BASEDIR, 'unexpected basedir path')

    def test_too_many_args(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(usage.UsageError, "I wasn't expecting so many arguments"):
            self.parse('arg1', 'arg2')

class TestMakerBase(BaseDirTestsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.runner.MakerBase class.
    """
    options_class = runner.MakerBase

class TestStopOptions(BaseDirTestsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.runner.StopOptions class.
    """
    options_class = runner.StopOptions

    def test_synopsis(self):
        if False:
            for i in range(10):
                print('nop')
        opts = runner.StopOptions()
        self.assertIn('buildbot-worker stop', opts.getSynopsis())

class TestStartOptions(OptionsMixin, BaseDirTestsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.runner.StartOptions class.
    """
    options_class = runner.StartOptions

    def test_synopsis(self):
        if False:
            while True:
                i = 10
        opts = runner.StartOptions()
        self.assertIn('buildbot-worker start', opts.getSynopsis())

    def test_all_args(self):
        if False:
            while True:
                i = 10
        opts = self.parse('--quiet', '--nodaemon', self.MY_BASEDIR)
        self.assertOptions(opts, {'quiet': True, 'nodaemon': True, 'basedir': self.ABSPATH_PREFIX + self.MY_BASEDIR})

class TestRestartOptions(OptionsMixin, BaseDirTestsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.runner.RestartOptions class.
    """
    options_class = runner.RestartOptions

    def test_synopsis(self):
        if False:
            while True:
                i = 10
        opts = runner.RestartOptions()
        self.assertIn('buildbot-worker restart', opts.getSynopsis())

    def test_all_args(self):
        if False:
            return 10
        opts = self.parse('--quiet', '--nodaemon', self.MY_BASEDIR)
        self.assertOptions(opts, {'quiet': True, 'nodaemon': True, 'basedir': self.ABSPATH_PREFIX + self.MY_BASEDIR})

class TestCreateWorkerOptions(OptionsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.runner.CreateWorkerOptions class.
    """
    req_args = ['bdir', 'mstr:5678', 'name', 'pswd']

    def parse(self, *args):
        if False:
            for i in range(10):
                print('nop')
        opts = runner.CreateWorkerOptions()
        opts.parseOptions(args)
        return opts

    def test_defaults(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(usage.UsageError, 'incorrect number of arguments'):
            self.parse()

    def test_synopsis(self):
        if False:
            return 10
        opts = runner.CreateWorkerOptions()
        self.assertIn('buildbot-worker create-worker', opts.getSynopsis())

    def test_min_args(self):
        if False:
            print('Hello World!')
        self.patch(runner.MakerBase, 'postOptions', mock.Mock())
        self.assertOptions(self.parse(*self.req_args), {'basedir': 'bdir', 'host': 'mstr', 'port': 5678, 'name': 'name', 'passwd': 'pswd'})

    def test_all_args(self):
        if False:
            return 10
        self.patch(runner.MakerBase, 'postOptions', mock.Mock())
        opts = self.parse('--force', '--relocatable', '--no-logrotate', '--keepalive=4', '--umask=0o22', '--maxdelay=3', '--numcpus=4', '--log-size=2', '--log-count=1', '--allow-shutdown=file', *self.req_args)
        self.assertOptions(opts, {'force': True, 'relocatable': True, 'no-logrotate': True, 'umask': '0o22', 'maxdelay': 3, 'numcpus': '4', 'log-size': 2, 'log-count': '1', 'allow-shutdown': 'file', 'basedir': 'bdir', 'host': 'mstr', 'port': 5678, 'name': 'name', 'passwd': 'pswd'})

    def test_master_url(self):
        if False:
            return 10
        with self.assertRaisesRegex(usage.UsageError, '<master> is not a URL - do not use URL'):
            self.parse('a', 'http://b.c', 'd', 'e')

    def test_inv_keepalive(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(usage.UsageError, 'keepalive parameter needs to be a number'):
            self.parse('--keepalive=X', *self.req_args)

    def test_inv_maxdelay(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(usage.UsageError, 'maxdelay parameter needs to be a number'):
            self.parse('--maxdelay=X', *self.req_args)

    def test_inv_log_size(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(usage.UsageError, 'log-size parameter needs to be a number'):
            self.parse('--log-size=X', *self.req_args)

    def test_inv_log_count(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(usage.UsageError, 'log-count parameter needs to be a number or None'):
            self.parse('--log-count=X', *self.req_args)

    def test_inv_numcpus(self):
        if False:
            return 10
        with self.assertRaisesRegex(usage.UsageError, 'numcpus parameter needs to be a number or None'):
            self.parse('--numcpus=X', *self.req_args)

    def test_inv_umask(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(usage.UsageError, 'umask parameter needs to be a number or None'):
            self.parse('--umask=X', *self.req_args)

    def test_inv_umask2(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(usage.UsageError, 'umask parameter needs to be a number or None'):
            self.parse('--umask=022', *self.req_args)

    def test_inv_allow_shutdown(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(usage.UsageError, "allow-shutdown needs to be one of 'signal' or 'file'"):
            self.parse('--allow-shutdown=X', *self.req_args)

    def test_too_few_args(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(usage.UsageError, 'incorrect number of arguments'):
            self.parse('arg1', 'arg2')

    def test_too_many_args(self):
        if False:
            return 10
        with self.assertRaisesRegex(usage.UsageError, 'incorrect number of arguments'):
            self.parse('extra_arg', *self.req_args)

    def test_validateMasterArgument_no_port(self):
        if False:
            while True:
                i = 10
        '\n        test calling CreateWorkerOptions.validateMasterArgument()\n        on <master> argument without port specified.\n        '
        opts = runner.CreateWorkerOptions()
        self.assertEqual(opts.validateMasterArgument('mstrhost'), ('mstrhost', 9989), 'incorrect master host and/or port')

    def test_validateMasterArgument_empty_master(self):
        if False:
            print('Hello World!')
        '\n        test calling CreateWorkerOptions.validateMasterArgument()\n        on <master> without host part specified.\n        '
        opts = runner.CreateWorkerOptions()
        with self.assertRaisesRegex(usage.UsageError, "invalid <master> argument ':1234'"):
            opts.validateMasterArgument(':1234')

    def test_validateMasterArgument_inv_port(self):
        if False:
            while True:
                i = 10
        '\n        test calling CreateWorkerOptions.validateMasterArgument()\n        on <master> without with unparsable port part\n        '
        opts = runner.CreateWorkerOptions()
        with self.assertRaisesRegex(usage.UsageError, "invalid master port 'apple', needs to be a number"):
            opts.validateMasterArgument('host:apple')

    def test_validateMasterArgument_ok(self):
        if False:
            print('Hello World!')
        '\n        test calling CreateWorkerOptions.validateMasterArgument()\n        on <master> with host and port parts specified.\n        '
        opts = runner.CreateWorkerOptions()
        self.assertEqual(opts.validateMasterArgument('mstrhost:4321'), ('mstrhost', 4321), 'incorrect master host and/or port')

    def test_validateMasterArgument_ipv4(self):
        if False:
            i = 10
            return i + 15
        '\n        test calling CreateWorkerOptions.validateMasterArgument()\n        on <master> with ipv4 host specified.\n        '
        opts = runner.CreateWorkerOptions()
        self.assertEqual(opts.validateMasterArgument('192.168.0.0'), ('192.168.0.0', 9989), 'incorrect master host and/or port')

    def test_validateMasterArgument_ipv4_port(self):
        if False:
            while True:
                i = 10
        '\n        test calling CreateWorkerOptions.validateMasterArgument()\n        on <master> with ipv4 host and port parts specified.\n        '
        opts = runner.CreateWorkerOptions()
        self.assertEqual(opts.validateMasterArgument('192.168.0.0:4321'), ('192.168.0.0', 4321), 'incorrect master host and/or port')

    def test_validateMasterArgument_ipv6(self):
        if False:
            print('Hello World!')
        '\n        test calling CreateWorkerOptions.validateMasterArgument()\n        on <master> with ipv6 host specified.\n        '
        opts = runner.CreateWorkerOptions()
        self.assertEqual(opts.validateMasterArgument('[2001:1:2:3:4::1]'), ('2001:1:2:3:4::1', 9989), 'incorrect master host and/or port')

    def test_validateMasterArgument_ipv6_port(self):
        if False:
            print('Hello World!')
        '\n        test calling CreateWorkerOptions.validateMasterArgument()\n        on <master> with ipv6 host and port parts specified.\n        '
        opts = runner.CreateWorkerOptions()
        self.assertEqual(opts.validateMasterArgument('[2001:1:2:3:4::1]:4321'), ('2001:1:2:3:4::1', 4321), 'incorrect master host and/or port')

    def test_validateMasterArgument_ipv6_no_bracket(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test calling CreateWorkerOptions.validateMasterArgument()\n        on <master> with ipv6 without [] specified.\n        '
        opts = runner.CreateWorkerOptions()
        with self.assertRaisesRegex(usage.UsageError, "invalid <master> argument '2001:1:2:3:4::1:4321', if it is an ipv6 address, it must be enclosed by \\[\\]"):
            opts.validateMasterArgument('2001:1:2:3:4::1:4321')

class TestOptions(misc.StdoutAssertionsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.runner.Options class.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setUpStdoutAssertions()

    def parse(self, *args):
        if False:
            for i in range(10):
                print('nop')
        opts = runner.Options()
        opts.parseOptions(args)
        return opts

    def test_defaults(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(usage.UsageError, 'must specify a command'):
            self.parse()

    def test_version(self):
        if False:
            while True:
                i = 10
        exception = self.assertRaises(SystemExit, self.parse, '--version')
        self.assertEqual(exception.code, 0, 'unexpected exit code')
        self.assertInStdout('worker version:')

    def test_verbose(self):
        if False:
            while True:
                i = 10
        self.patch(log, 'startLogging', mock.Mock())
        with self.assertRaises(usage.UsageError):
            self.parse('--verbose')
        log.startLogging.assert_called_once_with(sys.stderr)
functionPlaceholder = None

class TestRun(misc.StdoutAssertionsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.runner.run()
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.setUpStdoutAssertions()

    class TestSubCommand(usage.Options):
        subcommandFunction = __name__ + '.functionPlaceholder'
        optFlags = [['test-opt', None, None]]

    class TestOptions(usage.Options):
        """
        Option class that emulates usage error. The 'suboptions' flag
        enables emulation of usage error in a sub-option.
        """
        optFlags = [['suboptions', None, None]]

        def postOptions(self):
            if False:
                for i in range(10):
                    print('nop')
            if self['suboptions']:
                self.subOptions = 'SubOptionUsage'
            raise usage.UsageError('usage-error-message')

        def __str__(self):
            if False:
                i = 10
                return i + 15
            return 'GeneralUsage'

    def test_run_good(self):
        if False:
            i = 10
            return i + 15
        '\n        Test successful invocation of worker command.\n        '
        self.patch(sys, 'argv', ['command', 'test', '--test-opt'])
        self.patch(runner.Options, 'subCommands', [['test', None, self.TestSubCommand, None]])
        subcommand_func = mock.Mock(return_value=42)
        self.patch(sys.modules[__name__], 'functionPlaceholder', subcommand_func)
        exception = self.assertRaises(SystemExit, runner.run)
        subcommand_func.assert_called_once_with({'test-opt': 1})
        self.assertEqual(exception.code, 42, 'unexpected exit code')

    def test_run_bad_noargs(self):
        if False:
            i = 10
            return i + 15
        '\n        Test handling of invalid command line arguments.\n        '
        self.patch(sys, 'argv', ['command'])
        self.patch(runner, 'Options', self.TestOptions)
        exception = self.assertRaises(SystemExit, runner.run)
        self.assertEqual(exception.code, 1, 'unexpected exit code')
        self.assertStdoutEqual('command:  usage-error-message\n\nGeneralUsage\n', 'unexpected error message on stdout')

    def test_run_bad_suboption(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test handling of invalid command line arguments in a suboption.\n        '
        self.patch(sys, 'argv', ['command', '--suboptions'])
        self.patch(runner, 'Options', self.TestOptions)
        exception = self.assertRaises(SystemExit, runner.run)
        self.assertEqual(exception.code, 1, 'unexpected exit code')
        self.assertStdoutEqual('command:  usage-error-message\n\nSubOptionUsage\n', 'unexpected error message on stdout')