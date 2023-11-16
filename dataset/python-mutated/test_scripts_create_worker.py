from __future__ import absolute_import
from __future__ import print_function
import os
from twisted.trial import unittest
from buildbot_worker.scripts import create_worker
from buildbot_worker.test.util import misc
try:
    from unittest import mock
except ImportError:
    import mock

def _regexp_path(name, *names):
    if False:
        while True:
            i = 10
    '\n    Join two or more path components and create a regexp that will match that\n    path.\n    '
    return os.path.join(name, *names).replace('\\', '\\\\')

class TestDefaultOptionsMixin:
    options = {'no-logrotate': False, 'relocatable': False, 'quiet': False, 'use-tls': False, 'delete-leftover-dirs': False, 'basedir': 'bdir', 'allow-shutdown': None, 'umask': None, 'log-size': 16, 'log-count': 8, 'keepalive': 4, 'maxdelay': 2, 'numcpus': None, 'protocol': 'pb', 'maxretries': None, 'connection-string': None, 'proxy-connection-string': None, 'host': 'masterhost', 'port': 1234, 'name': 'workername', 'passwd': 'orange'}

class TestMakeTAC(TestDefaultOptionsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.create_worker._make_tac()
    """

    def assert_tac_file_contents(self, tac_contents, expected_args, relocate=None):
        if False:
            while True:
                i = 10
        '\n        Check that generated TAC file is a valid Python script and it does what\n        is typical for TAC file logic. Mainly create instance of Worker with\n        expected arguments.\n        '
        import twisted.application.service
        import twisted.python.logfile
        import buildbot_worker.bot
        application_mock = mock.Mock()
        application_class_mock = mock.Mock(return_value=application_mock)
        self.patch(twisted.application.service, 'Application', application_class_mock)
        logfile_mock = mock.Mock()
        self.patch(twisted.python.logfile.LogFile, 'fromFullPath', logfile_mock)
        worker_mock = mock.Mock()
        worker_class_mock = mock.Mock(return_value=worker_mock)
        self.patch(buildbot_worker.bot, 'Worker', worker_class_mock)
        globals_dict = {}
        if relocate:
            globals_dict['__file__'] = os.path.join(relocate, 'buildbot.tac')
        exec(tac_contents, globals_dict, globals_dict)
        application_class_mock.assert_called_once_with('buildbot-worker')
        worker_class_mock.assert_called_once_with(expected_args['host'], expected_args['port'], expected_args['name'], expected_args['passwd'], expected_args['basedir'], expected_args['keepalive'], umask=expected_args['umask'], numcpus=expected_args['numcpus'], protocol=expected_args['protocol'], maxdelay=expected_args['maxdelay'], allow_shutdown=expected_args['allow-shutdown'], maxRetries=expected_args['maxretries'], useTls=expected_args['use-tls'], delete_leftover_dirs=expected_args['delete-leftover-dirs'], connection_string=expected_args['connection-string'], proxy_connection_string=expected_args['proxy-connection-string'])
        self.assertEqual(worker_mock.method_calls, [mock.call.setServiceParent(application_mock)])
        self.assertTrue('application' in globals_dict, '.tac file doesn\'t define "application" variable')
        self.assertTrue(globals_dict['application'] is application_mock, 'defined "application" variable in .tac file is not Application instance')

    def test_default_tac_contents(self):
        if False:
            i = 10
            return i + 15
        '\n        test that with default options generated TAC file is valid.\n        '
        tac_contents = create_worker._make_tac(self.options.copy())
        self.assert_tac_file_contents(tac_contents, self.options)

    def test_backslash_in_basedir(self):
        if False:
            while True:
                i = 10
        "\n        test that using backslash (typical for Windows platform) in basedir\n        won't break generated TAC file.\n        "
        options = self.options.copy()
        options['basedir'] = 'C:\\buildbot-worker dir\\\\'
        tac_contents = create_worker._make_tac(options.copy())
        self.assert_tac_file_contents(tac_contents, options)

    def test_quotes_in_basedir(self):
        if False:
            i = 10
            return i + 15
        "\n        test that using quotes in basedir won't break generated TAC file.\n        "
        options = self.options.copy()
        options['basedir'] = 'Buildbot\'s \\"dir'
        tac_contents = create_worker._make_tac(options.copy())
        self.assert_tac_file_contents(tac_contents, options)

    def test_double_quotes_in_basedir(self):
        if False:
            return 10
        "\n        test that using double quotes at begin and end of basedir won't break\n        generated TAC file.\n        "
        options = self.options.copy()
        options['basedir'] = '\\"\\"Buildbot\'\''
        tac_contents = create_worker._make_tac(options.copy())
        self.assert_tac_file_contents(tac_contents, options)

    def test_special_characters_in_options(self):
        if False:
            return 10
        "\n        test that using special characters in options strings won't break\n        generated TAC file.\n        "
        test_string = '"" & | ^ # @ \\& \\| \\^ \\# \\@ \\n \x07 " \\" \' \\\' \'\''
        options = self.options.copy()
        options['basedir'] = test_string
        options['host'] = test_string
        options['passwd'] = test_string
        options['name'] = test_string
        tac_contents = create_worker._make_tac(options.copy())
        self.assert_tac_file_contents(tac_contents, options)

    def test_flags_with_non_default_values(self):
        if False:
            while True:
                i = 10
        '\n        test that flags with non-default values will be correctly written to\n        generated TAC file.\n        '
        options = self.options.copy()
        options['quiet'] = True
        options['use-tls'] = True
        options['delete-leftover-dirs'] = True
        tac_contents = create_worker._make_tac(options.copy())
        self.assert_tac_file_contents(tac_contents, options)

    def test_log_rotate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test that when --no-logrotate options is not used, correct tac file\n        is generated.\n        '
        options = self.options.copy()
        options['no-logrotate'] = False
        tac_contents = create_worker._make_tac(options.copy())
        self.assertIn('from twisted.python.logfile import LogFile', tac_contents)
        self.assert_tac_file_contents(tac_contents, options)

    def test_no_log_rotate(self):
        if False:
            i = 10
            return i + 15
        '\n        test that when --no-logrotate options is used, correct tac file\n        is generated.\n        '
        options = self.options.copy()
        options['no-logrotate'] = True
        tac_contents = create_worker._make_tac(options.copy())
        self.assertNotIn('from twisted.python.logfile import LogFile', tac_contents)
        self.assert_tac_file_contents(tac_contents, options)

    def test_relocatable_true(self):
        if False:
            return 10
        '\n        test that when --relocatable option is True, worker is created from\n        generated TAC file with correct basedir argument before and after\n        relocation.\n        '
        options = self.options.copy()
        options['relocatable'] = True
        options['basedir'] = os.path.join(os.getcwd(), 'worker1')
        tac_contents = create_worker._make_tac(options.copy())
        self.assert_tac_file_contents(tac_contents, options, relocate=options['basedir'])
        _relocate = os.path.join(os.getcwd(), 'worker2')
        options['basedir'] = _relocate
        self.assert_tac_file_contents(tac_contents, options, relocate=_relocate)

    def test_relocatable_false(self):
        if False:
            while True:
                i = 10
        '\n        test that when --relocatable option is False, worker is created from\n        generated TAC file with the same basedir argument before and after\n        relocation.\n        '
        options = self.options.copy()
        options['relocatable'] = False
        options['basedir'] = os.path.join(os.getcwd(), 'worker1')
        tac_contents = create_worker._make_tac(options.copy())
        self.assert_tac_file_contents(tac_contents, options, relocate=options['basedir'])
        _relocate = os.path.join(os.getcwd(), 'worker2')
        self.assert_tac_file_contents(tac_contents, options, relocate=_relocate)

    def test_options_with_non_default_values(self):
        if False:
            print('Hello World!')
        '\n        test that options with non-default values will be correctly written to\n        generated TAC file and used as argument of Worker.\n        '
        options = self.options.copy()
        options['allow-shutdown'] = 'signal'
        options['umask'] = '18'
        options['log-size'] = 160
        options['log-count'] = '80'
        options['keepalive'] = 40
        options['maxdelay'] = 20
        options['numcpus'] = '10'
        options['protocol'] = 'null'
        options['maxretries'] = '1'
        options['proxy-connection-string'] = 'TCP:proxy.com:8080'
        tac_contents = create_worker._make_tac(options.copy())
        self.assertIn('rotateLength = 160', tac_contents)
        self.assertIn('maxRotatedFiles = 80', tac_contents)
        self.assertIn('keepalive = 40', tac_contents)
        self.assertIn('maxdelay = 20', tac_contents)
        self.assertIn('umask = 18', tac_contents)
        self.assertIn('numcpus = 10', tac_contents)
        self.assertIn('maxretries = 1', tac_contents)
        options['umask'] = 18
        options['numcpus'] = 10
        options['maxretries'] = 1
        self.assert_tac_file_contents(tac_contents, options)

    def test_umask_octal_value(self):
        if False:
            return 10
        '\n        test that option umask with octal value will be correctly written to\n        generated TAC file and used as argument of Worker.\n        '
        options = self.options.copy()
        options['umask'] = '0o22'
        tac_contents = create_worker._make_tac(options.copy())
        self.assertIn('umask = 0o22', tac_contents)
        options['umask'] = 18
        self.assert_tac_file_contents(tac_contents, options)

    def test_connection_string(self):
        if False:
            while True:
                i = 10
        '\n        test that when --connection-string options is used, correct tac file\n        is generated.\n        '
        options = self.options.copy()
        options['connection-string'] = 'TLS:buildbot-master.com:9989'
        tac_contents = create_worker._make_tac(options.copy())
        options['host'] = None
        options['port'] = None
        self.assert_tac_file_contents(tac_contents, options)

class TestMakeBaseDir(misc.StdoutAssertionsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.create_worker._makeBaseDir()
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setUpStdoutAssertions()
        self.mkdir = mock.Mock()
        self.patch(os, 'mkdir', self.mkdir)

    def testBasedirExists(self):
        if False:
            return 10
        '\n        test calling _makeBaseDir() on existing base directory\n        '
        self.patch(os.path, 'exists', mock.Mock(return_value=True))
        create_worker._makeBaseDir('dummy', False)
        self.assertStdoutEqual('updating existing installation\n')
        self.assertFalse(self.mkdir.called, 'unexpected call to os.mkdir()')

    def testBasedirExistsQuiet(self):
        if False:
            while True:
                i = 10
        '\n        test calling _makeBaseDir() on existing base directory with\n        quiet flag enabled\n        '
        self.patch(os.path, 'exists', mock.Mock(return_value=True))
        create_worker._makeBaseDir('dummy', True)
        self.assertWasQuiet()
        self.assertFalse(self.mkdir.called, 'unexpected call to os.mkdir()')

    def testBasedirCreated(self):
        if False:
            while True:
                i = 10
        '\n        test creating new base directory with _makeBaseDir()\n        '
        self.patch(os.path, 'exists', mock.Mock(return_value=False))
        create_worker._makeBaseDir('dummy', False)
        self.mkdir.assert_called_once_with('dummy')
        self.assertStdoutEqual('mkdir dummy\n')

    def testBasedirCreatedQuiet(self):
        if False:
            while True:
                i = 10
        '\n        test creating new base directory with _makeBaseDir()\n        and quiet flag enabled\n        '
        self.patch(os.path, 'exists', mock.Mock(return_value=False))
        create_worker._makeBaseDir('dummy', True)
        self.mkdir.assert_called_once_with('dummy')
        self.assertWasQuiet()

    def testMkdirError(self):
        if False:
            i = 10
            return i + 15
        '\n        test that _makeBaseDir() handles error creating directory correctly\n        '
        self.patch(os.path, 'exists', mock.Mock(return_value=False))
        self.patch(os, 'mkdir', mock.Mock(side_effect=OSError(0, 'dummy-error')))
        with self.assertRaisesRegex(create_worker.CreateWorkerError, 'error creating directory dummy: dummy-error'):
            create_worker._makeBaseDir('dummy', False)

class TestMakeBuildbotTac(misc.StdoutAssertionsMixin, misc.FileIOMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.create_worker._makeBuildbotTac()
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setUpStdoutAssertions()
        self.chmod = mock.Mock()
        self.patch(os, 'chmod', self.chmod)
        self.tac_file_path = _regexp_path('bdir', 'buildbot.tac')

    def testTacOpenError(self):
        if False:
            return 10
        '\n        test that _makeBuildbotTac() handles open() errors on buildbot.tac\n        '
        self.patch(os.path, 'exists', mock.Mock(return_value=True))
        self.setUpOpenError()
        expected_message = 'error reading {0}: dummy-msg'.format(self.tac_file_path)
        with self.assertRaisesRegex(create_worker.CreateWorkerError, expected_message):
            create_worker._makeBuildbotTac('bdir', 'contents', False)

    def testTacReadError(self):
        if False:
            i = 10
            return i + 15
        '\n        test that _makeBuildbotTac() handles read() errors on buildbot.tac\n        '
        self.patch(os.path, 'exists', mock.Mock(return_value=True))
        self.setUpReadError()
        expected_message = 'error reading {0}: dummy-msg'.format(self.tac_file_path)
        with self.assertRaisesRegex(create_worker.CreateWorkerError, expected_message):
            create_worker._makeBuildbotTac('bdir', 'contents', False)

    def testTacWriteError(self):
        if False:
            while True:
                i = 10
        '\n        test that _makeBuildbotTac() handles write() errors on buildbot.tac\n        '
        self.patch(os.path, 'exists', mock.Mock(return_value=False))
        self.setUpWriteError(0)
        expected_message = 'could not write {0}: dummy-msg'.format(self.tac_file_path)
        with self.assertRaisesRegex(create_worker.CreateWorkerError, expected_message):
            create_worker._makeBuildbotTac('bdir', 'contents', False)

    def checkTacFileCorrect(self, quiet):
        if False:
            for i in range(10):
                print('nop')
        "\n        Utility function to test calling _makeBuildbotTac() on base directory\n        with existing buildbot.tac file, which does not need to be changed.\n\n        @param quiet: the value of 'quiet' argument for _makeBuildbotTac()\n        "
        self.patch(os.path, 'exists', mock.Mock(return_value=True))
        self.setUpOpen('test-tac-contents')
        create_worker._makeBuildbotTac('bdir', 'test-tac-contents', quiet)
        self.assertFalse(self.fileobj.write.called, 'unexpected write() call')
        if quiet:
            self.assertWasQuiet()
        else:
            self.assertStdoutEqual('buildbot.tac already exists and is correct\n')

    def testTacFileCorrect(self):
        if False:
            return 10
        '\n        call _makeBuildbotTac() on base directory which contains a buildbot.tac\n        file, which does not need to be changed\n        '
        self.checkTacFileCorrect(False)

    def testTacFileCorrectQuiet(self):
        if False:
            return 10
        '\n        call _makeBuildbotTac() on base directory which contains a buildbot.tac\n        file, which does not need to be changed. Check that quite flag works\n        '
        self.checkTacFileCorrect(True)

    def checkDiffTacFile(self, quiet):
        if False:
            while True:
                i = 10
        "\n        Utility function to test calling _makeBuildbotTac() on base directory\n        with a buildbot.tac file, with does needs to be changed.\n\n        @param quiet: the value of 'quiet' argument for _makeBuildbotTac()\n        "
        self.patch(os.path, 'exists', mock.Mock(return_value=True))
        self.setUpOpen('old-tac-contents')
        create_worker._makeBuildbotTac('bdir', 'new-tac-contents', quiet)
        tac_file_path = os.path.join('bdir', 'buildbot.tac')
        self.open.assert_has_calls([mock.call(tac_file_path, 'rt'), mock.call(tac_file_path + '.new', 'wt')])
        self.fileobj.write.assert_called_once_with('new-tac-contents')
        self.chmod.assert_called_once_with(tac_file_path + '.new', 384)
        if quiet:
            self.assertWasQuiet()
        else:
            self.assertStdoutEqual('not touching existing buildbot.tac\ncreating buildbot.tac.new instead\n')

    def testDiffTacFile(self):
        if False:
            print('Hello World!')
        '\n        call _makeBuildbotTac() on base directory which contains a buildbot.tac\n        file, with does needs to be changed.\n        '
        self.checkDiffTacFile(False)

    def testDiffTacFileQuiet(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        call _makeBuildbotTac() on base directory which contains a buildbot.tac\n        file, with does needs to be changed. Check that quite flag works\n        '
        self.checkDiffTacFile(True)

    def testNoTacFile(self):
        if False:
            print('Hello World!')
        '\n        call _makeBuildbotTac() on base directory with no buildbot.tac file\n        '
        self.patch(os.path, 'exists', mock.Mock(return_value=False))
        self.setUpOpen()
        create_worker._makeBuildbotTac('bdir', 'test-tac-contents', False)
        tac_file_path = os.path.join('bdir', 'buildbot.tac')
        self.open.assert_called_once_with(tac_file_path, 'wt')
        self.fileobj.write.assert_called_once_with('test-tac-contents')
        self.chmod.assert_called_once_with(tac_file_path, 384)

class TestMakeInfoFiles(misc.StdoutAssertionsMixin, misc.FileIOMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.create_worker._makeInfoFiles()
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setUpStdoutAssertions()

    def checkMkdirError(self, quiet):
        if False:
            print('Hello World!')
        "\n        Utility function to test _makeInfoFiles() when os.mkdir() fails.\n\n        Patch os.mkdir() to raise an exception, and check that _makeInfoFiles()\n        handles mkdir errors correctly.\n\n        @param quiet: the value of 'quiet' argument for _makeInfoFiles()\n        "
        self.patch(os.path, 'exists', mock.Mock(return_value=False))
        self.patch(os, 'mkdir', mock.Mock(side_effect=OSError(0, 'err-msg')))
        with self.assertRaisesRegex(create_worker.CreateWorkerError, 'error creating directory {}: err-msg'.format(_regexp_path('bdir', 'info'))):
            create_worker._makeInfoFiles('bdir', quiet)
        if quiet:
            self.assertWasQuiet()
        else:
            self.assertStdoutEqual('mkdir {0}\n'.format(os.path.join('bdir', 'info')))

    def testMkdirError(self):
        if False:
            print('Hello World!')
        '\n        test _makeInfoFiles() when os.mkdir() fails\n        '
        self.checkMkdirError(False)

    def testMkdirErrorQuiet(self):
        if False:
            while True:
                i = 10
        '\n        test _makeInfoFiles() when os.mkdir() fails and quiet flag is enabled\n        '
        self.checkMkdirError(True)

    def checkIOError(self, error_type, quiet):
        if False:
            while True:
                i = 10
        "\n        Utility function to test _makeInfoFiles() when open() or write() fails.\n\n        Patch file IO functions to raise an exception, and check that\n        _makeInfoFiles() handles file IO errors correctly.\n\n        @param error_type: type of error to emulate,\n                           'open' - patch open() to fail\n                           'write' - patch write() to fail\n        @param quiet: the value of 'quiet' argument for _makeInfoFiles()\n        "
        self.patch(os.path, 'exists', lambda path: path.endswith('info'))
        if error_type == 'open':
            self.setUpOpenError(strerror='info-err-msg')
        elif error_type == 'write':
            self.setUpWriteError(strerror='info-err-msg')
        else:
            self.fail("unexpected error_type '{0}'".format(error_type))
        with self.assertRaisesRegex(create_worker.CreateWorkerError, 'could not write {0}: info-err-msg'.format(_regexp_path('bdir', 'info', 'admin'))):
            create_worker._makeInfoFiles('bdir', quiet)
        if quiet:
            self.assertWasQuiet()
        else:
            self.assertStdoutEqual('Creating {}, you need to edit it appropriately.\n'.format(os.path.join('info', 'admin')))

    def testOpenError(self):
        if False:
            i = 10
            return i + 15
        '\n        test _makeInfoFiles() when open() fails\n        '
        self.checkIOError('open', False)

    def testOpenErrorQuiet(self):
        if False:
            i = 10
            return i + 15
        '\n        test _makeInfoFiles() when open() fails and quiet flag is enabled\n        '
        self.checkIOError('open', True)

    def testWriteError(self):
        if False:
            print('Hello World!')
        '\n        test _makeInfoFiles() when write() fails\n        '
        self.checkIOError('write', False)

    def testWriteErrorQuiet(self):
        if False:
            while True:
                i = 10
        '\n        test _makeInfoFiles() when write() fails and quiet flag is enabled\n        '
        self.checkIOError('write', True)

    def checkCreatedSuccessfully(self, quiet):
        if False:
            while True:
                i = 10
        "\n        Utility function to test _makeInfoFiles() when called on\n        base directory that does not have 'info' sub-directory.\n\n        @param quiet: the value of 'quiet' argument for _makeInfoFiles()\n        "
        self.patch(os.path, 'exists', mock.Mock(return_value=False))
        mkdir_mock = mock.Mock()
        self.patch(os, 'mkdir', mkdir_mock)
        self.setUpOpen()
        create_worker._makeInfoFiles('bdir', quiet)
        info_path = os.path.join('bdir', 'info')
        mkdir_mock.assert_called_once_with(info_path)
        self.open.assert_has_calls([mock.call(os.path.join(info_path, 'admin'), 'wt'), mock.call(os.path.join(info_path, 'host'), 'wt')])
        self.fileobj.write.assert_has_calls([mock.call('Your Name Here <admin@youraddress.invalid>\n'), mock.call('Please put a description of this build host here\n')])
        if quiet:
            self.assertWasQuiet()
        else:
            self.assertStdoutEqual('mkdir {}\nCreating {}, you need to edit it appropriately.\nCreating {}, you need to edit it appropriately.\nNot creating {} - add it if you wish\nPlease edit the files in {} appropriately.\n'.format(info_path, os.path.join('info', 'admin'), os.path.join('info', 'host'), os.path.join('info', 'access_uri'), info_path))

    def testCreatedSuccessfully(self):
        if False:
            i = 10
            return i + 15
        "\n        test calling _makeInfoFiles() on basedir without 'info' directory\n        "
        self.checkCreatedSuccessfully(False)

    def testCreatedSuccessfullyQuiet(self):
        if False:
            print('Hello World!')
        "\n        test calling _makeInfoFiles() on basedir without 'info' directory\n        and quiet flag is enabled\n        "
        self.checkCreatedSuccessfully(True)

    def testInfoDirExists(self):
        if False:
            return 10
        "\n        test calling _makeInfoFiles() on basedir with fully populated\n        'info' directory\n        "
        self.patch(os.path, 'exists', mock.Mock(return_value=True))
        create_worker._makeInfoFiles('bdir', False)
        self.assertWasQuiet()

class TestCreateWorker(misc.StdoutAssertionsMixin, TestDefaultOptionsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.create_worker.createWorker()
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setUpStdoutAssertions()

    def setUpMakeFunctions(self, exception=None):
        if False:
            i = 10
            return i + 15
        '\n        patch create_worker._make*() functions with a mocks\n\n        @param exception: if not None, the mocks will raise this exception.\n        '
        self._makeBaseDir = mock.Mock(side_effect=exception)
        self.patch(create_worker, '_makeBaseDir', self._makeBaseDir)
        self._makeBuildbotTac = mock.Mock(side_effect=exception)
        self.patch(create_worker, '_makeBuildbotTac', self._makeBuildbotTac)
        self._makeInfoFiles = mock.Mock(side_effect=exception)
        self.patch(create_worker, '_makeInfoFiles', self._makeInfoFiles)

    def assertMakeFunctionsCalls(self, basedir, tac_contents, quiet):
        if False:
            while True:
                i = 10
        '\n        assert that create_worker._make*() were called with specified arguments\n        '
        self._makeBaseDir.assert_called_once_with(basedir, quiet)
        self._makeBuildbotTac.assert_called_once_with(basedir, tac_contents, quiet)
        self._makeInfoFiles.assert_called_once_with(basedir, quiet)

    def testCreateError(self):
        if False:
            i = 10
            return i + 15
        '\n        test that errors while creating worker directory are handled\n        correctly by createWorker()\n        '
        self.setUpMakeFunctions(create_worker.CreateWorkerError('err-msg'))
        self.assertEqual(create_worker.createWorker(self.options), 1, 'unexpected exit code')
        self.assertStdoutEqual('err-msg\nfailed to configure worker in bdir\n')

    def testMinArgs(self):
        if False:
            return 10
        '\n        test calling createWorker() with only required arguments\n        '
        self.setUpMakeFunctions()
        self.assertEqual(create_worker.createWorker(self.options), 0, 'unexpected exit code')
        expected_tac_contents = create_worker._make_tac(self.options.copy())
        self.assertMakeFunctionsCalls(self.options['basedir'], expected_tac_contents, self.options['quiet'])
        self.assertStdoutEqual('worker configured in bdir\n')

    def testQuiet(self):
        if False:
            i = 10
            return i + 15
        '\n        test calling createWorker() with --quiet flag\n        '
        options = self.options.copy()
        options['quiet'] = True
        self.setUpMakeFunctions()
        self.assertEqual(create_worker.createWorker(options), 0, 'unexpected exit code')
        expected_tac_contents = create_worker._make_tac(self.options)
        self.assertMakeFunctionsCalls(options['basedir'], expected_tac_contents, options['quiet'])
        self.assertWasQuiet()