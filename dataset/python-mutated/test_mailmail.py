"""
Tests for L{twisted.mail.scripts.mailmail}, the implementation of the
command line program I{mailmail}.
"""
import os
import sys
from io import StringIO
from unittest import skipIf
from twisted.copyright import version
from twisted.internet.defer import Deferred
from twisted.internet.testing import MemoryReactor
from twisted.mail import smtp
from twisted.mail.scripts import mailmail
from twisted.mail.scripts.mailmail import parseOptions
from twisted.python.failure import Failure
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase

class OptionsTests(TestCase):
    """
    Tests for L{parseOptions} which parses command line arguments and reads
    message text from stdin to produce an L{Options} instance which can be
    used to send a message.
    """
    memoryReactor = MemoryReactor()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override some things in mailmail, so that we capture C{stdout},\n        and do not call L{reactor.stop}.\n        '
        self.out = StringIO()
        from twisted.logger import Logger, textFileLogObserver
        logObserver = textFileLogObserver(self.out)
        self.patch(mailmail, '_log', Logger(observer=logObserver))
        self.host = None
        self.options = None
        self.ident = None

        def sendmail(host, options, ident):
            if False:
                for i in range(10):
                    print('nop')
            self.host = host
            self.options = options
            self.ident = ident
            return smtp.sendmail(host, options.sender, options.to, options.body, reactor=self.memoryReactor)
        self.patch(mailmail, 'sendmail', sendmail)

    def test_unspecifiedRecipients(self):
        if False:
            print('Hello World!')
        '\n        If no recipients are given in the argument list and there is no\n        recipient header in the message text, L{parseOptions} raises\n        L{SystemExit} with a string describing the problem.\n        '
        self.patch(sys, 'stdin', StringIO('Subject: foo\n\nHello, goodbye.\n'))
        exc = self.assertRaises(SystemExit, parseOptions, [])
        self.assertEqual(exc.args, ('No recipients specified.',))

    def test_listQueueInformation(self):
        if False:
            return 10
        '\n        The I{-bp} option for listing queue information is unsupported and\n        if it is passed to L{parseOptions}, L{SystemExit} is raised.\n        '
        exc = self.assertRaises(SystemExit, parseOptions, ['-bp'])
        self.assertEqual(exc.args, ('Unsupported option.',))

    def test_stdioTransport(self):
        if False:
            return 10
        '\n        The I{-bs} option for using stdin and stdout as the SMTP transport\n        is unsupported and if it is passed to L{parseOptions}, L{SystemExit}\n        is raised.\n        '
        exc = self.assertRaises(SystemExit, parseOptions, ['-bs'])
        self.assertEqual(exc.args, ('Unsupported option.',))

    def test_ignoreFullStop(self):
        if False:
            while True:
                i = 10
        '\n        The I{-i} and I{-oi} options for ignoring C{"."} by itself on a line\n        are unsupported and if either is passed to L{parseOptions},\n        L{SystemExit} is raised.\n        '
        exc = self.assertRaises(SystemExit, parseOptions, ['-i'])
        self.assertEqual(exc.args, ('Unsupported option.',))
        exc = self.assertRaises(SystemExit, parseOptions, ['-oi'])
        self.assertEqual(exc.args, ('Unsupported option.',))

    def test_copyAliasedSender(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The I{-om} option for copying the sender if they appear in an alias\n        expansion is unsupported and if it is passed to L{parseOptions},\n        L{SystemExit} is raised.\n        '
        exc = self.assertRaises(SystemExit, parseOptions, ['-om'])
        self.assertEqual(exc.args, ('Unsupported option.',))

    def test_version(self):
        if False:
            return 10
        '\n        The I{--version} option displays the version and raises\n        L{SystemExit} with L{None} as the exit code.\n        '
        out = StringIO()
        self.patch(sys, 'stdout', out)
        systemExitCode = self.assertRaises(SystemExit, parseOptions, '--version')
        self.assertEqual(systemExitCode.code, None)
        data = out.getvalue()
        self.assertEqual(data, f'mailmail version: {version}\n')

    def test_backgroundDelivery(self):
        if False:
            return 10
        '\n        The I{-odb} flag specifies background delivery.\n        '
        stdin = StringIO('\n')
        self.patch(sys, 'stdin', stdin)
        o = parseOptions('-odb')
        self.assertTrue(o.background)

    def test_foregroundDelivery(self):
        if False:
            i = 10
            return i + 15
        '\n        The I{-odf} flags specifies foreground delivery.\n        '
        stdin = StringIO('\n')
        self.patch(sys, 'stdin', stdin)
        o = parseOptions('-odf')
        self.assertFalse(o.background)

    def test_recipientsFromHeaders(self):
        if False:
            i = 10
            return i + 15
        '\n        The I{-t} flags specifies that recipients should be obtained\n        from headers.\n        '
        stdin = StringIO('To: Curly <invaliduser2@example.com>\nCc: Larry <invaliduser1@example.com>\nBcc: Moe <invaliduser3@example.com>\n\nOh, a wise guy?\n')
        self.patch(sys, 'stdin', stdin)
        o = parseOptions('-t')
        self.assertEqual(len(o.to), 3)

    def test_setFrom(self):
        if False:
            print('Hello World!')
        '\n        When a message has no I{From:} header, a I{From:} value can be\n        specified with the I{-F} flag.\n        '
        stdin = StringIO('To: invaliduser2@example.com\nSubject: A wise guy?\n\n')
        self.patch(sys, 'stdin', stdin)
        o = parseOptions(['-F', 'Larry <invaliduser1@example.com>', '-t'])
        self.assertEqual(o.sender, 'Larry <invaliduser1@example.com>')

    def test_overrideFromFlagByFromHeader(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The I{-F} flag specifies the From: value.  However, I{-F} flag is\n        overriden by the value of From: in the e-mail header.\n        '
        stdin = StringIO('To: Curly <invaliduser4@example.com>\nFrom: Shemp <invaliduser4@example.com>\n')
        self.patch(sys, 'stdin', stdin)
        o = parseOptions(['-F', 'Groucho <invaliduser5@example.com>', '-t'])
        self.assertEqual(o.sender, 'invaliduser4@example.com')

    @skipIf(platformType == 'win32', 'mailmail.run() does not work on win32 due to lack of support for getuid()')
    def test_runErrorsToStderr(self):
        if False:
            while True:
                i = 10
        '\n        Call L{mailmail.run}, and specify I{-oep} to print errors\n        to stderr.  The sender, to, and printErrors options should be\n        set and there should be no failure.\n        '
        argv = ('test_mailmail.py', 'invaliduser2@example.com', '-oep')
        stdin = StringIO('\n')
        self.patch(sys, 'argv', argv)
        self.patch(sys, 'stdin', stdin)
        mailmail.run()
        self.assertEqual(self.options.sender, mailmail.getlogin())
        self.assertEqual(self.options.to, ['invaliduser2@example.com'])
        self.assertTrue(self.options.printErrors)
        self.assertIsNone(mailmail.failed)

    @skipIf(platformType == 'win32', 'mailmail.run() does not work on win32 due to lack of support for getuid()')
    def test_readInvalidConfig(self):
        if False:
            return 10
        '\n        Error messages for illegal UID value, illegal GID value, and illegal\n        identity entry will be sent to stderr.\n        '
        stdin = StringIO('\n')
        self.patch(sys, 'stdin', stdin)
        filename = self.mktemp()
        myUid = os.getuid()
        myGid = os.getgid()
        with open(filename, 'w') as f:
            f.write('[useraccess]\nallow=invaliduser2,invaliduser1\ndeny=invaliduser3,invaliduser4,{}\norder=allow,deny\n[groupaccess]\nallow=invalidgid1,invalidgid2\ndeny=invalidgid1,invalidgid2,{}\norder=deny,allow\n[identity]\nlocalhost=funny\n[addresses]\nsmarthost=localhost\ndefault_domain=example.com\n'.format(myUid, myGid))
        self.patch(mailmail, 'LOCAL_CFG', filename)
        argv = ('test_mailmail.py', 'invaliduser2@example.com', '-oep')
        self.patch(sys, 'argv', argv)
        mailmail.run()
        self.assertRegex(self.out.getvalue(), 'Illegal UID in \\[useraccess\\] section: invaliduser1')
        self.assertRegex(self.out.getvalue(), 'Illegal GID in \\[groupaccess\\] section: invalidgid1')
        self.assertRegex(self.out.getvalue(), 'Illegal entry in \\[identity\\] section: funny')

    def getConfigFromFile(self, config):
        if False:
            print('Hello World!')
        '\n        Read a mailmail configuration file.\n\n        The mailmail script checks the twisted.mail.scripts.mailmail.GLOBAL_CFG\n        variable and then the twisted.mail.scripts.mailmail.LOCAL_CFG\n        variable for the path to its  config file.\n\n        @param config: path to config file\n        @type config: L{str}\n\n        @return: A parsed config.\n        @rtype: L{twisted.mail.scripts.mailmail.Configuration}\n        '
        from twisted.mail.scripts.mailmail import loadConfig
        filename = self.mktemp()
        with open(filename, 'w') as f:
            f.write(config)
        return loadConfig(filename)

    def test_loadConfig(self):
        if False:
            return 10
        '\n        L{twisted.mail.scripts.mailmail.loadConfig}\n        parses the config file for mailmail.\n        '
        config = self.getConfigFromFile('\n[addresses]\nsmarthost=localhost')
        self.assertEqual(config.smarthost, 'localhost')
        config = self.getConfigFromFile('\n[addresses]\ndefault_domain=example.com')
        self.assertEqual(config.domain, 'example.com')
        config = self.getConfigFromFile('\n[addresses]\nsmarthost=localhost\ndefault_domain=example.com')
        self.assertEqual(config.smarthost, 'localhost')
        self.assertEqual(config.domain, 'example.com')
        config = self.getConfigFromFile('\n[identity]\nhost1=invalid\nhost2=username:password')
        self.assertNotIn('host1', config.identities)
        self.assertEqual(config.identities['host2'], ['username', 'password'])
        config = self.getConfigFromFile('\n[useraccess]\nallow=invalid1,35\norder=allow')
        self.assertEqual(config.allowUIDs, [35])
        config = self.getConfigFromFile('\n[useraccess]\ndeny=35,36\norder=deny')
        self.assertEqual(config.denyUIDs, [35, 36])
        config = self.getConfigFromFile('\n[useraccess]\nallow=35,36\ndeny=37,38\norder=deny')
        self.assertEqual(config.allowUIDs, [35, 36])
        self.assertEqual(config.denyUIDs, [37, 38])
        config = self.getConfigFromFile('\n[groupaccess]\nallow=gid1,41\norder=allow')
        self.assertEqual(config.allowGIDs, [41])
        config = self.getConfigFromFile('\n[groupaccess]\ndeny=41\norder=deny')
        self.assertEqual(config.denyGIDs, [41])
        config = self.getConfigFromFile('\n[groupaccess]\nallow=41,42\ndeny=43,44\norder=allow,deny')
        self.assertEqual(config.allowGIDs, [41, 42])
        self.assertEqual(config.denyGIDs, [43, 44])

    def test_senderror(self):
        if False:
            return 10
        '\n        L{twisted.mail.scripts.mailmail.senderror} sends mail back to the\n        sender if an error occurs while sending mail to the recipient.\n        '

        def sendmail(host, sender, recipient, body):
            if False:
                for i in range(10):
                    print('nop')
            self.assertRegex(sender, 'postmaster@')
            self.assertEqual(recipient, ['testsender'])
            self.assertRegex(body.getvalue(), 'ValueError')
            return Deferred()
        self.patch(smtp, 'sendmail', sendmail)
        opts = mailmail.Options()
        opts.sender = 'testsender'
        fail = Failure(ValueError())
        mailmail.senderror(fail, opts)