__author__ = 'Alexander Koeppe'
__copyright__ = 'Copyright (c) 2016 Cyril Jaquier, 2011-2013 Yaroslav Halchenko'
__license__ = 'GPL'
import unittest
from ..client.beautifier import Beautifier
from ..version import version
from ..server.ipdns import IPAddr
from ..exceptions import UnknownJailException, DuplicateJailException

class BeautifierTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        ' Call before every test case '
        super(BeautifierTest, self).setUp()
        self.b = Beautifier()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        ' Call after every test case '
        super(BeautifierTest, self).tearDown()

    def testGetInputCmd(self):
        if False:
            return 10
        cmd = ['test']
        self.b.setInputCmd(cmd)
        self.assertEqual(self.b.getInputCmd(), cmd)

    def testPing(self):
        if False:
            for i in range(10):
                print('nop')
        self.b.setInputCmd(['ping'])
        self.assertEqual(self.b.beautify('pong'), 'Server replied: pong')

    def testVersion(self):
        if False:
            return 10
        self.b.setInputCmd(['version'])
        self.assertEqual(self.b.beautify(version), version)

    def testAddJail(self):
        if False:
            for i in range(10):
                print('nop')
        self.b.setInputCmd(['add'])
        self.assertEqual(self.b.beautify('ssh'), 'Added jail ssh')

    def testStartJail(self):
        if False:
            print('Hello World!')
        self.b.setInputCmd(['start'])
        self.assertEqual(self.b.beautify(None), 'Jail started')

    def testStopJail(self):
        if False:
            for i in range(10):
                print('nop')
        self.b.setInputCmd(['stop', 'ssh'])
        self.assertEqual(self.b.beautify(None), 'Jail stopped')

    def testShutdown(self):
        if False:
            while True:
                i = 10
        self.b.setInputCmd(['stop'])
        self.assertEqual(self.b.beautify(None), 'Shutdown successful')

    def testStatus(self):
        if False:
            print('Hello World!')
        self.b.setInputCmd(['status'])
        response = (('Number of jails', 0), ('Jail list', ['ssh', 'exim4']))
        output = 'Status\n|- Number of jails:\t0\n`- Jail list:\tssh exim4'
        self.assertEqual(self.b.beautify(response), output)
        self.b.setInputCmd(['status', 'ssh'])
        response = (('Filter', [('Currently failed', 0), ('Total failed', 0), ('File list', '/var/log/auth.log')]), ('Actions', [('Currently banned', 3), ('Total banned', 3), ('Banned IP list', [IPAddr('192.168.0.1'), IPAddr('::ffff:10.2.2.1'), IPAddr('2001:db8::1')])]))
        output = 'Status for the jail: ssh\n'
        output += '|- Filter\n'
        output += '|  |- Currently failed:\t0\n'
        output += '|  |- Total failed:\t0\n'
        output += '|  `- File list:\t/var/log/auth.log\n'
        output += '`- Actions\n'
        output += '   |- Currently banned:\t3\n'
        output += '   |- Total banned:\t3\n'
        output += '   `- Banned IP list:\t192.168.0.1 10.2.2.1 2001:db8::1'
        self.assertEqual(self.b.beautify(response), output)

    def testFlushLogs(self):
        if False:
            while True:
                i = 10
        self.b.setInputCmd(['flushlogs'])
        self.assertEqual(self.b.beautify('rolled over'), 'logs: rolled over')

    def testSyslogSocket(self):
        if False:
            i = 10
            return i + 15
        self.b.setInputCmd(['get', 'syslogsocket'])
        output = 'Current syslog socket is:\n`- auto'
        self.assertEqual(self.b.beautify('auto'), output)

    def testLogTarget(self):
        if False:
            while True:
                i = 10
        self.b.setInputCmd(['get', 'logtarget'])
        output = 'Current logging target is:\n`- /var/log/fail2ban.log'
        self.assertEqual(self.b.beautify('/var/log/fail2ban.log'), output)

    def testLogLevel(self):
        if False:
            print('Hello World!')
        self.b.setInputCmd(['get', 'loglevel'])
        output = "Current logging level is 'INFO'"
        self.assertEqual(self.b.beautify('INFO'), output)

    def testDbFile(self):
        if False:
            print('Hello World!')
        self.b.setInputCmd(['get', 'dbfile'])
        response = '/var/lib/fail2ban/fail2ban.sqlite3'
        output = 'Current database file is:\n`- ' + response
        self.assertEqual(self.b.beautify(response), output)
        self.assertEqual(self.b.beautify(None), 'Database currently disabled')

    def testDbPurgeAge(self):
        if False:
            i = 10
            return i + 15
        self.b.setInputCmd(['get', 'dbpurgeage'])
        output = 'Current database purge age is:\n`- 86400seconds'
        self.assertEqual(self.b.beautify(86400), output)
        self.assertEqual(self.b.beautify(None), 'Database currently disabled')

    def testLogPath(self):
        if False:
            for i in range(10):
                print('nop')
        self.b.setInputCmd(['get', 'sshd', 'logpath'])
        response = []
        output = 'No file is currently monitored'
        self.assertEqual(self.b.beautify(response), output)
        response = ['/var/log/auth.log']
        output = 'Current monitored log file(s):\n`- /var/log/auth.log'
        self.assertEqual(self.b.beautify(response), output)
        self.b.setInputCmd(['set', 'sshd', 'addlogpath', '/var/log/messages'])
        response = ['/var/log/messages', '/var/log/auth.log']
        outputadd = 'Current monitored log file(s):\n'
        outputadd += '|- /var/log/messages\n`- /var/log/auth.log'
        self.assertEqual(self.b.beautify(response), outputadd)
        self.b.setInputCmd(['set', 'sshd', 'dellogpath', '/var/log/messages'])
        response = ['/var/log/auth.log']
        self.assertEqual(self.b.beautify(response), output)

    def testLogEncoding(self):
        if False:
            i = 10
            return i + 15
        self.b.setInputCmd(['get', 'sshd', 'logencoding'])
        output = 'Current log encoding is set to:\nUTF-8'
        self.assertEqual(self.b.beautify('UTF-8'), output)

    def testJournalMatch(self):
        if False:
            for i in range(10):
                print('nop')
        self.b.setInputCmd(['get', 'sshd', 'journalmatch'])
        self.assertEqual(self.b.beautify([]), 'No journal match filter set')
        self.b.setInputCmd(['set', 'sshd', 'addjournalmatch'])
        response = [['_SYSTEMD_UNIT', 'sshd.service']]
        output = 'Current match filter:\n'
        output += '_SYSTEMD_UNIT sshd.service'
        self.assertEqual(self.b.beautify(response), output)
        response.append(['_COMM', 'sshd'])
        output += ' + _COMM sshd'
        self.assertEqual(self.b.beautify(response), output)
        self.b.setInputCmd(['set', 'sshd', 'deljournalmatch'])
        response.remove(response[1])
        self.assertEqual(self.b.beautify(response), output.split(' + ')[0])

    def testDatePattern(self):
        if False:
            i = 10
            return i + 15
        self.b.setInputCmd(['get', 'sshd', 'datepattern'])
        output = 'Current date pattern set to: '
        response = (None, 'Default Detectors')
        self.assertEqual(self.b.beautify(None), output + 'Not set/required')
        self.assertEqual(self.b.beautify(response), output + 'Default Detectors')
        self.assertEqual(self.b.beautify(('test', 'test')), output + 'test (test)')

    def testIgnoreIP(self):
        if False:
            return 10
        self.b.setInputCmd(['get', 'sshd', 'ignoreip'])
        output = 'No IP address/network is ignored'
        self.assertEqual(self.b.beautify([]), output)
        self.b.setInputCmd(['set', 'sshd', 'addignoreip'])
        response = [IPAddr('127.0.0.0', 8), IPAddr('::1'), IPAddr('2001:db8::', 32), IPAddr('::ffff:10.0.2.1')]
        output = 'These IP addresses/networks are ignored:\n'
        output += '|- 127.0.0.0/8\n'
        output += '|- ::1\n'
        output += '|- 2001:db8::/32\n'
        output += '`- 10.0.2.1'
        self.assertEqual(self.b.beautify(response), output)

    def testFailRegex(self):
        if False:
            for i in range(10):
                print('nop')
        self.b.setInputCmd(['get', 'sshd', 'failregex'])
        output = 'No regular expression is defined'
        self.assertEqual(self.b.beautify([]), output)
        output = 'The following regular expression are defined:\n'
        output += '|- [0]: ^$\n`- [1]: .*'
        self.assertEqual(self.b.beautify(['^$', '.*']), output)

    def testActions(self):
        if False:
            while True:
                i = 10
        self.b.setInputCmd(['get', 'sshd', 'actions'])
        output = 'No actions for jail sshd'
        self.assertEqual(self.b.beautify([]), output)
        output = 'The jail sshd has the following actions:\n'
        output += 'iptables-multiport'
        self.assertEqual(self.b.beautify(['iptables-multiport']), output)

    def testActionProperties(self):
        if False:
            while True:
                i = 10
        self.b.setInputCmd(['get', 'sshd', 'actionproperties', 'iptables'])
        output = 'No properties for jail sshd action iptables'
        self.assertEqual(self.b.beautify([]), output)
        output = 'The jail sshd action iptables has the following properties:'
        output += '\nactionban, actionunban'
        response = ('actionban', 'actionunban')
        self.assertEqual(self.b.beautify(response), output)

    def testActionMethods(self):
        if False:
            i = 10
            return i + 15
        self.b.setInputCmd(['get', 'sshd', 'actionmethods', 'iptables'])
        output = 'No methods for jail sshd action iptables'
        self.assertEqual(self.b.beautify([]), output)
        output = 'The jail sshd action iptables has the following methods:\n'
        output += 'ban, unban'
        self.assertEqual(self.b.beautify(['ban', 'unban']), output)

    def testBeautifyError(self):
        if False:
            while True:
                i = 10
        response = UnknownJailException('sshd')
        output = "Sorry but the jail 'sshd' does not exist"
        self.assertEqual(self.b.beautifyError(response), output)
        response = DuplicateJailException('sshd')
        output = "The jail 'sshd' already exists"
        self.assertEqual(self.b.beautifyError(response), output)
        output = 'Sorry but the command is invalid'
        self.assertEqual(self.b.beautifyError(IndexError()), output)