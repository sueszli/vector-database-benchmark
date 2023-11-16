__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import os
import tempfile
import time
import unittest
from ..server.action import CommandAction, CallingMap, substituteRecursiveTags
from ..server.actions import OrderedDict, Actions
from ..server.utils import Utils
from .dummyjail import DummyJail
from .utils import pid_exists, with_tmpdir, LogCaptureTestCase

class CommandActionTest(LogCaptureTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Call before every test case.'
        LogCaptureTestCase.setUp(self)
        self.__action = CommandAction(None, 'Test')
        self.__action_started = False
        orgstart = self.__action.start

        def _action_start():
            if False:
                return 10
            self.__action_started = True
            return orgstart()
        self.__action.start = _action_start

    def tearDown(self):
        if False:
            print('Hello World!')
        'Call after every test case.'
        if self.__action_started:
            self.__action.stop()
        LogCaptureTestCase.tearDown(self)

    def testSubstituteRecursiveTags(self):
        if False:
            i = 10
            return i + 15
        aInfo = {'HOST': '192.0.2.0', 'ABC': '123 <HOST>', 'xyz': '890 <ABC>'}
        self.assertRaises(ValueError, lambda : substituteRecursiveTags({'A': '<A>'}))
        self.assertRaises(ValueError, lambda : substituteRecursiveTags({'A': '<B>', 'B': '<A>'}))
        self.assertRaises(ValueError, lambda : substituteRecursiveTags({'A': '<B>', 'B': '<C>', 'C': '<A>'}))
        self.assertRaises(ValueError, lambda : substituteRecursiveTags({'A': 'to=<B> fromip=<IP>', 'C': '<B>', 'B': '<C>', 'D': ''}))
        self.assertRaises(ValueError, lambda : substituteRecursiveTags({'failregex': 'to=<honeypot> fromip=<IP>', 'sweet': '<honeypot>', 'honeypot': '<sweet>', 'ignoreregex': ''}))
        self.assertEqual(substituteRecursiveTags(OrderedDict((('X', 'x=x<T>'), ('T', '1'), ('Z', '<X> <T> <Y>'), ('Y', 'y=y<T>')))), {'X': 'x=x1', 'T': '1', 'Y': 'y=y1', 'Z': 'x=x1 1 y=y1'})
        self.assertEqual(substituteRecursiveTags(OrderedDict((('X', 'x=x<T> <Z> <<R1>> <<R2>>'), ('R1', 'Z'), ('R2', 'Y'), ('T', '1'), ('Z', '<T> <Y>'), ('Y', 'y=y<T>')))), {'X': 'x=x1 1 y=y1 1 y=y1 y=y1', 'R1': 'Z', 'R2': 'Y', 'T': '1', 'Z': '1 y=y1', 'Y': 'y=y1'})
        self.assertEqual(substituteRecursiveTags(OrderedDict((('actionstart', 'ipset create <ipmset> hash:ip timeout <bantime> family <ipsetfamily>\n<iptables> -I <chain> <actiontype>'), ('ipmset', 'f2b-<name>'), ('name', 'any'), ('bantime', '600'), ('ipsetfamily', 'inet'), ('iptables', 'iptables <lockingopt>'), ('lockingopt', '-w'), ('chain', 'INPUT'), ('actiontype', '<multiport>'), ('multiport', '-p <protocol> -m multiport --dports <port> -m set --match-set <ipmset> src -j <blocktype>'), ('protocol', 'tcp'), ('port', 'ssh'), ('blocktype', 'REJECT')))), OrderedDict((('actionstart', 'ipset create f2b-any hash:ip timeout 600 family inet\niptables -w -I INPUT -p tcp -m multiport --dports ssh -m set --match-set f2b-any src -j REJECT'), ('ipmset', 'f2b-any'), ('name', 'any'), ('bantime', '600'), ('ipsetfamily', 'inet'), ('iptables', 'iptables -w'), ('lockingopt', '-w'), ('chain', 'INPUT'), ('actiontype', '-p tcp -m multiport --dports ssh -m set --match-set f2b-any src -j REJECT'), ('multiport', '-p tcp -m multiport --dports ssh -m set --match-set f2b-any src -j REJECT'), ('protocol', 'tcp'), ('port', 'ssh'), ('blocktype', 'REJECT'))))
        self.assertRaises(ValueError, lambda : substituteRecursiveTags(OrderedDict((('A', '<<B><C>>'), ('B', 'D'), ('C', 'E'), ('DE', 'cycle <A>')))))
        self.assertRaises(ValueError, lambda : substituteRecursiveTags(OrderedDict((('DE', 'cycle <A>'), ('A', '<<B><C>>'), ('B', 'D'), ('C', 'E')))))
        self.assertEqual(substituteRecursiveTags({'A': '<C>'}), {'A': '<C>'})
        self.assertEqual(substituteRecursiveTags({'A': '<C> <D> <X>', 'X': 'fun'}), {'A': '<C> <D> fun', 'X': 'fun'})
        self.assertEqual(substituteRecursiveTags({'A': '<C> <B>', 'B': 'cool'}), {'A': '<C> cool', 'B': 'cool'})
        self.assertEqual(substituteRecursiveTags({'A': '<matches> <B>', 'B': 'cool'}), {'A': '<matches> cool', 'B': 'cool'})
        self.assertEqual(substituteRecursiveTags({'failregex': 'to=<honeypot> fromip=<IP> evilperson=<honeypot>', 'honeypot': 'pokie', 'ignoreregex': ''}), {'failregex': 'to=pokie fromip=<IP> evilperson=pokie', 'honeypot': 'pokie', 'ignoreregex': ''})
        self.assertEqual(substituteRecursiveTags(aInfo), {'HOST': '192.0.2.0', 'ABC': '123 192.0.2.0', 'xyz': '890 123 192.0.2.0'})
        self.assertEqual(substituteRecursiveTags({'A': '<<PREF>HOST>', 'PREF': 'IPV4'}), {'A': '<IPV4HOST>', 'PREF': 'IPV4'})
        self.assertEqual(substituteRecursiveTags({'A': '<<PREF>HOST>', 'PREF': 'IPV4', 'IPV4HOST': '1.2.3.4'}), {'A': '1.2.3.4', 'PREF': 'IPV4', 'IPV4HOST': '1.2.3.4'})
        self.assertEqual(substituteRecursiveTags({'A': 'A <IP<PREF>HOST> B IP<PREF> C', 'PREF': 'V4', 'IPV4HOST': '1.2.3.4'}), {'A': 'A 1.2.3.4 B IPV4 C', 'PREF': 'V4', 'IPV4HOST': '1.2.3.4'})

    def testSubstRec_DontTouchUnusedCallable(self):
        if False:
            while True:
                i = 10
        cm = CallingMap({'A': 0, 'B': lambda self: '<A><A>', 'C': '', 'D': ''})
        substituteRecursiveTags(cm)
        cm['C'] = lambda self, i=0: 5 // int(self['A'])
        self.assertRaises(ZeroDivisionError, lambda : cm['C'])
        substituteRecursiveTags(cm)
        cm['D'] = 'test=<C>'
        self.assertRaises(ZeroDivisionError, lambda : substituteRecursiveTags(cm))
        self.assertEqual(self.__action.replaceTag('test=<A>', cm), 'test=0')
        self.assertEqual(self.__action.replaceTag('test=<A>--<B>--<A>', cm), 'test=0--<A><A>--0')
        self.assertRaises(ZeroDivisionError, lambda : self.__action.replaceTag('test=<C>', cm))
        self.assertEqual(self.__action.replaceTag('<D>', cm), 'test=<C>')

    def testReplaceTag(self):
        if False:
            print('Hello World!')
        aInfo = {'HOST': '192.0.2.0', 'ABC': '123', 'xyz': '890'}
        self.assertEqual(self.__action.replaceTag('Text<br>text', aInfo), 'Text\ntext')
        self.assertEqual(self.__action.replaceTag('Text <HOST> text', aInfo), 'Text 192.0.2.0 text')
        self.assertEqual(self.__action.replaceTag('Text <xyz> text <ABC> ABC', aInfo), 'Text 890 text 123 ABC')
        self.assertEqual(self.__action.replaceTag('<matches>', {'matches': 'some >char< should \\< be[ escap}ed&\n'}), 'some \\>char\\< should \\\\\\< be\\[ escap\\}ed\\&\\n')
        self.assertEqual(self.__action.replaceTag('<ipmatches>', {'ipmatches': 'some >char< should \\< be[ escap}ed&\n'}), 'some \\>char\\< should \\\\\\< be\\[ escap\\}ed\\&\\n')
        self.assertEqual(self.__action.replaceTag('<ipjailmatches>', {'ipjailmatches': 'some >char< should \\< be[ escap}ed&\r\n'}), 'some \\>char\\< should \\\\\\< be\\[ escap\\}ed\\&\\r\\n')
        aInfo['ABC'] = '<xyz>'
        self.assertEqual(self.__action.replaceTag('Text <xyz> text <ABC> ABC', aInfo), 'Text 890 text 890 ABC')
        self.assertEqual(self.__action.replaceTag('09 <matches> 11', CallingMap(matches=lambda self: str(10))), '09 10 11')

    def testReplaceNoTag(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.__action.replaceTag('abc', CallingMap(matches=lambda self: int('a'))), 'abc')

    def testReplaceTagSelfRecursion(self):
        if False:
            i = 10
            return i + 15
        setattr(self.__action, 'a', '<a')
        setattr(self.__action, 'b', 'c>')
        setattr(self.__action, 'b?family=inet6', 'b>')
        setattr(self.__action, 'ac', '<a><b>')
        setattr(self.__action, 'ab', '<ac>')
        setattr(self.__action, 'x?family=inet6', '')
        self.assertRaisesRegex(ValueError, 'properties contain self referencing definitions', lambda : self.__action.replaceTag('<a><b>', self.__action._properties, conditional='family=inet4'))
        delattr(self.__action, 'ac')
        self.assertRaisesRegex(ValueError, 'possible self referencing definitions in query', lambda : self.__action.replaceTag('<x' * 30 + '>' * 30, self.__action._properties, conditional='family=inet6'))

    def testReplaceTagConditionalCached(self):
        if False:
            while True:
                i = 10
        setattr(self.__action, 'abc', '123')
        setattr(self.__action, 'abc?family=inet4', '345')
        setattr(self.__action, 'abc?family=inet6', '567')
        setattr(self.__action, 'xyz', '890-<abc>')
        setattr(self.__action, 'banaction', 'Text <xyz> text <abc>')
        cache = self.__action._substCache
        for i in range(2):
            self.assertEqual(self.__action.replaceTag("<banaction> '<abc>'", self.__action._properties, conditional='', cache=cache), "Text 890-123 text 123 '123'")
            self.assertEqual(self.__action.replaceTag("<banaction> '<abc>'", self.__action._properties, conditional='family=inet4', cache=cache), "Text 890-345 text 345 '345'")
            self.assertEqual(self.__action.replaceTag("<banaction> '<abc>'", self.__action._properties, conditional='family=inet6', cache=cache), "Text 890-567 text 567 '567'")
        self.assertTrue(len(cache) >= 3)
        setattr(self.__action, 'xyz', '000-<abc>')
        self.assertEqual(len(cache), 0)
        for i in range(2):
            self.assertEqual(self.__action.replaceTag("<banaction> '<abc>'", self.__action._properties, conditional='', cache=cache), "Text 000-123 text 123 '123'")
            self.assertEqual(self.__action.replaceTag("<banaction> '<abc>'", self.__action._properties, conditional='family=inet4', cache=cache), "Text 000-345 text 345 '345'")
            self.assertEqual(self.__action.replaceTag("<banaction> '<abc>'", self.__action._properties, conditional='family=inet6', cache=cache), "Text 000-567 text 567 '567'")
        self.assertTrue(len(cache) >= 3)

    @with_tmpdir
    def testExecuteActionBan(self, tmp):
        if False:
            return 10
        tmp += '/fail2ban.test'
        self.__action.actionstart = "touch '%s'" % tmp
        self.__action.actionrepair = self.__action.actionstart
        self.assertEqual(self.__action.actionstart, "touch '%s'" % tmp)
        self.__action.actionstop = "rm -f '%s'" % tmp
        self.assertEqual(self.__action.actionstop, "rm -f '%s'" % tmp)
        self.__action.actionban = '<actioncheck> && echo -n'
        self.assertEqual(self.__action.actionban, '<actioncheck> && echo -n')
        self.__action.actioncheck = "[ -e '%s' ]" % tmp
        self.assertEqual(self.__action.actioncheck, "[ -e '%s' ]" % tmp)
        self.__action.actionunban = 'true'
        self.assertEqual(self.__action.actionunban, 'true')
        self.pruneLog()
        self.assertNotLogged('returned')
        self.__action.ban({'ip': None})
        self.assertLogged('Invariant check failed')
        self.assertLogged('returned successfully')
        self.__action.stop()
        self.assertLogged(self.__action.actionstop)

    def testExecuteActionEmptyUnban(self):
        if False:
            while True:
                i = 10
        self.__action.actionban = ''
        self.__action.actionunban = ''
        self.__action.actionflush = "echo -n 'flush'"
        self.__action.actionstop = "echo -n 'stop'"
        self.__action.start()
        self.__action.ban({})
        self.pruneLog()
        self.__action.unban({})
        self.assertLogged('Nothing to do', wait=True)
        self.__action.ban({})
        self.pruneLog('[phase 2]')
        self.__action.flush()
        self.__action.unban({})
        self.__action.stop()
        self.assertLogged('stop', wait=True)
        self.assertNotLogged('Nothing to do')

    @with_tmpdir
    def testExecuteActionStartCtags(self, tmp):
        if False:
            for i in range(10):
                print('nop')
        tmp += '/fail2ban.test'
        self.__action.HOST = '192.0.2.0'
        self.__action.actionstart = "touch '%s.<HOST>'" % tmp
        self.__action.actionstop = "rm -f '%s.<HOST>'" % tmp
        self.__action.actioncheck = "[ -e '%s.192.0.2.0' ]" % tmp
        self.__action.start()
        self.__action.consistencyCheck()

    @with_tmpdir
    def testExecuteActionCheckRestoreEnvironment(self, tmp):
        if False:
            i = 10
            return i + 15
        tmp += '/fail2ban.test'
        self.__action.actionstart = ''
        self.__action.actionstop = "rm -f '%s'" % tmp
        self.__action.actionban = "rm '%s'" % tmp
        self.__action.actioncheck = "[ -e '%s' ]" % tmp
        self.assertRaises(RuntimeError, self.__action.ban, {'ip': None})
        self.assertLogged('Invariant check failed', 'Unable to restore environment', all=True)
        self.pruneLog('[phase 2]')
        self.__action.actionstart = "touch '%s'" % tmp
        self.__action.actionstop = "rm '%s'" % tmp
        self.__action.actionban = '<actioncheck> && printf "%%%%b\n" <ip> >> \'%s\'' % tmp
        self.__action.actioncheck = "[ -e '%s' ]" % tmp
        self.__action.ban({'ip': None})
        self.assertLogged('Invariant check failed')
        self.assertNotLogged('Unable to restore environment')

    @with_tmpdir
    def testExecuteActionCheckOnBanFailure(self, tmp):
        if False:
            for i in range(10):
                print('nop')
        tmp += '/fail2ban.test'
        self.__action.actionstart = "touch '%s'; echo 'started ...'" % tmp
        self.__action.actionstop = "rm -f '%s'" % tmp
        self.__action.actionban = "[ -e '%s' ] && echo 'banned '<ip>" % tmp
        self.__action.actioncheck = "[ -e '%s' ] && echo 'check ok' || { echo 'check failed'; exit 1; }" % tmp
        self.__action.actionrepair = "echo 'repair ...'; touch '%s'" % tmp
        self.__action.actionstart_on_demand = False
        self.__action.start()
        for i in (1, 2, 3):
            self.pruneLog('[phase %s]' % i)
            self.__action.ban({'ip': '192.0.2.1'})
            self.assertLogged('stdout: %r' % 'banned 192.0.2.1', all=True)
            self.assertNotLogged('Invariant check failed. Trying', 'stdout: %r' % 'check failed', 'stdout: %r' % ('repair ...' if self.__action.actionrepair else 'started ...'), 'stdout: %r' % 'check ok', all=True)
            os.remove(tmp)
            self.pruneLog()
            self.__action.ban({'ip': '192.0.2.2'})
            self.assertLogged('Invariant check failed. Trying', 'stdout: %r' % 'check failed', 'stdout: %r' % ('repair ...' if self.__action.actionrepair else 'started ...'), 'stdout: %r' % 'check ok', 'stdout: %r' % 'banned 192.0.2.2', all=True)
            if self.__action.actionrepair:
                self.__action.actionrepair = ''
            elif not self.__action.actionstart_on_demand:
                self.__action.actionstart_on_demand = True

    @with_tmpdir
    def testExecuteActionCheckRepairEnvironment(self, tmp):
        if False:
            for i in range(10):
                print('nop')
        tmp += '/fail2ban.test'
        self.__action.actionstart = ''
        self.__action.actionstop = ''
        self.__action.actionban = "rm '%s'" % tmp
        self.__action.actioncheck = "[ -e '%s' ]" % tmp
        self.__action.actionrepair = "echo 'repair ...'; touch '%s'" % tmp
        self.__action.ban({'ip': None})
        self.assertLogged('Invariant check failed. Trying', "echo 'repair ...'", all=True)
        self.pruneLog()
        self.__action.actionrepair = "echo 'repair ...'"
        self.assertRaises(RuntimeError, self.__action.ban, {'ip': None})
        self.assertLogged('Invariant check failed. Trying', "echo 'repair ...'", 'Unable to restore environment', all=True)

    def testExecuteActionChangeCtags(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(AttributeError, getattr, self.__action, 'ROST')
        self.__action.ROST = '192.0.2.0'
        self.assertEqual(self.__action.ROST, '192.0.2.0')

    def testExecuteActionUnbanAinfo(self):
        if False:
            i = 10
            return i + 15
        aInfo = CallingMap({'ABC': '123', 'ip': '192.0.2.1', 'F-*': lambda self: {'fid': 111, 'fport': 222, 'user': 'tester'}})
        self.__action.actionban = "echo '<ABC>, failure <F-ID> of <F-USER> -<F-TEST>- from <ip>:<F-PORT>'"
        self.__action.actionunban = "echo '<ABC>, user <F-USER> unbanned'"
        self.__action.ban(aInfo)
        self.__action.unban(aInfo)
        self.assertLogged(" -- stdout: '123, failure 111 of tester -- from 192.0.2.1:222'", " -- stdout: '123, user tester unbanned'", all=True)

    def testExecuteActionStartEmpty(self):
        if False:
            i = 10
            return i + 15
        self.__action.actionstart = ''
        self.__action.start()
        self.assertTrue(self.__action.executeCmd(''))
        self.assertLogged('Nothing to do')
        self.pruneLog()
        self.assertTrue(self.__action._processCmd(''))
        self.assertLogged('Nothing to do')
        self.pruneLog()

    def testExecuteWithVars(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.__action.executeCmd('printf %b "foreign input:\\n -- $f2bV_A --\\n -- $f2bV_B --\\n -- $(echo -n $f2bV_C) --"', varsDict={'f2bV_A': "I'm a hacker; && $(echo $f2bV_B)", 'f2bV_B': 'I"m very bad hacker', 'f2bV_C': '`Very | very\n$(bad & worst hacker)`'}))
        self.assertLogged('foreign input:', " -- I'm a hacker; && $(echo $f2bV_B) --", ' -- I"m very bad hacker --', ' -- `Very | very $(bad & worst hacker)` --', all=True)

    def testExecuteReplaceEscapeWithVars(self):
        if False:
            for i in range(10):
                print('nop')
        self.__action.actionban = 'echo "** ban <ip>, reason: <reason> ...\\n<matches>"'
        self.__action.actionunban = 'echo "** unban <ip>"'
        self.__action.actionstop = 'echo "** stop monitoring"'
        matches = ['<actionunban>', '" Hooray! #', "`I'm cool script kiddy", '`I`m very cool > /here-is-the-path/to/bin/.x-attempt.sh', '<actionstop>']
        aInfo = {'ip': '192.0.2.1', 'reason': 'hacking attempt ( he thought he knows how f2b internally works ;)', 'matches': '\n'.join(matches)}
        self.pruneLog()
        self.__action.ban(aInfo)
        self.assertLogged('** ban %s' % aInfo['ip'], aInfo['reason'], *matches, all=True)
        self.assertNotLogged('** unban %s' % aInfo['ip'], '** stop monitoring', all=True)
        self.pruneLog()
        self.__action.unban(aInfo)
        self.__action.stop()
        self.assertLogged('** unban %s' % aInfo['ip'], '** stop monitoring', all=True)

    def testExecuteIncorrectCmd(self):
        if False:
            return 10
        CommandAction.executeCmd('/bin/ls >/dev/null\nbogusXXX now 2>/dev/null')
        self.assertLogged('HINT on 127: "Command not found"')

    def testExecuteTimeout(self):
        if False:
            while True:
                i = 10
        stime = time.time()
        timeout = 1 if not unittest.F2B.fast else 0.01
        self.assertFalse(CommandAction.executeCmd('sleep 30', timeout=timeout))
        self.assertTrue(time.time() >= stime + timeout and time.time() <= stime + timeout + 1)
        self.assertLogged('sleep 30', ' -- timed out after', all=True)
        self.assertLogged(' -- killed with SIGTERM', ' -- killed with SIGKILL')

    def testExecuteTimeoutWithNastyChildren(self):
        if False:
            print('Hello World!')
        tmpFilename = tempfile.mktemp('.sh', 'fail2ban_')
        with open(tmpFilename, 'w') as f:
            f.write('#!/bin/bash\n\t\ttrap : HUP EXIT TERM\n\n\t\techo "$$" > %s.pid\n\t\techo "my pid $$ . sleeping lo-o-o-ong"\n\t\tsleep 30\n\t\t' % tmpFilename)
        stime = 0

        def getnasty_tout():
            if False:
                return 10
            return getnastypid() is not None or time.time() - stime > 5

        def getnastypid():
            if False:
                for i in range(10):
                    print('nop')
            cpid = None
            if os.path.isfile(tmpFilename + '.pid'):
                with open(tmpFilename + '.pid') as f:
                    try:
                        cpid = int(f.read())
                    except ValueError:
                        pass
            return cpid
        stime = time.time()
        self.assertFalse(CommandAction.executeCmd('bash %s' % tmpFilename, timeout=getnasty_tout))
        cpid = getnastypid()
        self.assertTrue(Utils.wait_for(lambda : not pid_exists(cpid), 3))
        self.assertLogged('my pid ', 'Resource temporarily unavailable')
        self.assertLogged('timed out')
        self.assertLogged('killed with SIGTERM', 'killed with SIGKILL')
        os.unlink(tmpFilename + '.pid')
        stime = time.time()
        self.assertFalse(CommandAction.executeCmd('out=`bash %s`; echo ALRIGHT' % tmpFilename, timeout=getnasty_tout))
        cpid = getnastypid()
        self.assertTrue(Utils.wait_for(lambda : not pid_exists(cpid), 3))
        self.assertLogged('my pid ', 'Resource temporarily unavailable')
        self.assertLogged(' -- timed out')
        self.assertLogged(' -- killed with SIGTERM', ' -- killed with SIGKILL')
        os.unlink(tmpFilename)
        os.unlink(tmpFilename + '.pid')

    def testCaptureStdOutErr(self):
        if False:
            while True:
                i = 10
        CommandAction.executeCmd('echo "How now brown cow"')
        self.assertLogged("stdout: 'How now brown cow'\n")
        CommandAction.executeCmd('echo "The rain in Spain stays mainly in the plain" 1>&2')
        self.assertLogged("stderr: 'The rain in Spain stays mainly in the plain'\n")

    def testCallingMap(self):
        if False:
            for i in range(10):
                print('nop')
        mymap = CallingMap(callme=lambda self: str(10), error=lambda self: int('a'), dontcallme='string', number=17)
        self.assertEqual('%(callme)s okay %(dontcallme)s %(number)i' % mymap, '10 okay string 17')
        self.assertRaises(ValueError, lambda x: '%(error)i' % x, mymap)

    def testCallingMapModify(self):
        if False:
            return 10
        m = CallingMap({'a': lambda self: 2 + 3, 'b': lambda self: self['a'] + 6, 'c': 'test'})
        m.reset()
        m['a'] = 4
        del m['c']
        self.assertEqual(len(m), 2)
        self.assertNotIn('c', m)
        self.assertEqual((m['a'], m['b']), (4, 10))
        m.reset()
        s = repr(m)
        self.assertEqual(len(m), 3)
        self.assertIn('c', m)
        self.assertEqual((m['a'], m['b'], m['c']), (5, 11, 'test'))
        m['d'] = 'dddd'
        m2 = m.copy()
        m2['c'] = lambda self: self['a'] + 7
        m2['a'] = 1
        del m2['b']
        del m2['d']
        self.assertTrue('b' in m)
        self.assertTrue('d' in m)
        self.assertFalse('b' in m2)
        self.assertFalse('d' in m2)
        self.assertEqual((m['a'], m['b'], m['c'], m['d']), (5, 11, 'test', 'dddd'))
        self.assertEqual((m2['a'], m2['c']), (1, 8))

    def testCallingMapRep(self):
        if False:
            print('Hello World!')
        m = CallingMap({'a': lambda self: 2 + 3, 'b': lambda self: self['a'] + 6, 'c': ''})
        s = repr(m)
        self.assertNotIn("'a': ", s)
        self.assertNotIn("'b': ", s)
        self.assertIn("'c': ''", s)
        s = m._asrepr(True)
        self.assertIn("'a': 5", s)
        self.assertIn("'b': 11", s)
        self.assertIn("'c': ''", s)
        m['c'] = lambda self: self['xxx'] + 7
        s = m._asrepr(True)
        self.assertIn("'a': 5", s)
        self.assertIn("'b': 11", s)
        self.assertIn("'c': ", s)
        self.assertNotIn("'c': ''", s)

    def testActionsIdleMode(self):
        if False:
            while True:
                i = 10
        a = Actions(DummyJail())
        a.sleeptime = 0.0001
        a.idle = True
        a.start()
        self.assertLogged('Actions: enter idle mode', wait=10)
        a.idle = False
        self.assertLogged('Actions: leave idle mode', wait=10)
        a.active = False
        a.join()