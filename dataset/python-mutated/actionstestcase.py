__author__ = 'Daniel Black'
__copyright__ = 'Copyright (c) 2013 Daniel Black'
__license__ = 'GPL'
import time
import os
import tempfile
from ..server.ticket import FailTicket
from ..server.utils import Utils
from .dummyjail import DummyJail
from .utils import LogCaptureTestCase, with_alt_time, with_tmpdir, MyTime
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), 'files')

class ExecuteActions(LogCaptureTestCase):

    def setUp(self):
        if False:
            return 10
        'Call before every test case.'
        super(ExecuteActions, self).setUp()
        self.__jail = DummyJail()
        self.__actions = self.__jail.actions

    def tearDown(self):
        if False:
            return 10
        super(ExecuteActions, self).tearDown()

    def defaultAction(self, o={}):
        if False:
            i = 10
            return i + 15
        self.__actions.add('ip')
        act = self.__actions['ip']
        act.actionstart = 'echo ip start' + o.get('start', '')
        act.actionban = 'echo ip ban <ip>' + o.get('ban', '')
        act.actionunban = 'echo ip unban <ip>' + o.get('unban', '')
        act.actioncheck = 'echo ip check' + o.get('check', '')
        act.actionflush = 'echo ip flush' + o.get('flush', '')
        act.actionstop = 'echo ip stop' + o.get('stop', '')
        return act

    def testActionsAddDuplicateName(self):
        if False:
            for i in range(10):
                print('nop')
        self.__actions.add('test')
        self.assertRaises(ValueError, self.__actions.add, 'test')

    def testActionsManipulation(self):
        if False:
            i = 10
            return i + 15
        self.__actions.add('test')
        self.assertTrue(self.__actions['test'])
        self.assertIn('test', self.__actions)
        self.assertNotIn('nonexistant action', self.__actions)
        self.__actions.add('test1')
        del self.__actions['test']
        del self.__actions['test1']
        self.assertNotIn('test', self.__actions)
        self.assertEqual(len(self.__actions), 0)
        self.__actions.setBanTime(127)
        self.assertEqual(self.__actions.getBanTime(), 127)
        self.assertRaises(ValueError, self.__actions.removeBannedIP, '127.0.0.1')

    def testAddBannedIP(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.__actions.addBannedIP('192.0.2.1'), 1)
        self.assertLogged('Ban 192.0.2.1')
        self.pruneLog()
        self.assertEqual(self.__actions.addBannedIP(['192.0.2.1', '192.0.2.2', '192.0.2.3']), 2)
        self.assertLogged('192.0.2.1 already banned')
        self.assertNotLogged('Ban 192.0.2.1')
        self.assertLogged('Ban 192.0.2.2')
        self.assertLogged('Ban 192.0.2.3')

    def testActionsOutput(self):
        if False:
            i = 10
            return i + 15
        self.defaultAction()
        self.__actions.start()
        self.assertLogged('stdout: %r' % 'ip start', wait=True)
        self.__actions.stop()
        self.__actions.join()
        self.assertLogged('stdout: %r' % 'ip flush', 'stdout: %r' % 'ip stop')
        self.assertEqual(self.__actions.status(), [('Currently banned', 0), ('Total banned', 0), ('Banned IP list', [])])
        self.assertEqual(self.__actions.status('short'), [('Currently banned', 0), ('Total banned', 0)])

    def testAddActionPython(self):
        if False:
            for i in range(10):
                print('nop')
        self.__actions.add('Action', os.path.join(TEST_FILES_DIR, 'action.d/action.py'), {'opt1': 'value'})
        self.assertLogged('TestAction initialised')
        self.__actions.start()
        self.assertTrue(Utils.wait_for(lambda : self._is_logged('TestAction action start'), 3))
        self.__actions.stop()
        self.__actions.join()
        self.assertLogged('TestAction action stop')
        self.assertRaises(IOError, self.__actions.add, 'Action3', '/does/not/exist.py', {})
        self.__actions.add('Action4', os.path.join(TEST_FILES_DIR, 'action.d/action.py'), {'opt1': 'value', 'opt2': 'value2'})
        self.assertRaises(TypeError, self.__actions.add, 'Action5', os.path.join(TEST_FILES_DIR, 'action.d/action.py'), {'opt1': 'value', 'opt2': 'value2', 'opt3': 'value3'})
        self.assertRaises(TypeError, self.__actions.add, 'Action5', os.path.join(TEST_FILES_DIR, 'action.d/action.py'), {})

    def testAddPythonActionNOK(self):
        if False:
            print('Hello World!')
        self.assertRaises(RuntimeError, self.__actions.add, 'Action', os.path.join(TEST_FILES_DIR, 'action.d/action_noAction.py'), {})
        self.assertRaises(RuntimeError, self.__actions.add, 'Action', os.path.join(TEST_FILES_DIR, 'action.d/action_nomethod.py'), {})
        self.__actions.add('Action', os.path.join(TEST_FILES_DIR, 'action.d/action_errors.py'), {})
        self.__actions.start()
        self.assertTrue(Utils.wait_for(lambda : self._is_logged('Failed to start'), 3))
        self.__actions.stop()
        self.__actions.join()
        self.assertLogged('Failed to stop')

    def testBanActionsAInfo(self):
        if False:
            print('Hello World!')
        self.__actions.add('action1', os.path.join(TEST_FILES_DIR, 'action.d/action_modifyainfo.py'), {})
        self.__actions.add('action2', os.path.join(TEST_FILES_DIR, 'action.d/action_modifyainfo.py'), {})
        self.__jail.putFailTicket(FailTicket('1.2.3.4'))
        self.__actions._Actions__checkBan()
        self.assertNotLogged('Failed to execute ban')
        self.assertLogged('action1 ban deleted aInfo IP')
        self.assertLogged('action2 ban deleted aInfo IP')
        self.__actions._Actions__flushBan()
        self.assertNotLogged('Failed to execute unban')
        self.assertLogged('action1 unban deleted aInfo IP')
        self.assertLogged('action2 unban deleted aInfo IP')

    @with_alt_time
    def testUnbanOnBusyBanBombing(self):
        if False:
            return 10
        self.__actions.banPrecedence = 3
        self.__actions.unbanMaxCount = 5
        self.__actions.setBanTime(100)
        self.__actions.start()
        MyTime.setTime(0)
        i = 0
        while i < 20:
            ip = '192.0.2.%d' % i
            self.__jail.putFailTicket(FailTicket(ip, 0))
            i += 1
        self.assertLogged(' / 20,', wait=True)
        MyTime.setTime(200)
        while i < 50:
            ip = '192.0.2.%d' % i
            self.__jail.putFailTicket(FailTicket(ip, 200))
            i += 1
        self.assertLogged(' / 50,', wait=True)
        self.__actions.stop()
        self.__actions.join()
        self.assertLogged('Unbanned 30, 0 ticket(s)')
        self.assertNotLogged('Unbanned 50, 0 ticket(s)')

    def testActionsConsistencyCheck(self):
        if False:
            for i in range(10):
                print('nop')
        act = self.defaultAction({'check': ' <family>', 'flush': ' <family>'})
        act['actionflush?family=inet6'] = act.actionflush + '; exit 1'
        act.actionstart_on_demand = True
        act.actionban = '<actioncheck> ; ' + act.actionban
        act.actionunban = '<actioncheck> ; ' + act.actionunban
        self.__actions.start()
        self.assertNotLogged('stdout: %r' % 'ip start')
        self.assertEqual(self.__actions.addBannedIP('192.0.2.1'), 1)
        self.assertEqual(self.__actions.addBannedIP('2001:db8::1'), 1)
        self.assertLogged('Ban 192.0.2.1', 'Ban 2001:db8::1', 'stdout: %r' % 'ip start', 'stdout: %r' % 'ip ban 192.0.2.1', 'stdout: %r' % 'ip ban 2001:db8::1', all=True, wait=True)
        self.pruneLog('[test-phase 1a] simulate inconsistent irreparable env by unban')
        act['actioncheck?family=inet6'] = act.actioncheck + '; exit 1'
        self.__actions.removeBannedIP('2001:db8::1')
        self.assertLogged('Invariant check failed. Unban is impossible.', wait=True)
        self.pruneLog('[test-phase 1b] simulate inconsistent irreparable env by flush')
        self.__actions._Actions__flushBan()
        self.assertLogged('stdout: %r' % 'ip flush inet4', 'stdout: %r' % 'ip flush inet6', 'Failed to flush bans', 'No flush occurred, do consistency check', 'Invariant check failed. Trying to restore a sane environment', 'stdout: %r' % 'ip stop', 'Failed to flush bans', all=True, wait=True)
        self.pruneLog('[test-phase 2] consistent env')
        act['actioncheck?family=inet6'] = act.actioncheck
        self.assertEqual(self.__actions.addBannedIP('2001:db8::1'), 1)
        self.assertLogged('Ban 2001:db8::1', 'stdout: %r' % 'ip start', 'stdout: %r' % 'ip ban 2001:db8::1', all=True, wait=True)
        self.assertNotLogged('stdout: %r' % 'ip check inet4', all=True)
        self.pruneLog('[test-phase 3] failed flush in consistent env')
        self.__actions._Actions__flushBan()
        self.assertLogged('Failed to flush bans', 'No flush occurred, do consistency check', 'stdout: %r' % 'ip flush inet6', 'stdout: %r' % 'ip check inet6', all=True, wait=True)
        self.assertNotLogged('stdout: %r' % 'ip flush inet4', 'stdout: %r' % 'ip stop', 'stdout: %r' % 'ip start', 'Unable to restore environment', all=True)
        self.pruneLog('[test-phase end] flush successful')
        act['actionflush?family=inet6'] = act.actionflush
        self.__actions.stop()
        self.__actions.join()
        self.assertLogged('stdout: %r' % 'ip flush inet6', 'stdout: %r' % 'ip stop', 'action ip terminated', all=True, wait=True)
        self.assertNotLogged('ERROR', 'stdout: %r' % 'ip flush inet4', 'Unban tickets each individualy', all=True)

    def testActionsConsistencyCheckDiffFam(self):
        if False:
            for i in range(10):
                print('nop')
        act = self.defaultAction({'start': ' <family>', 'check': ' <family>', 'flush': ' <family>', 'stop': ' <family>'})
        act['actionflush?family=inet6'] = act.actionflush + '; exit 1'
        act.actionstart_on_demand = True
        act.actionrepair_on_unban = True
        act.actionban = '<actioncheck> ; ' + act.actionban
        act.actionunban = '<actioncheck> ; ' + act.actionunban
        self.__actions.start()
        self.assertNotLogged('stdout: %r' % 'ip start')
        self.assertEqual(self.__actions.addBannedIP('192.0.2.1'), 1)
        self.assertEqual(self.__actions.addBannedIP('2001:db8::1'), 1)
        self.assertLogged('Ban 192.0.2.1', 'Ban 2001:db8::1', 'stdout: %r' % 'ip start inet4', 'stdout: %r' % 'ip ban 192.0.2.1', 'stdout: %r' % 'ip start inet6', 'stdout: %r' % 'ip ban 2001:db8::1', all=True, wait=True)
        act['actioncheck?family=inet6'] = act.actioncheck + '; exit 1'
        self.pruneLog('[test-phase 1a] simulate inconsistent irreparable env by unban')
        self.__actions.removeBannedIP('2001:db8::1')
        self.assertLogged('Invariant check failed. Trying to restore a sane environment', 'stdout: %r' % 'ip stop inet6', all=True, wait=True)
        self.assertNotLogged('stdout: %r' % 'ip start inet6', 'stdout: %r' % 'ip stop inet4', 'stdout: %r' % 'ip start inet4', all=True)
        self.pruneLog('[test-phase 1b] simulate inconsistent irreparable env by ban')
        self.assertEqual(self.__actions.addBannedIP('2001:db8::1'), 1)
        self.assertLogged('Invariant check failed. Trying to restore a sane environment', 'stdout: %r' % 'ip stop inet6', 'stdout: %r' % 'ip start inet6', 'stdout: %r' % 'ip check inet6', 'Unable to restore environment', 'Failed to execute ban', all=True, wait=True)
        self.assertNotLogged('stdout: %r' % 'ip stop inet4', 'stdout: %r' % 'ip start inet4', all=True)
        act['actioncheck?family=inet6'] = act.actioncheck
        self.assertEqual(self.__actions.addBannedIP('2001:db8::2'), 1)
        act['actioncheck?family=inet6'] = act.actioncheck + '; exit 1'
        self.pruneLog('[test-phase 1c] simulate inconsistent irreparable env by flush')
        self.__actions._Actions__flushBan()
        self.assertLogged('stdout: %r' % 'ip flush inet4', 'stdout: %r' % 'ip flush inet6', 'Failed to flush bans', 'No flush occurred, do consistency check', 'Invariant check failed. Trying to restore a sane environment', 'stdout: %r' % 'ip stop inet6', 'Failed to flush bans in jail', all=True, wait=True)
        self.assertNotLogged('stdout: %r' % 'ip stop inet4', all=True)
        self.pruneLog('[test-phase 2] consistent env')
        act['actioncheck?family=inet6'] = act.actioncheck
        self.assertEqual(self.__actions.addBannedIP('2001:db8::1'), 1)
        self.assertLogged('Ban 2001:db8::1', 'stdout: %r' % 'ip start inet6', 'stdout: %r' % 'ip ban 2001:db8::1', all=True, wait=True)
        self.assertNotLogged('stdout: %r' % 'ip check inet4', 'stdout: %r' % 'ip start inet4', all=True)
        self.pruneLog('[test-phase 3] failed flush in consistent env')
        act['actioncheck?family=inet6'] = act.actioncheck
        self.__actions._Actions__flushBan()
        self.assertLogged('Failed to flush bans', 'No flush occurred, do consistency check', 'stdout: %r' % 'ip flush inet6', 'stdout: %r' % 'ip check inet6', all=True, wait=True)
        self.assertNotLogged('stdout: %r' % 'ip flush inet4', 'stdout: %r' % 'ip stop inet4', 'stdout: %r' % 'ip start inet4', 'stdout: %r' % 'ip stop inet6', 'stdout: %r' % 'ip start inet6', all=True)
        self.pruneLog('[test-phase end] flush successful')
        act['actionflush?family=inet6'] = act.actionflush
        self.__actions.stop()
        self.__actions.join()
        self.assertLogged('stdout: %r' % 'ip flush inet6', 'stdout: %r' % 'ip stop inet4', 'stdout: %r' % 'ip stop inet6', 'action ip terminated', all=True, wait=True)
        self.assertNotLogged('ERROR', 'stdout: %r' % 'ip flush inet4', 'Unban tickets each individualy', all=True)

    @with_alt_time
    @with_tmpdir
    def testActionsRebanBrokenAfterRepair(self, tmp):
        if False:
            print('Hello World!')
        act = self.defaultAction({'start': ' <family>; touch "<FN>"', 'check': ' <family>; test -f "<FN>"', 'flush': ' <family>; echo -n "" > "<FN>"', 'stop': ' <family>; rm -f "<FN>"', 'ban': ' <family>; echo "<ip> <family>" >> "<FN>"'})
        act['FN'] = tmp + '/<family>'
        act.actionstart_on_demand = True
        act.actionrepair = 'echo ip repair <family>; touch "<FN>"'
        act.actionreban = 'echo ip reban <ip> <family>; echo "<ip> <family> -- rebanned" >> "<FN>"'
        self.pruneLog('[test-phase 0] initial ban')
        self.assertEqual(self.__actions.addBannedIP(['192.0.2.1', '2001:db8::1']), 2)
        self.assertLogged('Ban 192.0.2.1', 'Ban 2001:db8::1', 'stdout: %r' % 'ip start inet4', 'stdout: %r' % 'ip ban 192.0.2.1 inet4', 'stdout: %r' % 'ip start inet6', 'stdout: %r' % 'ip ban 2001:db8::1 inet6', all=True)
        self.pruneLog('[test-phase 1] check ban')
        self.dumpFile(tmp + '/inet4')
        self.assertLogged('192.0.2.1 inet4')
        self.assertNotLogged('2001:db8::1 inet6')
        self.pruneLog()
        self.dumpFile(tmp + '/inet6')
        self.assertLogged('2001:db8::1 inet6')
        self.assertNotLogged('192.0.2.1 inet4')
        MyTime.setTime(MyTime.time() + 4)
        self.pruneLog('[test-phase 2] check already banned')
        self.assertEqual(self.__actions.addBannedIP(['192.0.2.1', '2001:db8::1', '2001:db8::2']), 1)
        self.assertLogged('192.0.2.1 already banned', '2001:db8::1 already banned', 'Ban 2001:db8::2', 'stdout: %r' % 'ip check inet4', 'stdout: %r' % 'ip check inet6', all=True)
        self.dumpFile(tmp + '/inet4')
        self.dumpFile(tmp + '/inet6')
        self.assertNotLogged('Reban 192.0.2.1', 'Reban 2001:db8::1', 'stdout: %r' % 'ip ban 192.0.2.1 inet4', 'stdout: %r' % 'ip reban 192.0.2.1 inet4', 'stdout: %r' % 'ip ban 2001:db8::1 inet6', 'stdout: %r' % 'ip reban 2001:db8::1 inet6', '192.0.2.1 inet4 -- repaired', '2001:db8::1 inet6 -- repaired', all=True)
        MyTime.setTime(MyTime.time() + 4)
        os.remove(tmp + '/inet4')
        os.remove(tmp + '/inet6')
        self.pruneLog('[test-phase 3a] check reban after sane env repaired')
        self.assertEqual(self.__actions.addBannedIP(['192.0.2.1', '2001:db8::1']), 2)
        self.assertLogged('Invariant check failed. Trying to restore a sane environment', 'stdout: %r' % 'ip repair inet4', 'stdout: %r' % 'ip repair inet6', "Reban 192.0.2.1, action 'ip'", "Reban 2001:db8::1, action 'ip'", 'stdout: %r' % 'ip reban 192.0.2.1 inet4', 'stdout: %r' % 'ip reban 2001:db8::1 inet6', all=True)
        self.pruneLog('[test-phase 3a] check reban by epoch mismatch (without repair)')
        self.assertEqual(self.__actions.addBannedIP('2001:db8::2'), 1)
        self.assertLogged("Reban 2001:db8::2, action 'ip'", 'stdout: %r' % 'ip reban 2001:db8::2 inet6', all=True)
        self.assertNotLogged('Invariant check failed. Trying to restore a sane environment', 'stdout: %r' % 'ip repair inet4', 'stdout: %r' % 'ip repair inet6', "Reban 192.0.2.1, action 'ip'", "Reban 2001:db8::1, action 'ip'", 'stdout: %r' % 'ip reban 192.0.2.1 inet4', 'stdout: %r' % 'ip reban 2001:db8::1 inet6', all=True)
        self.pruneLog('[test-phase 4] check reban')
        self.dumpFile(tmp + '/inet4')
        self.assertLogged('192.0.2.1 inet4 -- rebanned')
        self.assertNotLogged('2001:db8::1 inet6 -- rebanned')
        self.pruneLog()
        self.dumpFile(tmp + '/inet6')
        self.assertLogged('2001:db8::1 inet6 -- rebanned', '2001:db8::2 inet6 -- rebanned', all=True)
        self.assertNotLogged('192.0.2.1 inet4 -- rebanned')
        act.actionreban = ''
        act.actionban = 'exit 1'
        self.assertEqual(self.__actions._Actions__reBan(FailTicket('192.0.2.1', 0)), 0)
        self.assertLogged('Failed to execute reban', 'Error banning 192.0.2.1', all=True)