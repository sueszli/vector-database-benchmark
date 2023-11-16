import contextlib
from r2.tests import RedditControllerTestCase
from mock import patch, MagicMock
from r2.lib.validator import VByName, VUser, VModhash
from r2.models import Link, Message, Account
from pylons import app_globals as g

class DelMsgTest(RedditControllerTestCase):
    CONTROLLER = 'api'

    def setUp(self):
        if False:
            print('Hello World!')
        super(DelMsgTest, self).setUp()
        self.id = 1

    def test_del_msg_success(self):
        if False:
            for i in range(10):
                print('nop')
        'Del_msg succeeds: Returns 200 and sets del_on_recipient.'
        message = MagicMock(spec=Message)
        message.name = 'msg_1'
        message.to_id = self.id
        message.del_on_recipient = False
        with self.mock_del_msg(message):
            res = self.do_del_msg(message.name)
            self.assertEqual(res.status, 200)
            self.assertTrue(message.del_on_recipient)

    def test_del_msg_failure_with_link(self):
        if False:
            while True:
                i = 10
        'Del_msg fails: Returns 200 and does not set del_on_recipient.'
        link = MagicMock(spec=Link)
        link.del_on_recipient = False
        link.name = 'msg_2'
        with self.mock_del_msg(link):
            res = self.do_del_msg(link.name)
            self.assertEqual(res.status, 200)
            self.assertFalse(link.del_on_recipient)

    def test_del_msg_failure_with_null_msg(self):
        if False:
            print('Hello World!')
        'Del_msg fails: Returns 200 and does not set del_on_recipient.'
        message = MagicMock(spec=Message)
        message.name = 'msg_3'
        message.to_id = self.id
        message.del_on_recipient = False
        with self.mock_del_msg(message, False):
            res = self.do_del_msg(message.name)
            self.assertEqual(res.status, 200)
            self.assertFalse(message.del_on_recipient)

    def test_del_msg_failure_with_sender(self):
        if False:
            i = 10
            return i + 15
        'Del_msg fails: Returns 200 and does not set del_on_recipient.'
        message = MagicMock(spec=Message)
        message.name = 'msg_3'
        message.to_id = self.id + 1
        message.del_on_recipient = False
        with self.mock_del_msg(message):
            res = self.do_del_msg(message.name)
            self.assertEqual(res.status, 200)
            self.assertFalse(message.del_on_recipient)

    def mock_del_msg(self, thing, ret=True):
        if False:
            while True:
                i = 10
        'Context manager for mocking del_msg.'
        return contextlib.nested(patch.object(VByName, 'run', return_value=thing if ret else None), patch.object(VModhash, 'run', side_effect=None), patch.object(VUser, 'run', side_effect=None), patch.object(thing, '_commit', side_effect=None), patch.object(Account, '_id', self.id, create=True), patch.object(g.events, 'message_event', side_effect=None))

    def do_del_msg(self, name, **kw):
        if False:
            print('Hello World!')
        return self.do_post('del_msg', {'id': name}, **kw)