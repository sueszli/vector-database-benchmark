import contextlib
from r2.tests import RedditTestCase
from mock import patch, MagicMock
from r2.models import Message
from r2.models.builder import UserMessageBuilder, MessageBuilder
from pylons import tmpl_context as c

class UserMessageBuilderTest(RedditTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(UserMessageBuilderTest, self).setUp()
        self.user = MagicMock(name='user')
        self.message = MagicMock(spec=Message)

    def test_view_message_on_receiver_side_and_spam(self):
        if False:
            while True:
                i = 10
        user = MagicMock(name='user')
        userMessageBuilder = UserMessageBuilder(user)
        self.user._id = 1
        self.message.author_id = 2
        self.message._spam = True
        with self.mock_preparation():
            self.assertFalse(userMessageBuilder._viewable_message(self.message))

    def test_view_message_on_receiver_side_and_del(self):
        if False:
            print('Hello World!')
        user = MagicMock(name='user')
        userMessageBuilder = UserMessageBuilder(user)
        self.user._id = 1
        self.message.author_id = 2
        self.message.to_id = self.user._id
        self.message._spam = False
        self.message.del_on_recipient = True
        with self.mock_preparation():
            self.assertFalse(userMessageBuilder._viewable_message(self.message))

    def test_view_message_on_receiver_side(self):
        if False:
            while True:
                i = 10
        user = MagicMock(name='user')
        userMessageBuilder = UserMessageBuilder(user)
        self.user._id = 1
        self.message.author_id = 2
        self.message.to_id = self.user._id
        self.message._spam = False
        self.message.del_on_recipient = False
        with self.mock_preparation():
            self.assertTrue(userMessageBuilder._viewable_message(self.message))

    def test_view_message_on_sender_side_and_del(self):
        if False:
            for i in range(10):
                print('nop')
        user = MagicMock(name='user')
        userMessageBuilder = UserMessageBuilder(user)
        self.message.to_id = 1
        self.user._id = 2
        self.message.author_id = self.user._id
        self.message._spam = False
        self.message.del_on_recipient = True
        with self.mock_preparation():
            self.assertTrue(userMessageBuilder._viewable_message(self.message))

    def test_view_message_on_admin_and_del(self):
        if False:
            print('Hello World!')
        user = MagicMock(name='user')
        userMessageBuilder = UserMessageBuilder(user)
        self.user._id = 1
        self.message.author_id = 2
        self.message.to_id = self.user._id
        self.message._spam = False
        self.message.del_on_recipient = True
        with self.mock_preparation(True):
            self.assertTrue(userMessageBuilder._viewable_message(self.message))

    def mock_preparation(self, is_admin=False):
        if False:
            while True:
                i = 10
        ' Context manager for mocking function calls. '
        return contextlib.nested(patch.object(c, 'user', self.user, create=True), patch.object(c, 'user_is_admin', is_admin, create=True), patch.object(MessageBuilder, '_viewable_message', return_value=True))