import asyncio
import datetime
import pytest
from telegram import Bot, CallbackQuery, Chat, ChatInviteLink, ChatJoinRequest, ChosenInlineResult, Message, PreCheckoutQuery, ShippingQuery, Update, User
from telegram._utils.datetime import UTC
from telegram.ext import CallbackContext, ChatJoinRequestHandler, JobQueue
from tests.auxil.slots import mro_slots
message = Message(1, None, Chat(1, ''), from_user=User(1, '', False), text='Text')
params = [{'message': message}, {'edited_message': message}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat', message=message)}, {'channel_post': message}, {'edited_channel_post': message}, {'chosen_inline_result': ChosenInlineResult('id', User(1, '', False), '')}, {'shipping_query': ShippingQuery('id', User(1, '', False), '', None)}, {'pre_checkout_query': PreCheckoutQuery('id', User(1, '', False), '', 0, '')}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat')}]
ids = ('message', 'edited_message', 'callback_query', 'channel_post', 'edited_channel_post', 'chosen_inline_result', 'shipping_query', 'pre_checkout_query', 'callback_query_without_message')

@pytest.fixture(scope='class', params=params, ids=ids)
def false_update(request):
    if False:
        for i in range(10):
            print('nop')
    return Update(update_id=2, **request.param)

@pytest.fixture(scope='class')
def time():
    if False:
        while True:
            i = 10
    return datetime.datetime.now(tz=UTC)

@pytest.fixture(scope='class')
def chat_join_request(time, bot):
    if False:
        for i in range(10):
            print('nop')
    cjr = ChatJoinRequest(chat=Chat(1, Chat.SUPERGROUP), from_user=User(2, 'first_name', False, username='user_a'), date=time, bio='bio', invite_link=ChatInviteLink('https://invite.link', User(42, 'creator', False), creates_join_request=False, name='InviteLink', is_revoked=False, is_primary=False), user_chat_id=2)
    cjr.set_bot(bot)
    return cjr

@pytest.fixture()
def chat_join_request_update(bot, chat_join_request):
    if False:
        for i in range(10):
            print('nop')
    return Update(0, chat_join_request=chat_join_request)

class TestChatJoinRequestHandler:
    test_flag = False

    def test_slot_behaviour(self):
        if False:
            print('Hello World!')
        action = ChatJoinRequestHandler(self.callback)
        for attr in action.__slots__:
            assert getattr(action, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(action)) == len(set(mro_slots(action))), 'duplicate slot'

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            while True:
                i = 10
        self.test_flag = False

    async def callback(self, update, context):
        self.test_flag = isinstance(context, CallbackContext) and isinstance(context.bot, Bot) and isinstance(update, Update) and isinstance(context.update_queue, asyncio.Queue) and isinstance(context.job_queue, JobQueue) and isinstance(context.user_data, dict) and isinstance(context.chat_data, dict) and isinstance(context.bot_data, dict) and isinstance(update.chat_join_request, ChatJoinRequest)

    def test_with_chat_id(self, chat_join_request_update):
        if False:
            for i in range(10):
                print('nop')
        handler = ChatJoinRequestHandler(self.callback, chat_id=1)
        assert handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, chat_id=[1])
        assert handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, chat_id=2, username='@user_a')
        assert handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, chat_id=2)
        assert not handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, chat_id=[2])
        assert not handler.check_update(chat_join_request_update)

    def test_with_username(self, chat_join_request_update):
        if False:
            i = 10
            return i + 15
        handler = ChatJoinRequestHandler(self.callback, username='user_a')
        assert handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, username='@user_a')
        assert handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, username=['user_a'])
        assert handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, username=['@user_a'])
        assert handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, chat_id=1, username='@user_b')
        assert handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, username='user_b')
        assert not handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, username='@user_b')
        assert not handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, username=['user_b'])
        assert not handler.check_update(chat_join_request_update)
        handler = ChatJoinRequestHandler(self.callback, username=['@user_b'])
        assert not handler.check_update(chat_join_request_update)
        chat_join_request_update.chat_join_request.from_user._unfreeze()
        chat_join_request_update.chat_join_request.from_user.username = None
        assert not handler.check_update(chat_join_request_update)

    def test_other_update_types(self, false_update):
        if False:
            print('Hello World!')
        handler = ChatJoinRequestHandler(self.callback)
        assert not handler.check_update(false_update)
        assert not handler.check_update(True)

    async def test_context(self, app, chat_join_request_update):
        handler = ChatJoinRequestHandler(callback=self.callback)
        app.add_handler(handler)
        async with app:
            await app.process_update(chat_join_request_update)
            assert self.test_flag