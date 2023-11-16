import asyncio
import pytest
from telegram import Bot, CallbackQuery, Chat, ChosenInlineResult, InlineQuery, Location, Message, PreCheckoutQuery, ShippingQuery, Update, User
from telegram.ext import CallbackContext, InlineQueryHandler, JobQueue
from tests.auxil.slots import mro_slots
message = Message(1, None, Chat(1, ''), from_user=User(1, '', False), text='Text')
params = [{'message': message}, {'edited_message': message}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat', message=message)}, {'channel_post': message}, {'edited_channel_post': message}, {'chosen_inline_result': ChosenInlineResult('id', User(1, '', False), '')}, {'shipping_query': ShippingQuery('id', User(1, '', False), '', None)}, {'pre_checkout_query': PreCheckoutQuery('id', User(1, '', False), '', 0, '')}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat')}]
ids = ('message', 'edited_message', 'callback_query', 'channel_post', 'edited_channel_post', 'chosen_inline_result', 'shipping_query', 'pre_checkout_query', 'callback_query_without_message')

@pytest.fixture(scope='class', params=params, ids=ids)
def false_update(request):
    if False:
        return 10
    return Update(update_id=2, **request.param)

@pytest.fixture()
def inline_query(bot):
    if False:
        return 10
    update = Update(0, inline_query=InlineQuery('id', User(2, 'test user', False), 'test query', offset='22', location=Location(latitude=-23.691288, longitude=-46.788279)))
    update._unfreeze()
    update.inline_query._unfreeze()
    return update

class TestInlineQueryHandler:
    test_flag = False

    def test_slot_behaviour(self):
        if False:
            for i in range(10):
                print('nop')
        handler = InlineQueryHandler(self.callback)
        for attr in handler.__slots__:
            assert getattr(handler, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(handler)) == len(set(mro_slots(handler))), 'duplicate slot'

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            while True:
                i = 10
        self.test_flag = False

    async def callback(self, update, context):
        self.test_flag = isinstance(context, CallbackContext) and isinstance(context.bot, Bot) and isinstance(update, Update) and isinstance(context.update_queue, asyncio.Queue) and isinstance(context.job_queue, JobQueue) and isinstance(context.user_data, dict) and (context.chat_data is None) and isinstance(context.bot_data, dict) and isinstance(update.inline_query, InlineQuery)

    def callback_pattern(self, update, context):
        if False:
            i = 10
            return i + 15
        if context.matches[0].groups():
            self.test_flag = context.matches[0].groups() == ('t', ' query')
        if context.matches[0].groupdict():
            self.test_flag = context.matches[0].groupdict() == {'begin': 't', 'end': ' query'}

    def test_other_update_types(self, false_update):
        if False:
            print('Hello World!')
        handler = InlineQueryHandler(self.callback)
        assert not handler.check_update(false_update)

    async def test_context(self, app, inline_query):
        handler = InlineQueryHandler(self.callback)
        app.add_handler(handler)
        async with app:
            await app.process_update(inline_query)
        assert self.test_flag

    async def test_context_pattern(self, app, inline_query):
        handler = InlineQueryHandler(self.callback_pattern, pattern='(?P<begin>.*)est(?P<end>.*)')
        app.add_handler(handler)
        async with app:
            await app.process_update(inline_query)
            assert self.test_flag
            app.remove_handler(handler)
            handler = InlineQueryHandler(self.callback_pattern, pattern='(t)est(.*)')
            app.add_handler(handler)
            await app.process_update(inline_query)
            assert self.test_flag
            update = Update(update_id=0, inline_query=InlineQuery(id='id', from_user=None, query='', offset=''))
            update.inline_query._unfreeze()
            assert not handler.check_update(update)
            update.inline_query.query = 'not_a_match'
            assert not handler.check_update(update)

    @pytest.mark.parametrize('chat_types', [[Chat.SENDER], [Chat.SENDER, Chat.SUPERGROUP], []])
    @pytest.mark.parametrize(('chat_type', 'result'), [(Chat.SENDER, True), (Chat.CHANNEL, False), (None, False)])
    async def test_chat_types(self, app, inline_query, chat_types, chat_type, result):
        try:
            inline_query.inline_query.chat_type = chat_type
            handler = InlineQueryHandler(self.callback, chat_types=chat_types)
            app.add_handler(handler)
            async with app:
                await app.process_update(inline_query)
            if not chat_types:
                assert self.test_flag is False
            else:
                assert self.test_flag == result
        finally:
            inline_query.inline_query.chat_type = None