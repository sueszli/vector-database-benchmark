import asyncio
import pytest
from telegram import Bot, CallbackQuery, Chat, ChosenInlineResult, InlineQuery, Message, PreCheckoutQuery, ShippingQuery, Update, User
from telegram.ext import CallbackContext, JobQueue, PreCheckoutQueryHandler
from tests.auxil.slots import mro_slots
message = Message(1, None, Chat(1, ''), from_user=User(1, '', False), text='Text')
params = [{'message': message}, {'edited_message': message}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat', message=message)}, {'channel_post': message}, {'edited_channel_post': message}, {'inline_query': InlineQuery(1, User(1, '', False), '', '')}, {'chosen_inline_result': ChosenInlineResult('id', User(1, '', False), '')}, {'shipping_query': ShippingQuery('id', User(1, '', False), '', None)}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat')}]
ids = ('message', 'edited_message', 'callback_query', 'channel_post', 'edited_channel_post', 'inline_query', 'chosen_inline_result', 'shipping_query', 'callback_query_without_message')

@pytest.fixture(scope='class', params=params, ids=ids)
def false_update(request):
    if False:
        while True:
            i = 10
    return Update(update_id=1, **request.param)

@pytest.fixture(scope='class')
def pre_checkout_query():
    if False:
        for i in range(10):
            print('nop')
    return Update(1, pre_checkout_query=PreCheckoutQuery('id', User(1, 'test user', False), 'EUR', 223, 'invoice_payload'))

class TestPreCheckoutQueryHandler:
    test_flag = False

    def test_slot_behaviour(self):
        if False:
            for i in range(10):
                print('nop')
        inst = PreCheckoutQueryHandler(self.callback)
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            print('Hello World!')
        self.test_flag = False

    async def callback(self, update, context):
        self.test_flag = isinstance(context, CallbackContext) and isinstance(context.bot, Bot) and isinstance(update, Update) and isinstance(context.update_queue, asyncio.Queue) and isinstance(context.job_queue, JobQueue) and isinstance(context.user_data, dict) and (context.chat_data is None) and isinstance(context.bot_data, dict) and isinstance(update.pre_checkout_query, PreCheckoutQuery)

    def test_other_update_types(self, false_update):
        if False:
            for i in range(10):
                print('nop')
        handler = PreCheckoutQueryHandler(self.callback)
        assert not handler.check_update(false_update)

    async def test_context(self, app, pre_checkout_query):
        handler = PreCheckoutQueryHandler(self.callback)
        app.add_handler(handler)
        async with app:
            await app.process_update(pre_checkout_query)
        assert self.test_flag