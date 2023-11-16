import asyncio
import pytest
from telegram import Bot, CallbackQuery, Chat, ChosenInlineResult, Message, Poll, PollOption, PreCheckoutQuery, ShippingQuery, Update, User
from telegram.ext import CallbackContext, JobQueue, PollHandler
from tests.auxil.slots import mro_slots
message = Message(1, None, Chat(1, ''), from_user=User(1, '', False), text='Text')
params = [{'message': message}, {'edited_message': message}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat', message=message)}, {'channel_post': message}, {'edited_channel_post': message}, {'chosen_inline_result': ChosenInlineResult('id', User(1, '', False), '')}, {'shipping_query': ShippingQuery('id', User(1, '', False), '', None)}, {'pre_checkout_query': PreCheckoutQuery('id', User(1, '', False), '', 0, '')}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat')}]
ids = ('message', 'edited_message', 'callback_query', 'channel_post', 'edited_channel_post', 'chosen_inline_result', 'shipping_query', 'pre_checkout_query', 'callback_query_without_message')

@pytest.fixture(scope='class', params=params, ids=ids)
def false_update(request):
    if False:
        i = 10
        return i + 15
    return Update(update_id=2, **request.param)

@pytest.fixture()
def poll(bot):
    if False:
        i = 10
        return i + 15
    return Update(0, poll=Poll(1, 'question', [PollOption('1', 0), PollOption('2', 0)], 0, False, False, Poll.REGULAR, True))

class TestPollHandler:
    test_flag = False

    def test_slot_behaviour(self):
        if False:
            return 10
        inst = PollHandler(self.callback)
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_flag = False

    async def callback(self, update, context):
        self.test_flag = isinstance(context, CallbackContext) and isinstance(context.bot, Bot) and isinstance(update, Update) and isinstance(context.update_queue, asyncio.Queue) and isinstance(context.job_queue, JobQueue) and (context.user_data is None) and (context.chat_data is None) and isinstance(context.bot_data, dict) and isinstance(update.poll, Poll)

    def test_other_update_types(self, false_update):
        if False:
            return 10
        handler = PollHandler(self.callback)
        assert not handler.check_update(false_update)

    async def test_context(self, app, poll):
        handler = PollHandler(self.callback)
        app.add_handler(handler)
        async with app:
            await app.process_update(poll)
        assert self.test_flag