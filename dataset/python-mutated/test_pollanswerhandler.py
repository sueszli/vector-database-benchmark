import asyncio
import pytest
from telegram import Bot, CallbackQuery, Chat, ChosenInlineResult, Message, PollAnswer, PreCheckoutQuery, ShippingQuery, Update, User
from telegram.ext import CallbackContext, JobQueue, PollAnswerHandler
from tests.auxil.slots import mro_slots
message = Message(1, None, Chat(1, ''), from_user=User(1, '', False), text='Text')
params = [{'message': message}, {'edited_message': message}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat', message=message)}, {'channel_post': message}, {'edited_channel_post': message}, {'chosen_inline_result': ChosenInlineResult('id', User(1, '', False), '')}, {'shipping_query': ShippingQuery('id', User(1, '', False), '', None)}, {'pre_checkout_query': PreCheckoutQuery('id', User(1, '', False), '', 0, '')}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat')}]
ids = ('message', 'edited_message', 'callback_query', 'channel_post', 'edited_channel_post', 'chosen_inline_result', 'shipping_query', 'pre_checkout_query', 'callback_query_without_message')

@pytest.fixture(scope='class', params=params, ids=ids)
def false_update(request):
    if False:
        print('Hello World!')
    return Update(update_id=2, **request.param)

@pytest.fixture()
def poll_answer(bot):
    if False:
        print('Hello World!')
    return Update(0, poll_answer=PollAnswer(1, [0, 1], User(2, 'test user', False), Chat(1, '')))

class TestPollAnswerHandler:
    test_flag = False

    def test_slot_behaviour(self):
        if False:
            print('Hello World!')
        handler = PollAnswerHandler(self.callback)
        for attr in handler.__slots__:
            assert getattr(handler, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(handler)) == len(set(mro_slots(handler))), 'duplicate slot'

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            print('Hello World!')
        self.test_flag = False

    async def callback(self, update, context):
        self.test_flag = isinstance(context, CallbackContext) and isinstance(context.bot, Bot) and isinstance(update, Update) and isinstance(context.update_queue, asyncio.Queue) and isinstance(context.job_queue, JobQueue) and isinstance(context.user_data, dict) and (context.chat_data is None) and isinstance(context.bot_data, dict) and isinstance(update.poll_answer, PollAnswer)

    def test_other_update_types(self, false_update):
        if False:
            return 10
        handler = PollAnswerHandler(self.callback)
        assert not handler.check_update(false_update)

    async def test_context(self, app, poll_answer):
        handler = PollAnswerHandler(self.callback)
        app.add_handler(handler)
        async with app:
            await app.process_update(poll_answer)
        assert self.test_flag