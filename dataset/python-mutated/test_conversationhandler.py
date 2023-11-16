"""Persistence of conversations is tested in test_basepersistence.py"""
import asyncio
import functools
import logging
from pathlib import Path
from warnings import filterwarnings
import pytest
from telegram import CallbackQuery, Chat, ChosenInlineResult, InlineQuery, Message, MessageEntity, PreCheckoutQuery, ShippingQuery, Update, User
from telegram.ext import ApplicationBuilder, ApplicationHandlerStop, CallbackContext, CallbackQueryHandler, ChosenInlineResultHandler, CommandHandler, ConversationHandler, Defaults, InlineQueryHandler, JobQueue, MessageHandler, PollAnswerHandler, PollHandler, PreCheckoutQueryHandler, ShippingQueryHandler, StringCommandHandler, StringRegexHandler, TypeHandler, filters
from telegram.warnings import PTBUserWarning
from tests.auxil.build_messages import make_command_message
from tests.auxil.files import PROJECT_ROOT_PATH
from tests.auxil.pytest_classes import PytestBot, make_bot
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='class')
def user1():
    if False:
        for i in range(10):
            print('nop')
    return User(first_name='Misses Test', id=123, is_bot=False)

@pytest.fixture(scope='class')
def user2():
    if False:
        return 10
    return User(first_name='Mister Test', id=124, is_bot=False)

def raise_ahs(func):
    if False:
        print('Hello World!')

    @functools.wraps(func)
    async def decorator(self, *args, **kwargs):
        result = await func(self, *args, **kwargs)
        if self.raise_app_handler_stop:
            raise ApplicationHandlerStop(result)
        return result
    return decorator

class TestConversationHandler:
    """Persistence of conversations is tested in test_basepersistence.py"""
    (END, THIRSTY, BREWING, DRINKING, CODING) = range(-1, 4)
    (HOLDING, SIPPING, SWALLOWING, REPLENISHING, STOPPING) = map(chr, range(ord('a'), ord('f')))
    (current_state, entry_points, states, fallbacks) = (None, None, None, None)
    group = Chat(0, Chat.GROUP)
    second_group = Chat(1, Chat.GROUP)
    raise_app_handler_stop = False
    test_flag = False

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            return 10
        self.raise_app_handler_stop = False
        self.test_flag = False
        self.current_state = {}
        self.entry_points = [CommandHandler('start', self.start)]
        self.states = {self.THIRSTY: [CommandHandler('brew', self.brew), CommandHandler('wait', self.start)], self.BREWING: [CommandHandler('pourCoffee', self.drink)], self.DRINKING: [CommandHandler('startCoding', self.code), CommandHandler('drinkMore', self.drink), CommandHandler('end', self.end)], self.CODING: [CommandHandler('keepCoding', self.code), CommandHandler('gettingThirsty', self.start), CommandHandler('drinkMore', self.drink)]}
        self.fallbacks = [CommandHandler('eat', self.start)]
        self.is_timeout = False
        self.nested_states = {self.THIRSTY: [CommandHandler('brew', self.brew), CommandHandler('wait', self.start)], self.BREWING: [CommandHandler('pourCoffee', self.drink)], self.CODING: [CommandHandler('keepCoding', self.code), CommandHandler('gettingThirsty', self.start), CommandHandler('drinkMore', self.drink)]}
        self.drinking_entry_points = [CommandHandler('hold', self.hold)]
        self.drinking_states = {self.HOLDING: [CommandHandler('sip', self.sip)], self.SIPPING: [CommandHandler('swallow', self.swallow)], self.SWALLOWING: [CommandHandler('hold', self.hold)]}
        self.drinking_fallbacks = [CommandHandler('replenish', self.replenish), CommandHandler('stop', self.stop), CommandHandler('end', self.end), CommandHandler('startCoding', self.code), CommandHandler('drinkMore', self.drink)]
        self.drinking_entry_points.extend(self.drinking_fallbacks)
        self.drinking_map_to_parent = {self.REPLENISHING: self.BREWING, self.STOPPING: self.END, self.END: self.CODING, self.CODING: self.CODING, self.DRINKING: self.DRINKING}

    def _set_state(self, update, state):
        if False:
            return 10
        self.current_state[update.message.from_user.id] = state
        return state

    @raise_ahs
    async def start(self, update, context):
        if isinstance(update, Update):
            return self._set_state(update, self.THIRSTY)
        return self._set_state(context.bot, self.THIRSTY)

    @raise_ahs
    async def end(self, update, context):
        return self._set_state(update, self.END)

    @raise_ahs
    async def start_end(self, update, context):
        return self._set_state(update, self.END)

    @raise_ahs
    async def start_none(self, update, context):
        return self._set_state(update, None)

    @raise_ahs
    async def brew(self, update, context):
        if isinstance(update, Update):
            return self._set_state(update, self.BREWING)
        return self._set_state(context.bot, self.BREWING)

    @raise_ahs
    async def drink(self, update, context):
        return self._set_state(update, self.DRINKING)

    @raise_ahs
    async def code(self, update, context):
        return self._set_state(update, self.CODING)

    @raise_ahs
    async def passout(self, update, context):
        assert update.message.text == '/brew'
        assert isinstance(update, Update)
        self.is_timeout = True

    @raise_ahs
    async def passout2(self, update, context):
        assert isinstance(update, Update)
        self.is_timeout = True

    @raise_ahs
    async def passout_context(self, update, context):
        assert update.message.text == '/brew'
        assert isinstance(context, CallbackContext)
        self.is_timeout = True

    @raise_ahs
    async def passout2_context(self, update, context):
        assert isinstance(context, CallbackContext)
        self.is_timeout = True

    @raise_ahs
    async def hold(self, update, context):
        return self._set_state(update, self.HOLDING)

    @raise_ahs
    async def sip(self, update, context):
        return self._set_state(update, self.SIPPING)

    @raise_ahs
    async def swallow(self, update, context):
        return self._set_state(update, self.SWALLOWING)

    @raise_ahs
    async def replenish(self, update, context):
        return self._set_state(update, self.REPLENISHING)

    @raise_ahs
    async def stop(self, update, context):
        return self._set_state(update, self.STOPPING)

    def test_slot_behaviour(self):
        if False:
            while True:
                i = 10
        handler = ConversationHandler(entry_points=[], states={}, fallbacks=[])
        for attr in handler.__slots__:
            assert getattr(handler, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(handler)) == len(set(mro_slots(handler))), 'duplicate slot'

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        entry_points = []
        states = {}
        fallbacks = []
        map_to_parent = {}
        ch = ConversationHandler(entry_points=entry_points, states=states, fallbacks=fallbacks, per_chat='per_chat', per_user='per_user', per_message='per_message', persistent='persistent', name='name', allow_reentry='allow_reentry', conversation_timeout=42, map_to_parent=map_to_parent)
        assert ch.entry_points is entry_points
        assert ch.states is states
        assert ch.fallbacks is fallbacks
        assert ch.map_to_parent is map_to_parent
        assert ch.per_chat == 'per_chat'
        assert ch.per_user == 'per_user'
        assert ch.per_message == 'per_message'
        assert ch.persistent == 'persistent'
        assert ch.name == 'name'
        assert ch.allow_reentry == 'allow_reentry'

    def test_init_persistent_no_name(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError, match="can't be persistent when handler is unnamed"):
            ConversationHandler(self.entry_points, states=self.states, fallbacks=[], persistent=True)

    def test_repr_no_truncation(self):
        if False:
            print('Hello World!')
        ch = ConversationHandler(name='test_handler', entry_points=[], states=self.drinking_states, fallbacks=[])
        assert repr(ch) == "ConversationHandler[name=test_handler, states={'a': [CommandHandler[callback=TestConversationHandler.sip]], 'b': [CommandHandler[callback=TestConversationHandler.swallow]], 'c': [CommandHandler[callback=TestConversationHandler.hold]]}]"

    def test_repr_with_truncation(self):
        if False:
            while True:
                i = 10
        from copy import copy
        states = copy(self.drinking_states)
        states['extra_to_be_truncated'] = [CommandHandler('foo', self.start)]
        ch = ConversationHandler(name='test_handler', entry_points=[], states=states, fallbacks=[])
        assert repr(ch) == "ConversationHandler[name=test_handler, states={'a': [CommandHandler[callback=TestConversationHandler.sip]], 'b': [CommandHandler[callback=TestConversationHandler.swallow]], 'c': [CommandHandler[callback=TestConversationHandler.hold]], ...}]"

    async def test_check_update_returns_non(self, app, user1):
        """checks some cases where updates should not be handled"""
        conv_handler = ConversationHandler([], {}, [], per_message=True, per_chat=True)
        assert not conv_handler.check_update('not an Update')
        assert not conv_handler.check_update(Update(0))
        assert not conv_handler.check_update(Update(0, callback_query=CallbackQuery('1', from_user=user1, chat_instance='1')))

    async def test_handlers_generate_warning(self, recwarn):
        """this function tests all handler + per_* setting combinations."""
        filterwarnings(action='always', category=PTBUserWarning)

        class NotUpdate:
            pass
        recwarn.clear()
        ConversationHandler(entry_points=[StringCommandHandler('code', self.code)], states={self.BREWING: [StringRegexHandler('code', self.code), PollHandler(self.code), TypeHandler(NotUpdate, self.code)]}, fallbacks=[TypeHandler(Update, self.code)])
        ConversationHandler(entry_points=[ShippingQueryHandler(self.code)], states={self.BREWING: [InlineQueryHandler(self.code), PreCheckoutQueryHandler(self.code), PollAnswerHandler(self.code)]}, fallbacks=[ChosenInlineResultHandler(self.code)], per_chat=True)
        ConversationHandler(entry_points=[CallbackQueryHandler(self.code)], states={self.BREWING: [CommandHandler('code', self.code)]}, fallbacks=[CallbackQueryHandler(self.code)], per_message=True)
        ConversationHandler(entry_points=[CommandHandler('code', self.code)], states={self.BREWING: [CommandHandler('code', self.code)]}, fallbacks=[CallbackQueryHandler(self.code)], per_message=False)
        child = ConversationHandler(entry_points=[CommandHandler('code', self.code)], states={self.BREWING: [CommandHandler('code', self.code)]}, fallbacks=[CommandHandler('code', self.code)])
        ConversationHandler(entry_points=[CommandHandler('code', self.code)], states={self.BREWING: [child]}, fallbacks=[CommandHandler('code', self.code)], conversation_timeout=42)
        ConversationHandler(entry_points=[CallbackQueryHandler(self.code, 'code')], states={self.BREWING: [CallbackQueryHandler(self.code, 'code')]}, fallbacks=[CallbackQueryHandler(self.code, 'code')], per_message=True, per_chat=False)
        assert len(recwarn) == 13
        assert str(recwarn[0].message) == 'The `ConversationHandler` only handles updates of type `telegram.Update`. StringCommandHandler handles updates of type `str`.'
        assert str(recwarn[1].message) == 'The `ConversationHandler` only handles updates of type `telegram.Update`. StringRegexHandler handles updates of type `str`.'
        assert str(recwarn[2].message) == 'PollHandler will never trigger in a conversation since it has no information about the chat or the user who voted in it. Do you mean the `PollAnswerHandler`?'
        assert str(recwarn[3].message) == 'The `ConversationHandler` only handles updates of type `telegram.Update`. The TypeHandler is set to handle NotUpdate.'
        per_faq_link = ' Read this FAQ entry to learn more about the per_* settings: https://github.com/python-telegram-bot/python-telegram-bot/wiki/Frequently-Asked-Questions#what-do-the-per_-settings-in-conversationhandler-do.'
        assert str(recwarn[4].message) == "Updates handled by ShippingQueryHandler only have information about the user, so this handler won't ever be triggered if `per_chat=True`." + per_faq_link
        assert str(recwarn[5].message) == "Updates handled by ChosenInlineResultHandler only have information about the user, so this handler won't ever be triggered if `per_chat=True`." + per_faq_link
        assert str(recwarn[6].message) == "Updates handled by InlineQueryHandler only have information about the user, so this handler won't ever be triggered if `per_chat=True`." + per_faq_link
        assert str(recwarn[7].message) == "Updates handled by PreCheckoutQueryHandler only have information about the user, so this handler won't ever be triggered if `per_chat=True`." + per_faq_link
        assert str(recwarn[8].message) == "Updates handled by PollAnswerHandler only have information about the user, so this handler won't ever be triggered if `per_chat=True`." + per_faq_link
        assert str(recwarn[9].message) == "If 'per_message=True', all entry points, state handlers, and fallbacks must be 'CallbackQueryHandler', since no other handlers have a message context." + per_faq_link
        assert str(recwarn[10].message) == "If 'per_message=False', 'CallbackQueryHandler' will not be tracked for every message." + per_faq_link
        assert str(recwarn[11].message) == 'Using `conversation_timeout` with nested conversations is currently not supported. You can still try to use it, but it will likely behave differently from what you expect.'
        assert str(recwarn[12].message) == "If 'per_message=True' is used, 'per_chat=True' should also be used, since message IDs are not globally unique."
        for warning in recwarn:
            assert warning.category is PTBUserWarning
            assert warning.filename == __file__, 'incorrect stacklevel!'

    @pytest.mark.parametrize('attr', ['entry_points', 'states', 'fallbacks', 'per_chat', 'per_user', 'per_message', 'name', 'persistent', 'allow_reentry', 'conversation_timeout', 'map_to_parent'], indirect=False)
    def test_immutable(self, attr):
        if False:
            while True:
                i = 10
        ch = ConversationHandler(entry_points=[], states={}, fallbacks=[])
        with pytest.raises(AttributeError, match=f'You can not assign a new value to {attr}'):
            setattr(ch, attr, True)

    def test_per_all_false(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match="can't all be 'False'"):
            ConversationHandler(entry_points=[], states={}, fallbacks=[], per_chat=False, per_user=False, per_message=False)

    @pytest.mark.parametrize('raise_ahs', [True, False])
    async def test_basic_and_app_handler_stop(self, app, bot, user1, user2, raise_ahs):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks)
        app.add_handler(handler)

        async def callback(_, __):
            self.test_flag = True
        app.add_handler(TypeHandler(object, callback), group=100)
        self.raise_app_handler_stop = raise_ahs
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.THIRSTY
            assert self.test_flag == (not raise_ahs)
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.BREWING
            assert self.test_flag == (not raise_ahs)
            message.text = '/nothing'
            message.entities[0].length = len('/nothing')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.BREWING
            assert self.test_flag is True
            self.test_flag = False
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.DRINKING
            assert self.test_flag == (not raise_ahs)
            message.from_user = user2
            await app.process_update(Update(update_id=0, message=message))
            with pytest.raises(KeyError):
                self.current_state[user2.id]

    async def test_conversation_handler_end(self, caplog, app, bot, user1):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/end'
            message.entities[0].length = len('/end')
            caplog.clear()
            with caplog.at_level(logging.ERROR):
                await app.process_update(Update(update_id=0, message=message))
            assert len(caplog.records) == 0
            assert self.current_state[user1.id] == self.END
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(update_id=0, message=message))

    async def test_conversation_handler_fallback(self, app, bot, user1, user2):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/eat', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/eat'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            with pytest.raises(KeyError):
                self.current_state[user1.id]
            message.text = '/start'
            message.entities[0].length = len('/start')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.THIRSTY
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.BREWING
            message.text = '/eat'
            message.entities[0].length = len('/eat')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.THIRSTY

    async def test_unknown_state_warning(self, app, bot, user1, recwarn):

        def build_callback(state):
            if False:
                print('Hello World!')

            async def callback(_, __):
                return state
            return callback
        handler = ConversationHandler(entry_points=[CommandHandler('start', build_callback(1))], states={1: [TypeHandler(Update, build_callback(69))], 2: [TypeHandler(Update, build_callback(42))]}, fallbacks=self.fallbacks, name='xyz')
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            try:
                await app.process_update(Update(update_id=1, message=message))
            except Exception as exc:
                print(exc)
                raise exc
            assert len(recwarn) == 1
            assert recwarn[0].category is PTBUserWarning
            assert Path(recwarn[0].filename) == PROJECT_ROOT_PATH / 'telegram' / 'ext' / '_conversationhandler.py', 'wrong stacklevel!'
            assert str(recwarn[0].message) == "'callback' returned state 69 which is unknown to the ConversationHandler xyz."

    async def test_conversation_handler_per_chat(self, app, bot, user1, user2):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, per_user=False)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            message.from_user = user2
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            await app.process_update(Update(update_id=0, message=message))
            message.from_user = user1
            message.text = '/startCoding'
            message.entities[0].length = len('/startCoding')
            assert handler.check_update(Update(update_id=0, message=message))
            message.from_user = user2
            assert handler.check_update(Update(update_id=0, message=message))

    async def test_conversation_handler_per_user(self, app, bot, user1):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, per_chat=False)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        message.from_user = None
        assert not handler.check_update(Update(update_id=0, message=message))
        message.from_user = user1
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            message.chat = self.second_group
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            await app.process_update(Update(update_id=0, message=message))
            message.chat = self.group
            message.text = '/startCoding'
            message.entities[0].length = len('/startCoding')
            assert handler.check_update(Update(update_id=0, message=message))
            message.chat = self.second_group
            assert handler.check_update(Update(update_id=0, message=message))

    @pytest.mark.parametrize('inline', [True, False])
    @pytest.mark.filterwarnings("ignore: If 'per_message=True' is used, 'per_chat=True'")
    async def test_conversation_handler_per_message(self, app, bot, user1, user2, inline):

        async def entry(update, context):
            return 1

        async def one(update, context):
            return 2

        async def two(update, context):
            return ConversationHandler.END
        handler = ConversationHandler(entry_points=[CallbackQueryHandler(entry)], states={1: [CallbackQueryHandler(one, pattern='^1$')], 2: [CallbackQueryHandler(two, pattern='^2$')]}, fallbacks=[], per_message=True, per_chat=not inline)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='msg w/ inlinekeyboard') if not inline else None
        if message:
            message.set_bot(bot)
            message._unfreeze()
        inline_message_id = '42' if inline else None
        async with app:
            cbq_1 = CallbackQuery(0, user1, None, message=message, data='1', inline_message_id=inline_message_id)
            cbq_1.set_bot(bot)
            cbq_2 = CallbackQuery(0, user1, None, message=message, data='2', inline_message_id=inline_message_id)
            cbq_2.set_bot(bot)
            cbq_2._unfreeze()
            await app.process_update(Update(update_id=0, callback_query=cbq_1))
            assert handler.check_update(Update(0, callback_query=cbq_1))
            assert not handler.check_update(Update(0, callback_query=cbq_2))
            await app.process_update(Update(update_id=0, callback_query=cbq_1))
            assert not handler.check_update(Update(0, callback_query=cbq_1))
            assert handler.check_update(Update(0, callback_query=cbq_2))
            cbq_2.from_user = user2
            await app.process_update(Update(update_id=0, callback_query=cbq_2))
            cbq_2.from_user = user1
            assert not handler.check_update(Update(0, callback_query=cbq_1))
            assert handler.check_update(Update(0, callback_query=cbq_2))

    async def test_end_on_first_message(self, app, bot, user1):
        handler = ConversationHandler(entry_points=[CommandHandler('start', self.start_end)], states={}, fallbacks=[])
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            assert handler.check_update(Update(update_id=0, message=message))

    async def test_end_on_first_message_non_blocking_handler(self, app, bot, user1):
        handler = ConversationHandler(entry_points=[CommandHandler('start', callback=self.start_end, block=False)], states={}, fallbacks=[])
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.1)
            assert handler.check_update(Update(0, message=message))

    async def test_none_on_first_message(self, app, bot, user1):
        handler = ConversationHandler(entry_points=[MessageHandler(filters.ALL, self.start_none)], states={}, fallbacks=[])
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start')
        message.set_bot(bot)
        message._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            assert handler.check_update(Update(0, message=message))

    async def test_none_on_first_message_non_blocking_handler(self, app, bot, user1):
        handler = ConversationHandler(entry_points=[CommandHandler('start', self.start_none, block=False)], states={}, fallbacks=[])
        app.add_handler(handler)
        message = Message(0, None, self.group, text='/start', from_user=user1, entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.1)
            assert handler.check_update(Update(0, message=message))

    async def test_per_chat_message_without_chat(self, bot, user1):
        handler = ConversationHandler(entry_points=[CommandHandler('start', self.start_end)], states={}, fallbacks=[])
        cbq = CallbackQuery(0, user1, None, None)
        cbq.set_bot(bot)
        update = Update(0, callback_query=cbq)
        assert not handler.check_update(update)

    async def test_channel_message_without_chat(self, bot):
        handler = ConversationHandler(entry_points=[MessageHandler(filters.ALL, self.start_end)], states={}, fallbacks=[])
        message = Message(0, date=None, chat=Chat(0, Chat.CHANNEL, 'Misses Test'))
        message.set_bot(bot)
        message._unfreeze()
        update = Update(0, channel_post=message)
        assert not handler.check_update(update)
        update = Update(0, edited_channel_post=message)
        assert not handler.check_update(update)

    async def test_all_update_types(self, app, bot, user1):
        handler = ConversationHandler(entry_points=[CommandHandler('start', self.start_end)], states={}, fallbacks=[])
        message = Message(0, None, self.group, from_user=user1, text='ignore')
        message.set_bot(bot)
        message._unfreeze()
        callback_query = CallbackQuery(0, user1, None, message=message, data='data')
        callback_query.set_bot(bot)
        chosen_inline_result = ChosenInlineResult(0, user1, 'query')
        chosen_inline_result.set_bot(bot)
        inline_query = InlineQuery(0, user1, 'query', offset='')
        inline_query.set_bot(bot)
        pre_checkout_query = PreCheckoutQuery(0, user1, 'USD', 100, [])
        pre_checkout_query.set_bot(bot)
        shipping_query = ShippingQuery(0, user1, [], None)
        shipping_query.set_bot(bot)
        assert not handler.check_update(Update(0, callback_query=callback_query))
        assert not handler.check_update(Update(0, chosen_inline_result=chosen_inline_result))
        assert not handler.check_update(Update(0, inline_query=inline_query))
        assert not handler.check_update(Update(0, message=message))
        assert not handler.check_update(Update(0, pre_checkout_query=pre_checkout_query))
        assert not handler.check_update(Update(0, shipping_query=shipping_query))

    @pytest.mark.parametrize('jq', [True, False])
    async def test_no_running_job_queue_warning(self, app, bot, user1, recwarn, jq):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, conversation_timeout=0.5)
        if not jq:
            app = ApplicationBuilder().token(bot.token).job_queue(None).build()
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.5)
            if jq:
                assert len(recwarn) == 1
            else:
                assert len(recwarn) == 2
            assert str(recwarn[0].message if jq else recwarn[1].message).startswith('Ignoring `conversation_timeout`')
            assert ('is not running' if jq else 'No `JobQueue` set up.') in str(recwarn[0].message)
            for warning in recwarn:
                assert warning.category is PTBUserWarning
                assert Path(warning.filename) == PROJECT_ROOT_PATH / 'telegram' / 'ext' / '_conversationhandler.py', 'wrong stacklevel!'

    async def test_schedule_job_exception(self, app, bot, user1, monkeypatch, caplog):

        def mocked_run_once(*a, **kw):
            if False:
                i = 10
                return i + 15
            raise Exception('job error')

        class DictJB(JobQueue):
            pass
        app = ApplicationBuilder().token(bot.token).job_queue(DictJB()).build()
        monkeypatch.setattr(app.job_queue, 'run_once', mocked_run_once)
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, conversation_timeout=100)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.start()
            with caplog.at_level(logging.ERROR):
                await app.process_update(Update(update_id=0, message=message))
                await asyncio.sleep(0.5)
            assert len(caplog.records) == 1
            assert caplog.records[0].message == 'Failed to schedule timeout.'
            assert caplog.records[0].name == 'telegram.ext.ConversationHandler'
            assert str(caplog.records[0].exc_info[1]) == 'job error'
            await app.stop()

    @pytest.mark.parametrize(argnames='test_type', argvalues=['none', 'exception'])
    async def test_non_blocking_exception_or_none(self, app, bot, user1, caplog, test_type):
        """Here we make sure that when a non-blocking handler raises an
        exception or returns None, the state isn't changed.
        """
        error = Exception('task exception')

        async def conv_entry(*a, **kw):
            return 1

        async def raise_error(*a, **kw):
            if test_type == 'none':
                return
            raise error
        handler = ConversationHandler(entry_points=[CommandHandler('start', conv_entry)], states={1: [MessageHandler(filters.Text(['error']), raise_error)]}, fallbacks=self.fallbacks, block=False)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.1)
            message.text = 'error'
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.1)
            caplog.clear()
            with caplog.at_level(logging.ERROR):
                assert handler.check_update(Update(0, message=message))
            if test_type == 'exception':
                assert len(caplog.records) == 1
                assert caplog.records[0].name == 'telegram.ext.ConversationHandler'
                assert caplog.records[0].message == 'Task function raised exception. Falling back to old state 1'
                assert caplog.records[0].exc_info[1] is None
            else:
                assert len(caplog.records) == 0

    async def test_non_blocking_entry_point_exception(self, app, bot, user1, caplog):
        """Here we make sure that when a non-blocking entry point raises an
        exception, the state isn't changed.
        """
        error = Exception('task exception')

        async def raise_error(*a, **kw):
            raise error
        handler = ConversationHandler(entry_points=[CommandHandler('start', raise_error, block=False)], states={}, fallbacks=self.fallbacks)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.1)
            caplog.clear()
            with caplog.at_level(logging.ERROR):
                assert handler.check_update(Update(0, message=message))
            assert len(caplog.records) == 1
            assert caplog.records[0].name == 'telegram.ext.ConversationHandler'
            assert caplog.records[0].message == 'Task function raised exception. Falling back to old state None'
            assert caplog.records[0].exc_info[1] is None

    async def test_conversation_timeout(self, app, bot, user1):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, conversation_timeout=0.5)
        app.add_handler(handler)
        start_message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        start_message.set_bot(bot)
        brew_message = Message(0, None, self.group, from_user=user1, text='/brew', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/brew'))])
        brew_message.set_bot(bot)
        pour_coffee_message = Message(0, None, self.group, from_user=user1, text='/pourCoffee', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/pourCoffee'))])
        pour_coffee_message.set_bot(bot)
        async with app:
            await app.start()
            await app.process_update(Update(update_id=0, message=start_message))
            assert handler.check_update(Update(0, message=brew_message))
            await asyncio.sleep(0.75)
            assert handler.check_update(Update(0, message=start_message))
            await app.process_update(Update(update_id=1, message=start_message))
            assert handler.check_update(Update(0, message=brew_message))
            await app.process_update(Update(update_id=2, message=brew_message))
            assert handler.check_update(Update(0, message=pour_coffee_message))
            await asyncio.sleep(0.75)
            assert handler.check_update(Update(0, message=start_message))
            await app.stop()

    async def test_timeout_not_triggered_on_conv_end_non_blocking(self, bot, app, user1):

        def timeout(*a, **kw):
            if False:
                while True:
                    i = 10
            self.test_flag = True
        self.states.update({ConversationHandler.TIMEOUT: [TypeHandler(Update, timeout)]})
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, conversation_timeout=0.5, block=False)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.1)
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=1, message=message))
            await asyncio.sleep(0.1)
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            await app.process_update(Update(update_id=2, message=message))
            await asyncio.sleep(0.1)
            message.text = '/end'
            message.entities[0].length = len('/end')
            await app.process_update(Update(update_id=3, message=message))
            await asyncio.sleep(1)
            assert self.test_flag is False

    async def test_conversation_timeout_application_handler_stop(self, app, bot, user1, recwarn):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, conversation_timeout=0.5)

        def timeout(*args, **kwargs):
            if False:
                return 10
            raise ApplicationHandlerStop
        self.states.update({ConversationHandler.TIMEOUT: [TypeHandler(Update, timeout)]})
        app.add_handler(handler)
        message = Message(0, None, self.group, text='/start', from_user=user1, entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        brew_message = Message(0, None, self.group, from_user=user1, text='/brew', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/brew'))])
        brew_message.set_bot(bot)
        async with app:
            await app.start()
            await app.process_update(Update(update_id=0, message=message))
            assert handler.check_update(Update(0, message=brew_message))
            await app.process_update(Update(0, message=brew_message))
            await asyncio.sleep(0.9)
            assert handler.check_update(Update(0, message=message))
            assert len(recwarn) == 1
            assert str(recwarn[0].message).startswith('ApplicationHandlerStop in TIMEOUT')
            assert recwarn[0].category is PTBUserWarning
            assert Path(recwarn[0].filename) == PROJECT_ROOT_PATH / 'telegram' / 'ext' / '_jobqueue.py', 'wrong stacklevel!'
            await app.stop()

    async def test_conversation_handler_timeout_update_and_context(self, app, bot, user1):
        context = None

        async def start_callback(u, c):
            nonlocal context, self
            context = c
            return await self.start(u, c)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        update = Update(update_id=0, message=message)

        async def timeout_callback(u, c):
            nonlocal update, context
            assert u is update
            assert c is context
            self.is_timeout = u is update and c is context
        states = self.states
        timeout_handler = CommandHandler('start', timeout_callback)
        states.update({ConversationHandler.TIMEOUT: [timeout_handler]})
        handler = ConversationHandler(entry_points=[CommandHandler('start', start_callback)], states=states, fallbacks=self.fallbacks, conversation_timeout=0.5)
        app.add_handler(handler)
        async with app:
            await app.start()
            await app.process_update(update)
            await asyncio.sleep(0.9)
            assert handler.check_update(Update(0, message=message))
            assert self.is_timeout
            await app.stop()

    @pytest.mark.flaky(3, 1)
    async def test_conversation_timeout_keeps_extending(self, app, bot, user1):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, conversation_timeout=0.5)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.start()
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            assert handler.check_update(Update(0, message=message))
            await asyncio.sleep(0.35)
            assert handler.check_update(Update(0, message=message))
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            assert handler.check_update(Update(0, message=message))
            await asyncio.sleep(0.25)
            assert handler.check_update(Update(0, message=message))
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/startCoding'
            message.entities[0].length = len('/startCoding')
            assert handler.check_update(Update(0, message=message))
            await asyncio.sleep(0.4)
            assert handler.check_update(Update(0, message=message))
            await asyncio.sleep(0.3)
            assert not handler.check_update(Update(0, message=message))
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(0, message=message))
            await app.stop()

    async def test_conversation_timeout_two_users(self, app, bot, user1, user2):
        handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, conversation_timeout=0.5)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.start()
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            assert handler.check_update(Update(0, message=message))
            message.from_user = user2
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(0, message=message))
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            assert handler.check_update(Update(0, message=message))
            await asyncio.sleep(0.7)
            message.text = '/start'
            message.entities[0].length = len('/start')
            message.from_user = user1
            assert handler.check_update(Update(0, message=message))
            message.from_user = user2
            assert handler.check_update(Update(0, message=message))
            await app.stop()

    async def test_conversation_handler_timeout_state(self, app, bot, user1):
        states = self.states
        states.update({ConversationHandler.TIMEOUT: [CommandHandler('brew', self.passout), MessageHandler(~filters.Regex('oding'), self.passout2)]})
        handler = ConversationHandler(entry_points=self.entry_points, states=states, fallbacks=self.fallbacks, conversation_timeout=0.5)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.start()
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.7)
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(0, message=message))
            assert self.is_timeout
            self.is_timeout = False
            message.text = '/start'
            message.entities[0].length = len('/start')
            await app.process_update(Update(update_id=1, message=message))
            await asyncio.sleep(0.7)
            assert handler.check_update(Update(0, message=message))
            assert self.is_timeout
            self.is_timeout = False
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/startCoding'
            message.entities[0].length = len('/startCoding')
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.7)
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(0, message=message))
            assert not self.is_timeout
            await app.stop()

    async def test_conversation_handler_timeout_state_context(self, app, bot, user1):
        states = self.states
        states.update({ConversationHandler.TIMEOUT: [CommandHandler('brew', self.passout_context), MessageHandler(~filters.Regex('oding'), self.passout2_context)]})
        handler = ConversationHandler(entry_points=self.entry_points, states=states, fallbacks=self.fallbacks, conversation_timeout=0.5)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.start()
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.7)
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(0, message=message))
            assert self.is_timeout
            self.is_timeout = False
            message.text = '/start'
            message.entities[0].length = len('/start')
            await app.process_update(Update(update_id=1, message=message))
            await asyncio.sleep(0.7)
            assert handler.check_update(Update(0, message=message))
            assert self.is_timeout
            self.is_timeout = False
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/startCoding'
            message.entities[0].length = len('/startCoding')
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.7)
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(0, message=message))
            assert not self.is_timeout
            await app.stop()

    async def test_conversation_timeout_cancel_conflict(self, app, bot, user1):

        async def slowbrew(_update, context):
            await asyncio.sleep(0.25)
            await asyncio.sleep(0.25)
        states = self.states
        states[self.THIRSTY].append(CommandHandler('slowbrew', slowbrew))
        states.update({ConversationHandler.TIMEOUT: [MessageHandler(None, self.passout2)]})
        handler = ConversationHandler(entry_points=self.entry_points, states=states, fallbacks=self.fallbacks, conversation_timeout=0.5)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.start()
            await app.process_update(Update(update_id=0, message=message))
            await asyncio.sleep(0.25)
            message.text = '/slowbrew'
            message.entities[0].length = len('/slowbrew')
            await app.process_update(Update(update_id=0, message=message))
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert not handler.check_update(Update(0, message=message))
            assert not self.is_timeout
            await asyncio.sleep(0.7)
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(0, message=message))
            assert self.is_timeout
            await app.stop()

    async def test_nested_conversation_handler(self, app, bot, user1, user2):
        self.nested_states[self.DRINKING] = [ConversationHandler(entry_points=self.drinking_entry_points, states=self.drinking_states, fallbacks=self.drinking_fallbacks, map_to_parent=self.drinking_map_to_parent)]
        handler = ConversationHandler(entry_points=self.entry_points, states=self.nested_states, fallbacks=self.fallbacks)
        app.add_handler(handler)
        message = Message(0, None, self.group, from_user=user1, text='/start', entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.THIRSTY
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.BREWING
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.DRINKING
            message.text = '/hold'
            message.entities[0].length = len('/hold')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.HOLDING
            message.text = '/sip'
            message.entities[0].length = len('/sip')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.SIPPING
            message.text = '/swallow'
            message.entities[0].length = len('/swallow')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.SWALLOWING
            message.text = '/hold'
            message.entities[0].length = len('/hold')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.HOLDING
            message.text = '/replenish'
            message.entities[0].length = len('/replenish')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.REPLENISHING
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            assert handler.check_update(Update(0, message=message))
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.DRINKING
            message.text = '/startCoding'
            message.entities[0].length = len('/startCoding')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.CODING
            message.text = '/drinkMore'
            message.entities[0].length = len('/drinkMore')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.DRINKING
            message.text = '/hold'
            message.entities[0].length = len('/hold')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.HOLDING
            message.text = '/end'
            message.entities[0].length = len('/end')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.END
            message.text = '/drinkMore'
            message.entities[0].length = len('/drinkMore')
            assert handler.check_update(Update(0, message=message))
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.DRINKING
            message.text = '/stop'
            message.entities[0].length = len('/stop')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.STOPPING
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(0, message=message))

    async def test_nested_conversation_application_handler_stop(self, app, bot, user1, user2):
        self.nested_states[self.DRINKING] = [ConversationHandler(entry_points=self.drinking_entry_points, states=self.drinking_states, fallbacks=self.drinking_fallbacks, map_to_parent=self.drinking_map_to_parent)]
        handler = ConversationHandler(entry_points=self.entry_points, states=self.nested_states, fallbacks=self.fallbacks)

        def test_callback(u, c):
            if False:
                for i in range(10):
                    print('nop')
            self.test_flag = True
        app.add_handler(handler)
        app.add_handler(TypeHandler(Update, test_callback), group=1)
        self.raise_app_handler_stop = True
        message = Message(0, None, self.group, text='/start', from_user=user1, entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len('/start'))])
        message.set_bot(bot)
        message._unfreeze()
        message.entities[0]._unfreeze()
        async with app:
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.THIRSTY
            assert not self.test_flag
            message.text = '/brew'
            message.entities[0].length = len('/brew')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.BREWING
            assert not self.test_flag
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.DRINKING
            assert not self.test_flag
            message.text = '/hold'
            message.entities[0].length = len('/hold')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.HOLDING
            assert not self.test_flag
            message.text = '/sip'
            message.entities[0].length = len('/sip')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.SIPPING
            assert not self.test_flag
            message.text = '/swallow'
            message.entities[0].length = len('/swallow')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.SWALLOWING
            assert not self.test_flag
            message.text = '/hold'
            message.entities[0].length = len('/hold')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.HOLDING
            assert not self.test_flag
            message.text = '/replenish'
            message.entities[0].length = len('/replenish')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.REPLENISHING
            message.text = '/pourCoffee'
            message.entities[0].length = len('/pourCoffee')
            assert handler.check_update(Update(0, message=message))
            assert not self.test_flag
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.DRINKING
            assert not self.test_flag
            message.text = '/startCoding'
            message.entities[0].length = len('/startCoding')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.CODING
            assert not self.test_flag
            message.text = '/drinkMore'
            message.entities[0].length = len('/drinkMore')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.DRINKING
            assert not self.test_flag
            message.text = '/hold'
            message.entities[0].length = len('/hold')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.HOLDING
            assert not self.test_flag
            message.text = '/end'
            message.entities[0].length = len('/end')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.END
            message.text = '/drinkMore'
            message.entities[0].length = len('/drinkMore')
            assert handler.check_update(Update(0, message=message))
            assert not self.test_flag
            message.text = '/drinkMore'
            message.entities[0].length = len('/drinkMore')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.DRINKING
            assert not self.test_flag
            message.text = '/stop'
            message.entities[0].length = len('/stop')
            await app.process_update(Update(update_id=0, message=message))
            assert self.current_state[user1.id] == self.STOPPING
            message.text = '/start'
            message.entities[0].length = len('/start')
            assert handler.check_update(Update(0, message=message))
            assert not self.test_flag

    @pytest.mark.parametrize('callback_raises', [True, False])
    async def test_timeout_non_block(self, app, user1, callback_raises):
        event = asyncio.Event()

        async def callback(_, __):
            await event.wait()
            if callback_raises:
                raise RuntimeError
            return 1
        conv_handler = ConversationHandler(entry_points=[MessageHandler(filters.ALL, callback=callback, block=False)], states={ConversationHandler.TIMEOUT: [TypeHandler(Update, self.passout2)]}, fallbacks=[], conversation_timeout=0.5)
        app.add_handler(conv_handler)
        async with app:
            await app.start()
            message = Message(0, None, self.group, text='/start', from_user=user1)
            assert conv_handler.check_update(Update(0, message=message))
            await app.process_update(Update(0, message=message))
            await asyncio.sleep(0.7)
            tasks = asyncio.all_tasks()
            assert any((':handle_update:non_blocking_cb' in t.get_name() for t in tasks))
            assert any((':handle_update:timeout_job' in t.get_name() for t in tasks))
            assert not self.is_timeout
            event.set()
            await asyncio.sleep(0.7)
            assert self.is_timeout == (not callback_raises)
            await app.stop()

    async def test_no_timeout_on_end(self, app, user1):
        conv_handler = ConversationHandler(entry_points=[MessageHandler(filters.ALL, callback=self.start_end)], states={ConversationHandler.TIMEOUT: [TypeHandler(Update, self.passout2)]}, fallbacks=[], conversation_timeout=0.5)
        app.add_handler(conv_handler)
        async with app:
            await app.start()
            message = Message(0, None, self.group, text='/start', from_user=user1)
            assert conv_handler.check_update(Update(0, message=message))
            await app.process_update(Update(0, message=message))
            await asyncio.sleep(0.7)
            assert not self.is_timeout
            await app.stop()

    async def test_conversation_handler_block_dont_override(self, app):
        """This just makes sure that we don't change any attributes of the handlers of the conv"""
        conv_handler = ConversationHandler(entry_points=self.entry_points, states=self.states, fallbacks=self.fallbacks, block=False)
        all_handlers = conv_handler.entry_points + conv_handler.fallbacks
        for state_handlers in conv_handler.states.values():
            all_handlers += state_handlers
        for handler in all_handlers:
            assert handler.block
        conv_handler = ConversationHandler(entry_points=[CommandHandler('start', self.start_end, block=False)], states={1: [CommandHandler('start', self.start_end, block=False)]}, fallbacks=[CommandHandler('start', self.start_end, block=False)], block=True)
        all_handlers = conv_handler.entry_points + conv_handler.fallbacks
        for state_handlers in conv_handler.states.values():
            all_handlers += state_handlers
        for handler in all_handlers:
            assert handler.block is False

    @pytest.mark.parametrize('default_block', [True, False, None])
    @pytest.mark.parametrize('ch_block', [True, False, None])
    @pytest.mark.parametrize('handler_block', [True, False, None])
    @pytest.mark.parametrize('ext_bot', [True, False], ids=['ExtBot', 'Bot'])
    async def test_blocking_resolution_order(self, bot_info, default_block, ch_block, handler_block, ext_bot):
        event = asyncio.Event()

        async def callback(_, __):
            await event.wait()
            event.clear()
            self.test_flag = True
            return 1
        if handler_block is not None:
            handler = CommandHandler('start', callback=callback, block=handler_block)
            fallback = MessageHandler(filters.ALL, callback, block=handler_block)
        else:
            handler = CommandHandler('start', callback=callback)
            fallback = MessageHandler(filters.ALL, callback, block=handler_block)
        defaults = Defaults(block=default_block) if default_block is not None else None
        if ch_block is not None:
            conv_handler = ConversationHandler(entry_points=[handler], states={1: [handler]}, fallbacks=[fallback], block=ch_block)
        else:
            conv_handler = ConversationHandler(entry_points=[handler], states={1: [handler]}, fallbacks=[fallback])
        bot = make_bot(bot_info, defaults=defaults) if ext_bot else PytestBot(bot_info['token'])
        app = ApplicationBuilder().bot(bot).build()
        app.add_handler(conv_handler)
        async with app:
            start_message = make_command_message('/start')
            start_message.set_bot(bot)
            fallback_message = make_command_message('/fallback')
            fallback_message.set_bot(bot)
            for message in [start_message, fallback_message]:
                process_update_task = asyncio.create_task(app.process_update(Update(0, message=message)))
                if handler_block is False or (handler_block is None and ch_block is False) or (handler_block is None and ch_block is None and ext_bot and (default_block is False)):
                    await asyncio.sleep(0.01)
                    assert process_update_task.done()
                else:
                    assert not process_update_task.done()
                assert not self.test_flag
                event.set()
                await asyncio.sleep(0.01)
                assert process_update_task.done()
                assert self.test_flag
                self.test_flag = False

    async def test_waiting_state(self, app, user1):
        event = asyncio.Event()

        async def callback_1(_, __):
            self.test_flag = 1

        async def callback_2(_, __):
            self.test_flag = 2

        async def callback_3(_, __):
            self.test_flag = 3

        async def blocking(_, __):
            await event.wait()
            return 1
        conv_handler = ConversationHandler(entry_points=[MessageHandler(filters.ALL, callback=blocking, block=False)], states={ConversationHandler.WAITING: [MessageHandler(filters.Regex('1'), callback_1), MessageHandler(filters.Regex('2'), callback_2)], 1: [MessageHandler(filters.Regex('2'), callback_3)]}, fallbacks=[])
        app.add_handler(conv_handler)
        message = Message(0, None, self.group, text='/start', from_user=user1)
        message._unfreeze()
        async with app:
            await app.process_update(Update(0, message=message))
            assert not self.test_flag
            message.text = '1'
            await app.process_update(Update(0, message=message))
            assert self.test_flag == 1
            message.text = '2'
            await app.process_update(Update(0, message=message))
            assert self.test_flag == 2
            event.set()
            await asyncio.sleep(0.05)
            self.test_flag = None
            await app.process_update(Update(0, message=message))
            assert self.test_flag == 3