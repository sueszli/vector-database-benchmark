import datetime
import inspect
import pickle
import re
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
import pytest
from telegram import Bot, BotCommand, Chat, Message, PhotoSize, TelegramObject, User
from telegram.ext import PicklePersistence
from telegram.warnings import PTBUserWarning
from tests.auxil.files import data_file
from tests.auxil.slots import mro_slots

def all_subclasses(cls):
    if False:
        return 10
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)]).union({cls})
TO_SUBCLASSES = sorted(all_subclasses(TelegramObject), key=lambda cls: cls.__name__)

class TestTelegramObject:

    class Sub(TelegramObject):

        def __init__(self, private, normal, b):
            if False:
                return 10
            super().__init__()
            self._private = private
            self.normal = normal
            self._bot = b

    class ChangingTO(TelegramObject):
        pass

    def test_to_json(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')

        class Subclass(TelegramObject):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.arg = 'arg'
                self.arg2 = ['arg2', 'arg2']
                self.arg3 = {'arg3': 'arg3'}
                self.empty_tuple = ()
        json = Subclass().to_json()
        assert '"arg": "arg"' in json
        assert '"arg2": ["arg2", "arg2"]' in json
        assert '"arg3": {"arg3": "arg3"}' in json
        assert 'empty_tuple' not in json
        d = {('str', 'str'): 'str'}
        monkeypatch.setattr('telegram.TelegramObject.to_dict', lambda _: d)
        with pytest.raises(TypeError):
            TelegramObject().to_json()

    def test_de_json_api_kwargs(self, bot):
        if False:
            while True:
                i = 10
        to = TelegramObject.de_json(data={'foo': 'bar'}, bot=bot)
        assert to.api_kwargs == {'foo': 'bar'}
        assert to.get_bot() is bot

    def test_de_list(self, bot):
        if False:
            print('Hello World!')

        class SubClass(TelegramObject):

            def __init__(self, arg: int, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(**kwargs)
                self.arg = arg
                self._id_attrs = (self.arg,)
        assert SubClass.de_list([{'arg': 1}, None, {'arg': 2}, None], bot) == (SubClass(1), SubClass(2))

    def test_api_kwargs_read_only(self):
        if False:
            return 10
        tg_object = TelegramObject(api_kwargs={'foo': 'bar'})
        tg_object._freeze()
        assert isinstance(tg_object.api_kwargs, MappingProxyType)
        with pytest.raises(TypeError):
            tg_object.api_kwargs['foo'] = 'baz'
        with pytest.raises(AttributeError, match="can't be set"):
            tg_object.api_kwargs = {'foo': 'baz'}

    @pytest.mark.parametrize('cls', TO_SUBCLASSES, ids=[cls.__name__ for cls in TO_SUBCLASSES])
    def test_subclasses_have_api_kwargs(self, cls):
        if False:
            print('Hello World!')
        'Checks that all subclasses of TelegramObject have an api_kwargs argument that is\n        kw-only. Also, tries to check that this argument is passed to super - by checking that\n        the `__init__` contains `api_kwargs=api_kwargs`\n        '
        if issubclass(cls, Bot):
            return
        if inspect.getsourcefile(cls.__init__) != inspect.getsourcefile(cls):
            return
        source_file = Path(inspect.getsourcefile(cls))
        parents = source_file.parents
        is_test_file = Path(__file__).parent.resolve() in parents
        if is_test_file:
            return
        signature = inspect.signature(cls)
        assert signature.parameters.get('api_kwargs').kind == inspect.Parameter.KEYWORD_ONLY
        if cls is TelegramObject:
            return
        assert 'api_kwargs=api_kwargs' in inspect.getsource(cls.__init__), f"{cls.__name__} doesn't seem to pass `api_kwargs` to `super().__init__`"

    def test_de_json_arbitrary_exceptions(self, bot):
        if False:
            for i in range(10):
                print('nop')

        class SubClass(TelegramObject):

            def __init__(self, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__(**kwargs)
                raise TypeError('This is a test')
        with pytest.raises(TypeError, match='This is a test'):
            SubClass.de_json({}, bot)

    def test_to_dict_private_attribute(self):
        if False:
            while True:
                i = 10

        class TelegramObjectSubclass(TelegramObject):
            __slots__ = ('a', '_b')

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.a = 1
                self._b = 2
        subclass_instance = TelegramObjectSubclass()
        assert subclass_instance.to_dict() == {'a': 1}

    def test_to_dict_api_kwargs(self):
        if False:
            print('Hello World!')
        to = TelegramObject(api_kwargs={'foo': 'bar'})
        assert to.to_dict() == {'foo': 'bar'}

    def test_to_dict_missing_attribute(self):
        if False:
            return 10
        message = Message(1, datetime.datetime.now(), Chat(1, 'private'), from_user=User(1, '', False))
        message._unfreeze()
        del message.chat
        message_dict = message.to_dict()
        assert 'chat' not in message_dict
        message_dict = message.to_dict(recursive=False)
        assert message_dict['chat'] is None

    def test_to_dict_recursion(self):
        if False:
            i = 10
            return i + 15

        class Recursive(TelegramObject):
            __slots__ = ('recursive',)

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.recursive = 'recursive'

        class SubClass(TelegramObject):
            """This class doesn't have `__slots__`, so has `__dict__` instead."""

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.subclass = Recursive()
        to = SubClass()
        to_dict_no_recurse = to.to_dict(recursive=False)
        assert to_dict_no_recurse
        assert isinstance(to_dict_no_recurse['subclass'], Recursive)
        to_dict_recurse = to.to_dict(recursive=True)
        assert to_dict_recurse
        assert isinstance(to_dict_recurse['subclass'], dict)
        assert to_dict_recurse['subclass']['recursive'] == 'recursive'

    def test_slot_behaviour(self):
        if False:
            while True:
                i = 10
        inst = TelegramObject()
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_meaningless_comparison(self, recwarn):
        if False:
            return 10
        expected_warning = 'Objects of type TGO can not be meaningfully tested for equivalence.'

        class TGO(TelegramObject):
            pass
        a = TGO()
        b = TGO()
        assert a == b
        assert len(recwarn) == 1
        assert str(recwarn[0].message) == expected_warning
        assert recwarn[0].category is PTBUserWarning
        assert recwarn[0].filename == __file__, 'wrong stacklevel'

    def test_meaningful_comparison(self, recwarn):
        if False:
            for i in range(10):
                print('nop')

        class TGO(TelegramObject):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self._id_attrs = (1,)
        a = TGO()
        b = TGO()
        assert a == b
        assert len(recwarn) == 0
        assert b == a
        assert len(recwarn) == 0

    def test_bot_instance_none(self):
        if False:
            print('Hello World!')
        tg_object = TelegramObject()
        with pytest.raises(RuntimeError):
            tg_object.get_bot()

    @pytest.mark.parametrize('bot_inst', ['bot', None])
    def test_bot_instance_states(self, bot_inst):
        if False:
            return 10
        tg_object = TelegramObject()
        tg_object.set_bot('bot' if bot_inst == 'bot' else bot_inst)
        if bot_inst == 'bot':
            assert tg_object.get_bot() == 'bot'
        elif bot_inst is None:
            with pytest.raises(RuntimeError):
                tg_object.get_bot()

    def test_subscription(self):
        if False:
            while True:
                i = 10
        chat = Chat(2, Chat.PRIVATE)
        user = User(3, 'first_name', False)
        message = Message(1, None, chat=chat, from_user=user, text='foobar')
        assert message['text'] == 'foobar'
        assert message['chat'] is chat
        assert message['chat_id'] == 2
        assert message['from'] is user
        assert message['from_user'] is user
        with pytest.raises(KeyError, match="Message don't have an attribute called `no_key`"):
            message['no_key']

    def test_pickle(self, bot):
        if False:
            return 10
        chat = Chat(2, Chat.PRIVATE)
        user = User(3, 'first_name', False)
        date = datetime.datetime.now()
        photo = PhotoSize('file_id', 'unique', 21, 21)
        photo.set_bot(bot)
        msg = Message(1, date, chat, from_user=user, text='foobar', photo=[photo], api_kwargs={'api': 'kwargs'})
        msg.set_bot(bot)
        assert msg.get_bot()
        unpickled = pickle.loads(pickle.dumps(msg))
        with pytest.raises(RuntimeError):
            unpickled.get_bot()
        assert unpickled.chat == chat, f'{unpickled.chat._id_attrs} != {chat._id_attrs}'
        assert unpickled.from_user == user
        assert unpickled.date == date, f'{unpickled.date} != {date}'
        assert unpickled.photo[0] == photo
        assert isinstance(unpickled.api_kwargs, MappingProxyType)
        assert unpickled.api_kwargs == {'api': 'kwargs'}

    def test_pickle_apply_api_kwargs(self):
        if False:
            while True:
                i = 10
        'Makes sure that when a class gets new attributes, the api_kwargs are moved to the\n        new attributes on unpickling.'
        obj = self.ChangingTO(api_kwargs={'foo': 'bar'})
        pickled = pickle.dumps(obj)
        self.ChangingTO.foo = None
        obj = pickle.loads(pickled)
        assert obj.foo == 'bar'
        assert obj.api_kwargs == {}

    async def test_pickle_backwards_compatibility(self):
        """Test when newer versions of the library remove or add attributes from classes (which
        the old pickled versions still/don't have).
        """
        pp = PicklePersistence(data_file('20a5_modified_chat.pickle'))
        chat = (await pp.get_chat_data())[1]
        assert chat.id == 1
        assert chat.type == Chat.PRIVATE
        assert chat.api_kwargs == {'all_members_are_administrators': True, 'something': 'Manually inserted'}
        with pytest.raises(AttributeError):
            chat.all_members_are_administrators
        with pytest.raises(AttributeError):
            chat.is_forum
        chat.id = 7
        assert chat.id == 7

    def test_deepcopy_telegram_obj(self, bot):
        if False:
            while True:
                i = 10
        chat = Chat(2, Chat.PRIVATE)
        user = User(3, 'first_name', False)
        date = datetime.datetime.now()
        photo = PhotoSize('file_id', 'unique', 21, 21)
        photo.set_bot(bot)
        msg = Message(1, date, chat, from_user=user, text='foobar', photo=[photo], api_kwargs={'foo': 'bar'})
        msg.set_bot(bot)
        new_msg = deepcopy(msg)
        assert new_msg == msg
        assert new_msg is not msg
        assert new_msg.get_bot() == bot
        assert new_msg.get_bot() is bot
        assert new_msg.date == date
        assert new_msg.date is not date
        assert new_msg.chat == chat
        assert new_msg.chat is not chat
        assert new_msg.from_user == user
        assert new_msg.from_user is not user
        assert new_msg.photo[0] == photo
        assert new_msg.photo[0] is not photo
        assert new_msg.api_kwargs == {'foo': 'bar'}
        assert new_msg.api_kwargs is not msg.api_kwargs
        with pytest.raises(AttributeError, match="Attribute `text` of class `Message` can't be set!"):
            new_msg.text = 'new text'
        msg._unfreeze()
        new_message = deepcopy(msg)
        new_message.text = 'new text'
        assert new_message.text == 'new text'

    def test_deepcopy_subclass_telegram_obj(self, bot):
        if False:
            while True:
                i = 10
        s = self.Sub('private', 'normal', bot)
        d = deepcopy(s)
        assert d is not s
        assert d._private == s._private
        assert d._bot == s._bot
        assert d._bot is s._bot
        assert d.normal == s.normal

    def test_string_representation(self):
        if False:
            return 10

        class TGO(TelegramObject):

            def __init__(self, api_kwargs=None):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(api_kwargs=api_kwargs)
                self.string_attr = 'string'
                self.int_attr = 42
                self.to_attr = BotCommand('command', 'description')
                self.list_attr = [BotCommand('command_1', 'description_1'), BotCommand('command_2', 'description_2')]
                self.dict_attr = {BotCommand('command_1', 'description_1'): BotCommand('command_2', 'description_2')}
                self.empty_tuple_attrs = ()
                self.empty_str_attribute = ''
                self.none_attr = None
        expected_without_api_kwargs = "TGO(dict_attr={BotCommand(command='command_1', description='description_1'): BotCommand(command='command_2', description='description_2')}, int_attr=42, list_attr=[BotCommand(command='command_1', description='description_1'), BotCommand(command='command_2', description='description_2')], string_attr='string', to_attr=BotCommand(command='command', description='description'))"
        assert str(TGO()) == expected_without_api_kwargs
        assert repr(TGO()) == expected_without_api_kwargs
        expected_with_api_kwargs = "TGO(api_kwargs={'foo': 'bar'}, dict_attr={BotCommand(command='command_1', description='description_1'): BotCommand(command='command_2', description='description_2')}, int_attr=42, list_attr=[BotCommand(command='command_1', description='description_1'), BotCommand(command='command_2', description='description_2')], string_attr='string', to_attr=BotCommand(command='command', description='description'))"
        assert str(TGO(api_kwargs={'foo': 'bar'})) == expected_with_api_kwargs
        assert repr(TGO(api_kwargs={'foo': 'bar'})) == expected_with_api_kwargs

    @pytest.mark.parametrize('cls', TO_SUBCLASSES, ids=[cls.__name__ for cls in TO_SUBCLASSES])
    def test_subclasses_are_frozen(self, cls):
        if False:
            for i in range(10):
                print('nop')
        if cls is TelegramObject or cls.__name__.startswith('_'):
            return
        source_file = inspect.getsourcefile(cls.__init__)
        parents = Path(source_file).parents
        is_test_file = Path(__file__).parent.resolve() in parents
        if is_test_file:
            return
        if source_file.endswith('telegramobject.py'):
            pytest.fail(f'{cls.__name__} does not have its own `__init__` and can therefore not be frozen correctly')
        (source_lines, first_line) = inspect.getsourcelines(cls.__init__)
        last_line_freezes = re.match('\\s*self\\.\\_freeze\\(\\)', source_lines[-1])
        uses_with_unfrozen = re.search('\\n\\s*with self\\.\\_unfrozen\\(\\)\\:', inspect.getsource(cls.__init__))
        assert last_line_freezes or uses_with_unfrozen, f'{cls.__name__} is not frozen correctly'

    def test_freeze_unfreeze(self):
        if False:
            i = 10
            return i + 15

        class TestSub(TelegramObject):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self._protected = True
                self.public = True
                self._freeze()
        foo = TestSub()
        foo._protected = False
        assert foo._protected is False
        with pytest.raises(AttributeError, match="Attribute `public` of class `TestSub` can't be set!"):
            foo.public = False
        with pytest.raises(AttributeError, match="Attribute `public` of class `TestSub` can't be deleted!"):
            del foo.public
        foo._unfreeze()
        foo._protected = True
        assert foo._protected is True
        foo.public = False
        assert foo.public is False
        del foo.public
        del foo._protected
        assert not hasattr(foo, 'public')
        assert not hasattr(foo, '_protected')