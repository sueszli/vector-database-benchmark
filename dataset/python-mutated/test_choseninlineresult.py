import pytest
from telegram import ChosenInlineResult, Location, User, Voice
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def user():
    if False:
        for i in range(10):
            print('nop')
    user = User(1, 'First name', False)
    user._unfreeze()
    return user

@pytest.fixture(scope='module')
def chosen_inline_result(user):
    if False:
        while True:
            i = 10
    return ChosenInlineResult(TestChosenInlineResultBase.result_id, user, TestChosenInlineResultBase.query)

class TestChosenInlineResultBase:
    result_id = 'result id'
    query = 'query text'

class TestChosenInlineResultWithoutRequest(TestChosenInlineResultBase):

    def test_slot_behaviour(self, chosen_inline_result):
        if False:
            print('Hello World!')
        inst = chosen_inline_result
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_de_json_required(self, bot, user):
        if False:
            return 10
        json_dict = {'result_id': self.result_id, 'from': user.to_dict(), 'query': self.query}
        result = ChosenInlineResult.de_json(json_dict, bot)
        assert result.api_kwargs == {}
        assert result.result_id == self.result_id
        assert result.from_user == user
        assert result.query == self.query

    def test_de_json_all(self, bot, user):
        if False:
            while True:
                i = 10
        loc = Location(-42.003, 34.004)
        json_dict = {'result_id': self.result_id, 'from': user.to_dict(), 'query': self.query, 'location': loc.to_dict(), 'inline_message_id': 'a random id'}
        result = ChosenInlineResult.de_json(json_dict, bot)
        assert result.api_kwargs == {}
        assert result.result_id == self.result_id
        assert result.from_user == user
        assert result.query == self.query
        assert result.location == loc
        assert result.inline_message_id == 'a random id'

    def test_to_dict(self, chosen_inline_result):
        if False:
            return 10
        chosen_inline_result_dict = chosen_inline_result.to_dict()
        assert isinstance(chosen_inline_result_dict, dict)
        assert chosen_inline_result_dict['result_id'] == chosen_inline_result.result_id
        assert chosen_inline_result_dict['from'] == chosen_inline_result.from_user.to_dict()
        assert chosen_inline_result_dict['query'] == chosen_inline_result.query

    def test_equality(self, user):
        if False:
            i = 10
            return i + 15
        a = ChosenInlineResult(self.result_id, user, 'Query', '')
        b = ChosenInlineResult(self.result_id, user, 'Query', '')
        c = ChosenInlineResult(self.result_id, user, '', '')
        d = ChosenInlineResult('', user, 'Query', '')
        e = Voice(self.result_id, 'unique_id', 0)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a == c
        assert hash(a) == hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)