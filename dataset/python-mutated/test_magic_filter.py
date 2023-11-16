from dataclasses import dataclass
from re import Match
from aiogram import F
from aiogram.utils.magic_filter import MagicFilter

@dataclass
class MyObject:
    text: str

class TestMagicFilter:

    def test_operation_as(self):
        if False:
            i = 10
            return i + 15
        magic: MagicFilter = F.text.regexp('^(\\d+)$').as_('match')
        assert not magic.resolve(MyObject(text='test'))
        result = magic.resolve(MyObject(text='123'))
        assert isinstance(result, dict)
        assert isinstance(result['match'], Match)

    def test_operation_as_not_none(self):
        if False:
            i = 10
            return i + 15
        magic = F.cast(int).as_('value')
        result = magic.resolve('0')
        assert result == {'value': 0}

    def test_operation_as_not_none_iterable(self):
        if False:
            return 10
        magic = F.as_('value')
        result = magic.resolve([])
        assert result is None