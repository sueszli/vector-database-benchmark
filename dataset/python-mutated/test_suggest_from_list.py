from __future__ import annotations
import pytest
from textual.dom import DOMNode
from textual.suggester import SuggestFromList, SuggestionReady
countries = ['England', 'Portugal', 'Scotland', 'portugal', 'PORTUGAL']

class LogListNode(DOMNode):

    def __init__(self, log_list: list[tuple[str, str]]) -> None:
        if False:
            return 10
        self.log_list = log_list

    def post_message(self, message: SuggestionReady):
        if False:
            i = 10
            return i + 15
        self.log_list.append((message.suggestion, message.value))

async def test_first_suggestion_has_priority():
    suggester = SuggestFromList(countries)
    assert 'Portugal' == await suggester.get_suggestion('P')

@pytest.mark.parametrize('value', ['s', 'S', 'sc', 'sC', 'Sc', 'SC'])
async def test_case_insensitive_suggestions(value):
    suggester = SuggestFromList(countries, case_sensitive=False)
    log = []
    await suggester._get_suggestion(LogListNode(log), value)
    assert log == [('Scotland', value)]

@pytest.mark.parametrize('value', ['p', 'P', 'po', 'Po', 'pO', 'PO', 'port', 'Port', 'pORT', 'PORT'])
async def test_first_suggestion_has_priority_case_insensitive(value):
    suggester = SuggestFromList(countries, case_sensitive=False)
    log = []
    await suggester._get_suggestion(LogListNode(log), value)
    assert log == [('Portugal', value)]