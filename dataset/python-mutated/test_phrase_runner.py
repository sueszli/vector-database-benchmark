from unittest.mock import MagicMock, patch
import pytest
from hamcrest import *
import autokey.service
import autokey.model.key
from autokey.service import PhraseRunner
from autokey.model.phrase import Phrase

def _create_phrase_runner(phrase_content: str) -> PhraseRunner:
    if False:
        for i in range(10):
            print('nop')
    mock_service = MagicMock()
    runner = PhraseRunner(mock_service)
    with patch.object(runner, 'execute', new=runner.execute._original):
        setattr(runner, 'execute', runner.execute._original.__get__(runner, PhraseRunner))
        runner.execute(_generate_phrase(phrase_content))
    return runner

def _generate_phrase(content: str) -> Phrase:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate a Phrase instance with the given content.\n    '
    phrase = Phrase('description', content)
    phrase.parent = MagicMock()
    return phrase

def generate_test_cases_for_test_can_undo_expansion():
    if False:
        return 10
    yield ('', True)
    yield ('abc', True)
    yield ('<code>', True)
    yield ('abc<code>12', True)
    yield ('<code1A>', True)
    yield ('<left>', False)
    yield ('<shift>+<left>', False)
    yield ('<code50>', False)
    yield ('abc<ALT>12', False)
    yield ('<Ctrl>', False)
    yield ('<abc<up>>', False)
    for key in autokey.model.key.Key:
        yield (key, False)

@pytest.mark.parametrize('content, expected', generate_test_cases_for_test_can_undo_expansion())
def test_can_undo_expansion(content: str, expected: bool):
    if False:
        for i in range(10):
            print('nop')
    runner = _create_phrase_runner(content)
    assert_that(runner.lastPhrase, is_(not_none()), 'Test setup failed. The PhraseRunner holds no Phrase.')
    assert_that(runner.lastPhrase.phrase, is_(equal_to(content)), 'Test setup failed. The PhraseRunner holds an unexpected Phrase.')
    assert_that(runner.can_undo(), is_(equal_to(expected)), 'can_undo() returned wrong result')