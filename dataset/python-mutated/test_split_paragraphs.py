from parameterized import parameterized
from tests.support.help_reformatting import split_paragraphs

@parameterized.expand([('one line', ['one line']), ('one line\n', ['one line\n']), ('one\ntwo\n', ['one\ntwo\n']), ('one\n\ntwo\n', ['one\n', 'two\n']), ('one\n    \ntwo\n', ['one\n', 'two\n'])])
def test_split_paragraphs(text, expected_result):
    if False:
        i = 10
        return i + 15
    assert split_paragraphs(text) == expected_result