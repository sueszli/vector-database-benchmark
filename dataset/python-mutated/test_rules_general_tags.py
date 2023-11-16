import pytest
from tribler.core.components.knowledge.rules.rules_general_tags import delimiter_re, extension_re, general_rules, parentheses_re, square_brackets_re
from tribler.core.components.knowledge.rules.tag_rules_base import extract_tags
DELIMITERS = [('word1 word2 word3', ['word1', 'word2', 'word3']), ('word1,word2,word3', ['word1', 'word2', 'word3']), ('word1/word2/word3', ['word1', 'word2', 'word3']), ('word1|word2|word3', ['word1', 'word2', 'word3']), ('word1 /.,word2', ['word1', 'word2'])]
SQUARE_BRACKETS = [('[word1] [word2 word3]', ['word1', 'word2 word3']), ('[word1 [word2] word3]', ['word2'])]
PARENTHESES = [('(word1) (word2 word3)', ['word1', 'word2 word3']), ('(word1 (word2) word3)', ['word2'])]
EXTENSIONS = [('some.ext', ['ext']), ('some.ext4', ['ext4']), ('some', []), ('some. ext', []), ('some.ext ', [])]

@pytest.mark.parametrize('text, words', DELIMITERS)
def test_delimiter(text, words):
    if False:
        while True:
            i = 10
    assert delimiter_re.findall(text) == words

@pytest.mark.parametrize('text, words', SQUARE_BRACKETS)
def test_square_brackets(text, words):
    if False:
        while True:
            i = 10
    assert square_brackets_re.findall(text) == words

@pytest.mark.parametrize('text, words', PARENTHESES)
def test_parentheses(text, words):
    if False:
        i = 10
        return i + 15
    assert parentheses_re.findall(text) == words

@pytest.mark.parametrize('text, words', EXTENSIONS)
def test_extension(text, words):
    if False:
        print('Hello World!')
    assert extension_re.findall(text) == words

def test_tags_in_square_brackets():
    if False:
        for i in range(10):
            print('nop')
    text = 'text [tag1, tag2] text1 [tag3|tag4] text2'
    expected_tags = {'tag1', 'tag2', 'tag3', 'tag4'}
    actual_tags = set(extract_tags(text, rules=general_rules))
    assert actual_tags == expected_tags

def test_tags_in_parentheses():
    if False:
        for i in range(10):
            print('nop')
    text = 'text (tag1, tag2) text1 (tag3|tag4) text2'
    expected_tags = {'tag1', 'tag2', 'tag3', 'tag4'}
    actual_tags = set(extract_tags(text, rules=general_rules))
    assert actual_tags == expected_tags

def test_general_rules():
    if False:
        print('Hello World!')
    text = 'text (tag1, tag2) text1 (tag3|tag4) text2, [tag5, tag6].ext'
    expected_tags = {'tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6', 'ext'}
    actual_tags = set(extract_tags(text, rules=general_rules))
    assert actual_tags == expected_tags