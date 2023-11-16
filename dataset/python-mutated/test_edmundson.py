from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from sumy._compat import to_unicode
from sumy.summarizers.edmundson import EdmundsonSummarizer
from ..utils import build_document, build_document_from_string

def test_bonus_words_property():
    if False:
        return 10
    summarizer = EdmundsonSummarizer()
    assert summarizer.bonus_words == frozenset()
    words = ('word', 'another', 'and', 'some', 'next')
    summarizer.bonus_words = words
    assert summarizer.bonus_words == frozenset(words)

def test_stigma_words_property():
    if False:
        for i in range(10):
            print('nop')
    summarizer = EdmundsonSummarizer()
    assert summarizer.stigma_words == frozenset()
    words = ('word', 'another', 'and', 'some', 'next')
    summarizer.stigma_words = words
    assert summarizer.stigma_words == frozenset(words)

def test_null_words_property():
    if False:
        i = 10
        return i + 15
    summarizer = EdmundsonSummarizer()
    assert summarizer.null_words == frozenset()
    words = ('word', 'another', 'and', 'some', 'next')
    summarizer.null_words = words
    assert summarizer.null_words == frozenset(words)

def test_empty_document():
    if False:
        return 10
    summarizer = EdmundsonSummarizer(cue_weight=0, key_weight=0, title_weight=0, location_weight=0)
    sentences = summarizer(build_document(), 10)
    assert len(sentences) == 0

def test_mixed_cue_key():
    if False:
        for i in range(10):
            print('nop')
    document = build_document_from_string('\n        # This is cool heading\n        Because I am sentence I like words\n        And because I am string I like characters\n\n        # blank and heading\n        This is next paragraph because of blank line above\n        Here is the winner because contains words like cool and heading\n    ')
    summarizer = EdmundsonSummarizer(cue_weight=1, key_weight=1, title_weight=0, location_weight=0)
    summarizer.bonus_words = ('cool', 'heading', 'sentence', 'words', 'like', 'because')
    summarizer.stigma_words = ('this', 'is', 'I', 'am', 'and')
    sentences = summarizer(document, 2)
    assert list(map(to_unicode, sentences)) == ['Because I am sentence I like words', 'Here is the winner because contains words like cool and heading']

def test_cue_with_no_words():
    if False:
        print('Hello World!')
    summarizer = EdmundsonSummarizer()
    with pytest.raises(ValueError):
        summarizer.cue_method(build_document(), 10)

def test_cue_with_no_stigma_words():
    if False:
        for i in range(10):
            print('nop')
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('great', 'very', 'beautiful')
    with pytest.raises(ValueError):
        summarizer.cue_method(build_document(), 10)

def test_cue_with_no_bonus_words():
    if False:
        print('Hello World!')
    summarizer = EdmundsonSummarizer()
    summarizer.stigma_words = ('useless', 'bad', 'spinach')
    with pytest.raises(ValueError):
        summarizer.cue_method(build_document(), 10)

def test_cue_empty():
    if False:
        i = 10
        return i + 15
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('ba', 'bb', 'bc')
    summarizer.stigma_words = ('sa', 'sb', 'sc')
    sentences = summarizer.cue_method(build_document(), 10)
    assert list(map(to_unicode, sentences)) == []

def test_cue_letters_case():
    if False:
        i = 10
        return i + 15
    document = build_document(('X X X', 'x x x x'), ('w w w', 'W W W W'))
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('X', 'w')
    summarizer.stigma_words = ('stigma',)
    sentences = summarizer.cue_method(document, 2)
    assert list(map(to_unicode, sentences)) == ['x x x x', 'W W W W']

def test_cue_1():
    if False:
        i = 10
        return i + 15
    document = build_document(('ba bb bc bb unknown ľščťžýáíé sb sc sb',))
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('ba', 'bb', 'bc')
    summarizer.stigma_words = ('sa', 'sb', 'sc')
    sentences = summarizer.cue_method(document, 10)
    assert len(sentences) == 1

def test_cue_2():
    if False:
        return 10
    document = build_document(('ba bb bc bb unknown ľščťžýáíé sb sc sb',), ('Pepek likes spinach',))
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('ba', 'bb', 'bc')
    summarizer.stigma_words = ('sa', 'sb', 'sc')
    sentences = summarizer.cue_method(document, 10)
    assert list(map(to_unicode, sentences)) == ['ba bb bc bb unknown ľščťžýáíé sb sc sb', 'Pepek likes spinach']
    sentences = summarizer.cue_method(document, 1)
    assert list(map(to_unicode, sentences)) == ['ba bb bc bb unknown ľščťžýáíé sb sc sb']

def test_cue_3():
    if False:
        i = 10
        return i + 15
    document = build_document(('ba ' * 10, 'bb ' * 10, ' sa' * 8 + ' bb' * 10, 'bb bc ba'), (), ('babbbc ' * 10, 'na nb nc nd sa' + ' bc' * 10, ' ba n' * 10))
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('ba', 'bb', 'bc')
    summarizer.stigma_words = ('sa', 'sb', 'sc')
    sentences = summarizer.cue_method(document, 5)
    assert list(map(to_unicode, sentences)) == [('ba ' * 10).strip(), ('bb ' * 10).strip(), 'bb bc ba', 'na nb nc nd sa bc bc bc bc bc bc bc bc bc bc', ('ba n ' * 10).strip()]

def test_key_empty():
    if False:
        for i in range(10):
            print('nop')
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('ba', 'bb', 'bc')
    sentences = summarizer.key_method(build_document(), 10)
    assert list(map(to_unicode, sentences)) == []

def test_key_without_bonus_words():
    if False:
        i = 10
        return i + 15
    summarizer = EdmundsonSummarizer()
    with pytest.raises(ValueError):
        summarizer.key_method(build_document(), 10)

def test_key_no_bonus_words_in_document():
    if False:
        return 10
    document = build_document(('wa wb wc wd', 'I like music'), ('This is test sentence with some extra words',))
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('ba', 'bb', 'bc', 'bonus')
    sentences = summarizer.key_method(document, 10)
    assert list(map(to_unicode, sentences)) == ['wa wb wc wd', 'I like music', 'This is test sentence with some extra words']

def test_key_1():
    if False:
        while True:
            i = 10
    document = build_document(('wa wb wc wd', 'I like music'), ('This is test sentence with some extra words and bonus',))
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('ba', 'bb', 'bc', 'bonus')
    sentences = summarizer.key_method(document, 1)
    assert list(map(to_unicode, sentences)) == ['This is test sentence with some extra words and bonus']

def test_key_2():
    if False:
        print('Hello World!')
    document = build_document(('Om nom nom nom nom', 'Sure I summarize it, with bonus'), ('This is bonus test sentence with some extra words and bonus',))
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('nom', 'bonus')
    sentences = summarizer.key_method(document, 2)
    assert list(map(to_unicode, sentences)) == ['Om nom nom nom nom', 'This is bonus test sentence with some extra words and bonus']

def test_key_3():
    if False:
        while True:
            i = 10
    document = build_document(('wa', 'wa wa', 'wa wa wa', 'wa wa wa wa', 'wa Wa Wa Wa wa'), ('x X x X',))
    summarizer = EdmundsonSummarizer()
    summarizer.bonus_words = ('wa', 'X')
    sentences = summarizer.key_method(document, 3)
    assert list(map(to_unicode, sentences)) == ['wa wa wa', 'wa wa wa wa', 'wa Wa Wa Wa wa']
    sentences = summarizer.key_method(document, 3, weight=0)
    assert list(map(to_unicode, sentences)) == ['wa wa wa wa', 'wa Wa Wa Wa wa', 'x X x X']

def test_title_method_with_empty_document():
    if False:
        for i in range(10):
            print('nop')
    summarizer = EdmundsonSummarizer()
    summarizer.null_words = ('ba', 'bb', 'bc')
    sentences = summarizer.title_method(build_document(), 10)
    assert list(map(to_unicode, sentences)) == []

def test_title_method_without_null_words():
    if False:
        for i in range(10):
            print('nop')
    summarizer = EdmundsonSummarizer()
    with pytest.raises(ValueError):
        summarizer.title_method(build_document(), 10)

def test_title_method_without_title():
    if False:
        for i in range(10):
            print('nop')
    document = build_document(('This is sentence', 'This is another one'), ('And some next sentence but no heading',))
    summarizer = EdmundsonSummarizer()
    summarizer.null_words = ('this', 'is', 'some', 'and')
    sentences = summarizer.title_method(document, 10)
    assert list(map(to_unicode, sentences)) == ['This is sentence', 'This is another one', 'And some next sentence but no heading']

def test_title_method_1():
    if False:
        i = 10
        return i + 15
    document = build_document_from_string('\n        # This is cool heading\n        Because I am sentence I like words\n        And because I am string I like characters\n\n        # blank and heading\n        This is next paragraph because of blank line above\n        Here is the winner because contains words like cool and heading\n    ')
    summarizer = EdmundsonSummarizer()
    summarizer.null_words = ('this', 'is', 'I', 'am', 'and')
    sentences = summarizer.title_method(document, 1)
    assert list(map(to_unicode, sentences)) == ['Here is the winner because contains words like cool and heading']

def test_title_method_2():
    if False:
        while True:
            i = 10
    document = build_document_from_string('\n        # This is cool heading\n        Because I am sentence I like words\n        And because I am string I like characters\n\n        # blank and heading\n        This is next paragraph because of blank line above\n        Here is the winner because contains words like cool and heading\n    ')
    summarizer = EdmundsonSummarizer()
    summarizer.null_words = ('this', 'is', 'I', 'am', 'and')
    sentences = summarizer.title_method(document, 2)
    assert list(map(to_unicode, sentences)) == ['This is next paragraph because of blank line above', 'Here is the winner because contains words like cool and heading']

def test_title_method_3():
    if False:
        i = 10
        return i + 15
    document = build_document_from_string('\n        # This is cool heading\n        Because I am sentence I like words\n        And because I am string I like characters\n\n        # blank and heading\n        This is next paragraph because of blank line above\n        Here is the winner because contains words like cool and heading\n    ')
    summarizer = EdmundsonSummarizer()
    summarizer.null_words = ('this', 'is', 'I', 'am', 'and')
    sentences = summarizer.title_method(document, 3)
    assert list(map(to_unicode, sentences)) == ['Because I am sentence I like words', 'This is next paragraph because of blank line above', 'Here is the winner because contains words like cool and heading']

def test_location_method_with_empty_document():
    if False:
        i = 10
        return i + 15
    summarizer = EdmundsonSummarizer()
    summarizer.null_words = ('na', 'nb', 'nc')
    sentences = summarizer.location_method(build_document(), 10)
    assert list(map(to_unicode, sentences)) == []

def test_location_method_without_null_words():
    if False:
        i = 10
        return i + 15
    summarizer = EdmundsonSummarizer()
    with pytest.raises(ValueError):
        summarizer.location_method(build_document(), 10)

def test_location_method_1():
    if False:
        i = 10
        return i + 15
    document = build_document_from_string('\n        # na nb nc ha hb\n        ha = 1 + 1 + 1 = 3\n        ha hb = 2 + 1 + 1 = 4\n\n        first = 1\n        ha hb ha = 3\n        last = 1\n\n        # hc hd\n        hb hc hd = 3 + 1 + 1 = 5\n        ha hb = 2 + 1 + 1 = 4\n    ')
    summarizer = EdmundsonSummarizer()
    summarizer.null_words = ('na', 'nb', 'nc', 'nd', 'ne')
    sentences = summarizer.location_method(document, 4)
    assert list(map(to_unicode, sentences)) == ['ha = 1 + 1 + 1 = 3', 'ha hb = 2 + 1 + 1 = 4', 'hb hc hd = 3 + 1 + 1 = 5', 'ha hb = 2 + 1 + 1 = 4']

def test_location_method_2():
    if False:
        return 10
    document = build_document_from_string('\n        # na nb nc ha hb\n        ha = 1 + 1 + 0 = 2\n        middle = 0\n        ha hb = 2 + 1 + 0 = 3\n\n        first = 1\n        ha hb ha = 3\n        last = 1\n\n        # hc hd\n        hb hc hd = 3 + 1 + 0 = 4\n        ha hb = 2 + 1 + 0 = 3\n    ')
    summarizer = EdmundsonSummarizer()
    summarizer.null_words = ('na', 'nb', 'nc', 'nd', 'ne')
    sentences = summarizer.location_method(document, 4, w_p1=0, w_p2=0)
    assert list(map(to_unicode, sentences)) == ['ha hb = 2 + 1 + 0 = 3', 'ha hb ha = 3', 'hb hc hd = 3 + 1 + 0 = 4', 'ha hb = 2 + 1 + 0 = 3']