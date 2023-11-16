import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore

@pytest.fixture
def matcher(en_vocab):
    if False:
        i = 10
        return i + 15
    rules = {'JS': [[{'ORTH': 'JavaScript'}]], 'GoogleNow': [[{'ORTH': 'Google'}, {'ORTH': 'Now'}]], 'Java': [[{'LOWER': 'java'}]]}
    matcher = Matcher(en_vocab)
    for (key, patterns) in rules.items():
        matcher.add(key, patterns)
    return matcher

def test_matcher_from_api_docs(en_vocab):
    if False:
        print('Hello World!')
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'test'}]
    assert len(matcher) == 0
    matcher.add('Rule', [pattern])
    assert len(matcher) == 1
    matcher.remove('Rule')
    assert 'Rule' not in matcher
    matcher.add('Rule', [pattern])
    assert 'Rule' in matcher
    (on_match, patterns) = matcher.get('Rule')
    assert len(patterns[0])

def test_matcher_empty_patterns_warns(en_vocab):
    if False:
        print('Hello World!')
    matcher = Matcher(en_vocab)
    assert len(matcher) == 0
    doc = Doc(en_vocab, words=['This', 'is', 'quite', 'something'])
    with pytest.warns(UserWarning):
        matcher(doc)
    assert len(doc.ents) == 0

def test_matcher_from_usage_docs(en_vocab):
    if False:
        return 10
    text = 'Wow 😀 This is really cool! 😂 😂'
    doc = Doc(en_vocab, words=text.split(' '))
    pos_emoji = ['😀', '😃', '😂', '🤣', '😊', '😍']
    pos_patterns = [[{'ORTH': emoji}] for emoji in pos_emoji]

    def label_sentiment(matcher, doc, i, matches):
        if False:
            i = 10
            return i + 15
        (match_id, start, end) = matches[i]
        if doc.vocab.strings[match_id] == 'HAPPY':
            doc.sentiment += 0.1
        span = doc[start:end]
        with doc.retokenize() as retokenizer:
            retokenizer.merge(span)
        token = doc[start]
        token.vocab[token.text].norm_ = 'happy emoji'
    matcher = Matcher(en_vocab)
    matcher.add('HAPPY', pos_patterns, on_match=label_sentiment)
    matcher(doc)
    assert doc.sentiment != 0
    assert doc[1].norm_ == 'happy emoji'

def test_matcher_len_contains(matcher):
    if False:
        while True:
            i = 10
    assert len(matcher) == 3
    matcher.add('TEST', [[{'ORTH': 'test'}]])
    assert 'TEST' in matcher
    assert 'TEST2' not in matcher

def test_matcher_add_new_api(en_vocab):
    if False:
        while True:
            i = 10
    doc = Doc(en_vocab, words=['a', 'b'])
    patterns = [[{'TEXT': 'a'}], [{'TEXT': 'a'}, {'TEXT': 'b'}]]
    matcher = Matcher(en_vocab)
    on_match = Mock()
    matcher = Matcher(en_vocab)
    matcher.add('NEW_API', patterns)
    assert len(matcher(doc)) == 2
    matcher = Matcher(en_vocab)
    on_match = Mock()
    matcher.add('NEW_API_CALLBACK', patterns, on_match=on_match)
    assert len(matcher(doc)) == 2
    assert on_match.call_count == 2

def test_matcher_no_match(matcher):
    if False:
        print('Hello World!')
    doc = Doc(matcher.vocab, words=['I', 'like', 'cheese', '.'])
    assert matcher(doc) == []

def test_matcher_match_start(matcher):
    if False:
        for i in range(10):
            print('nop')
    doc = Doc(matcher.vocab, words=['JavaScript', 'is', 'good'])
    assert matcher(doc) == [(matcher.vocab.strings['JS'], 0, 1)]

def test_matcher_match_end(matcher):
    if False:
        i = 10
        return i + 15
    words = ['I', 'like', 'java']
    doc = Doc(matcher.vocab, words=words)
    assert matcher(doc) == [(doc.vocab.strings['Java'], 2, 3)]

def test_matcher_match_middle(matcher):
    if False:
        for i in range(10):
            print('nop')
    words = ['I', 'like', 'Google', 'Now', 'best']
    doc = Doc(matcher.vocab, words=words)
    assert matcher(doc) == [(doc.vocab.strings['GoogleNow'], 2, 4)]

def test_matcher_match_multi(matcher):
    if False:
        return 10
    words = ['I', 'like', 'Google', 'Now', 'and', 'java', 'best']
    doc = Doc(matcher.vocab, words=words)
    assert matcher(doc) == [(doc.vocab.strings['GoogleNow'], 2, 4), (doc.vocab.strings['Java'], 5, 6)]

@pytest.mark.parametrize('rules,match_locs', [({'GoogleNow': [[{'ORTH': {'FUZZY': 'Google'}}, {'ORTH': 'Now'}]]}, [(2, 4)]), ({'Java': [[{'LOWER': {'FUZZY': 'java'}}]]}, [(5, 6)]), ({'JS': [[{'ORTH': {'FUZZY': 'JavaScript'}}]], 'GoogleNow': [[{'ORTH': {'FUZZY': 'Google'}}, {'ORTH': 'Now'}]], 'Java': [[{'LOWER': {'FUZZY': 'java'}}]]}, [(2, 4), (5, 6), (8, 9)]), ({'A': [[{'ORTH': {'FUZZY': 'Javascripts'}}]], 'B': [[{'ORTH': {'FUZZY5': 'Javascripts'}}]]}, [(8, 9)])])
def test_matcher_match_fuzzy(en_vocab, rules, match_locs):
    if False:
        i = 10
        return i + 15
    words = ['They', 'like', 'Goggle', 'Now', 'and', 'Jav', 'but', 'not', 'JvvaScrpt']
    doc = Doc(en_vocab, words=words)
    matcher = Matcher(en_vocab)
    for (key, patterns) in rules.items():
        matcher.add(key, patterns)
    assert match_locs == [(start, end) for (m_id, start, end) in matcher(doc)]

@pytest.mark.parametrize('set_op', ['IN', 'NOT_IN'])
def test_matcher_match_fuzzy_set_op_longest(en_vocab, set_op):
    if False:
        while True:
            i = 10
    rules = {'GoogleNow': [[{'ORTH': {'FUZZY': {set_op: ['Google', 'Now']}}, 'OP': '+'}]]}
    matcher = Matcher(en_vocab)
    for (key, patterns) in rules.items():
        matcher.add(key, patterns, greedy='LONGEST')
    words = ['They', 'like', 'Goggle', 'Noo']
    doc = Doc(en_vocab, words=words)
    assert len(matcher(doc)) == 1

def test_matcher_match_fuzzy_set_multiple(en_vocab):
    if False:
        for i in range(10):
            print('nop')
    rules = {'GoogleNow': [[{'ORTH': {'FUZZY': {'IN': ['Google', 'Now']}, 'NOT_IN': ['Goggle']}, 'OP': '+'}]]}
    matcher = Matcher(en_vocab)
    for (key, patterns) in rules.items():
        matcher.add(key, patterns, greedy='LONGEST')
    words = ['They', 'like', 'Goggle', 'Noo']
    doc = Doc(matcher.vocab, words=words)
    assert matcher(doc) == [(doc.vocab.strings['GoogleNow'], 3, 4)]

@pytest.mark.parametrize('fuzzyn', range(1, 10))
def test_matcher_match_fuzzyn_all_insertions(en_vocab, fuzzyn):
    if False:
        return 10
    matcher = Matcher(en_vocab)
    matcher.add('GoogleNow', [[{'ORTH': {f'FUZZY{fuzzyn}': 'GoogleNow'}}]])
    words = ['GoogleNow' + 'a' * i for i in range(0, 10)]
    doc = Doc(en_vocab, words)
    assert len(matcher(doc)) == fuzzyn + 1

@pytest.mark.parametrize('fuzzyn', range(1, 6))
def test_matcher_match_fuzzyn_various_edits(en_vocab, fuzzyn):
    if False:
        return 10
    matcher = Matcher(en_vocab)
    matcher.add('GoogleNow', [[{'ORTH': {f'FUZZY{fuzzyn}': 'GoogleNow'}}]])
    words = ['GoogleNow', 'GoogleNuw', 'GoogleNuew', 'GoogleNoweee', 'GiggleNuw3', 'gouggle5New']
    doc = Doc(en_vocab, words)
    assert len(matcher(doc)) == fuzzyn + 1

@pytest.mark.parametrize('greedy', ['FIRST', 'LONGEST'])
@pytest.mark.parametrize('set_op', ['IN', 'NOT_IN'])
def test_matcher_match_fuzzyn_set_op_longest(en_vocab, greedy, set_op):
    if False:
        return 10
    rules = {'GoogleNow': [[{'ORTH': {'FUZZY2': {set_op: ['Google', 'Now']}}, 'OP': '+'}]]}
    matcher = Matcher(en_vocab)
    for (key, patterns) in rules.items():
        matcher.add(key, patterns, greedy=greedy)
    words = ['They', 'like', 'Goggle', 'Noo']
    doc = Doc(matcher.vocab, words=words)
    spans = matcher(doc, as_spans=True)
    assert len(spans) == 1
    if set_op == 'IN':
        assert spans[0].text == 'Goggle Noo'
    else:
        assert spans[0].text == 'They like'

def test_matcher_match_fuzzyn_set_multiple(en_vocab):
    if False:
        print('Hello World!')
    rules = {'GoogleNow': [[{'ORTH': {'FUZZY1': {'IN': ['Google', 'Now']}, 'NOT_IN': ['Goggle']}, 'OP': '+'}]]}
    matcher = Matcher(en_vocab)
    for (key, patterns) in rules.items():
        matcher.add(key, patterns, greedy='LONGEST')
    words = ['They', 'like', 'Goggle', 'Noo']
    doc = Doc(matcher.vocab, words=words)
    assert matcher(doc) == [(doc.vocab.strings['GoogleNow'], 3, 4)]

def test_matcher_empty_dict(en_vocab):
    if False:
        print('Hello World!')
    'Test matcher allows empty token specs, meaning match on any token.'
    matcher = Matcher(en_vocab)
    doc = Doc(matcher.vocab, words=['a', 'b', 'c'])
    matcher.add('A.C', [[{'ORTH': 'a'}, {}, {'ORTH': 'c'}]])
    matches = matcher(doc)
    assert len(matches) == 1
    assert matches[0][1:] == (0, 3)
    matcher = Matcher(en_vocab)
    matcher.add('A.', [[{'ORTH': 'a'}, {}]])
    matches = matcher(doc)
    assert matches[0][1:] == (0, 2)

def test_matcher_operator_shadow(en_vocab):
    if False:
        return 10
    matcher = Matcher(en_vocab)
    doc = Doc(matcher.vocab, words=['a', 'b', 'c'])
    pattern = [{'ORTH': 'a'}, {'IS_ALPHA': True, 'OP': '+'}, {'ORTH': 'c'}]
    matcher.add('A.C', [pattern])
    matches = matcher(doc)
    assert len(matches) == 1
    assert matches[0][1:] == (0, 3)

def test_matcher_match_zero(matcher):
    if False:
        while True:
            i = 10
    words1 = 'He said , " some words " ...'.split()
    words2 = 'He said , " some three words " ...'.split()
    pattern1 = [{'ORTH': '"'}, {'OP': '!', 'IS_PUNCT': True}, {'OP': '!', 'IS_PUNCT': True}, {'ORTH': '"'}]
    pattern2 = [{'ORTH': '"'}, {'IS_PUNCT': True}, {'IS_PUNCT': True}, {'IS_PUNCT': True}, {'ORTH': '"'}]
    matcher.add('Quote', [pattern1])
    doc = Doc(matcher.vocab, words=words1)
    assert len(matcher(doc)) == 1
    doc = Doc(matcher.vocab, words=words2)
    assert len(matcher(doc)) == 0
    matcher.add('Quote', [pattern2])
    assert len(matcher(doc)) == 0

def test_matcher_match_zero_plus(matcher):
    if False:
        print('Hello World!')
    words = 'He said , " some words " ...'.split()
    pattern = [{'ORTH': '"'}, {'OP': '*', 'IS_PUNCT': False}, {'ORTH': '"'}]
    matcher = Matcher(matcher.vocab)
    matcher.add('Quote', [pattern])
    doc = Doc(matcher.vocab, words=words)
    assert len(matcher(doc)) == 1

def test_matcher_match_one_plus(matcher):
    if False:
        while True:
            i = 10
    control = Matcher(matcher.vocab)
    control.add('BasicPhilippe', [[{'ORTH': 'Philippe'}]])
    doc = Doc(control.vocab, words=['Philippe', 'Philippe'])
    m = control(doc)
    assert len(m) == 2
    pattern = [{'ORTH': 'Philippe'}, {'ORTH': 'Philippe', 'OP': '+'}]
    matcher.add('KleenePhilippe', [pattern])
    m = matcher(doc)
    assert len(m) == 1

def test_matcher_any_token_operator(en_vocab):
    if False:
        print('Hello World!')
    'Test that patterns with "any token" {} work with operators.'
    matcher = Matcher(en_vocab)
    matcher.add('TEST', [[{'ORTH': 'test'}, {'OP': '*'}]])
    doc = Doc(en_vocab, words=['test', 'hello', 'world'])
    matches = [doc[start:end].text for (_, start, end) in matcher(doc)]
    assert len(matches) == 3
    assert matches[0] == 'test'
    assert matches[1] == 'test hello'
    assert matches[2] == 'test hello world'

@pytest.mark.usefixtures('clean_underscore')
def test_matcher_extension_attribute(en_vocab):
    if False:
        i = 10
        return i + 15
    matcher = Matcher(en_vocab)
    get_is_fruit = lambda token: token.text in ('apple', 'banana')
    Token.set_extension('is_fruit', getter=get_is_fruit, force=True)
    pattern = [{'ORTH': 'an'}, {'_': {'is_fruit': True}}]
    matcher.add('HAVING_FRUIT', [pattern])
    doc = Doc(en_vocab, words=['an', 'apple'])
    matches = matcher(doc)
    assert len(matches) == 1
    doc = Doc(en_vocab, words=['an', 'aardvark'])
    matches = matcher(doc)
    assert len(matches) == 0

def test_matcher_set_value(en_vocab):
    if False:
        while True:
            i = 10
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': {'IN': ['an', 'a']}}]
    matcher.add('A_OR_AN', [pattern])
    doc = Doc(en_vocab, words=['an', 'a', 'apple'])
    matches = matcher(doc)
    assert len(matches) == 2
    doc = Doc(en_vocab, words=['aardvark'])
    matches = matcher(doc)
    assert len(matches) == 0

def test_matcher_set_value_operator(en_vocab):
    if False:
        print('Hello World!')
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': {'IN': ['a', 'the']}, 'OP': '?'}, {'ORTH': 'house'}]
    matcher.add('DET_HOUSE', [pattern])
    doc = Doc(en_vocab, words=['In', 'a', 'house'])
    matches = matcher(doc)
    assert len(matches) == 2
    doc = Doc(en_vocab, words=['my', 'house'])
    matches = matcher(doc)
    assert len(matches) == 1

def test_matcher_subset_value_operator(en_vocab):
    if False:
        print('Hello World!')
    matcher = Matcher(en_vocab)
    pattern = [{'MORPH': {'IS_SUBSET': ['Feat=Val', 'Feat2=Val2']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    assert len(matcher(doc)) == 3
    doc[0].set_morph('Feat=Val')
    assert len(matcher(doc)) == 3
    doc[0].set_morph('Feat=Val|Feat2=Val2')
    assert len(matcher(doc)) == 3
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3')
    assert len(matcher(doc)) == 2
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3|Feat4=Val4')
    assert len(matcher(doc)) == 2
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUBSET': ['A', 'B']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUBSET': []}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 0
    Token.set_extension('ext', default=[])
    matcher = Matcher(en_vocab)
    pattern = [{'_': {'ext': {'IS_SUBSET': ['A', 'B']}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = ['A']
    doc[1]._.ext = ['C', 'D']
    assert len(matcher(doc)) == 2

def test_matcher_superset_value_operator(en_vocab):
    if False:
        while True:
            i = 10
    matcher = Matcher(en_vocab)
    pattern = [{'MORPH': {'IS_SUPERSET': ['Feat=Val', 'Feat2=Val2', 'Feat3=Val3']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    assert len(matcher(doc)) == 0
    doc[0].set_morph('Feat=Val|Feat2=Val2')
    assert len(matcher(doc)) == 0
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3')
    assert len(matcher(doc)) == 1
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3|Feat4=Val4')
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUPERSET': ['A', 'B']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 0
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUPERSET': ['A']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUPERSET': []}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 3
    Token.set_extension('ext', default=[])
    matcher = Matcher(en_vocab)
    pattern = [{'_': {'ext': {'IS_SUPERSET': ['A']}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = ['A', 'B']
    assert len(matcher(doc)) == 1

def test_matcher_intersect_value_operator(en_vocab):
    if False:
        for i in range(10):
            print('nop')
    matcher = Matcher(en_vocab)
    pattern = [{'MORPH': {'INTERSECTS': ['Feat=Val', 'Feat2=Val2', 'Feat3=Val3']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    assert len(matcher(doc)) == 0
    doc[0].set_morph('Feat=Val')
    assert len(matcher(doc)) == 1
    doc[0].set_morph('Feat=Val|Feat2=Val2')
    assert len(matcher(doc)) == 1
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3')
    assert len(matcher(doc)) == 1
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3|Feat4=Val4')
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'INTERSECTS': ['A', 'B']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'INTERSECTS': []}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 0
    Token.set_extension('ext', default=[])
    matcher = Matcher(en_vocab)
    pattern = [{'_': {'ext': {'INTERSECTS': ['A', 'C']}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = ['A', 'B']
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'_': {'ext': {'INTERSECTS': ['Abx', 'C']}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = [['Abx'], 'B']
    assert len(matcher(doc)) == 0
    doc[0]._.ext = ['Abx', 'B']
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'_': {'ext': {'INTERSECTS': []}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = ['A', 'B']
    assert len(matcher(doc)) == 0
    matcher = Matcher(en_vocab)
    pattern = [{'_': {'ext': {'INTERSECTS': ['A', 'B']}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = []
    assert len(matcher(doc)) == 0

def test_matcher_morph_handling(en_vocab):
    if False:
        for i in range(10):
            print('nop')
    matcher = Matcher(en_vocab)
    pattern1 = [{'MORPH': {'IN': ['Feat1=Val1|Feat2=Val2']}}]
    pattern2 = [{'MORPH': {'IN': ['Feat2=Val2|Feat1=Val1']}}]
    matcher.add('M', [pattern1])
    matcher.add('N', [pattern2])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    assert len(matcher(doc)) == 0
    doc[0].set_morph('Feat2=Val2|Feat1=Val1')
    assert len(matcher(doc)) == 2
    doc[0].set_morph('Feat1=Val1|Feat2=Val2')
    assert len(matcher(doc)) == 2
    matcher = Matcher(en_vocab)
    pattern1 = [{'MORPH': {'IS_SUPERSET': ['Feat1=Val1', 'Feat2=Val2']}}]
    pattern2 = [{'MORPH': {'IS_SUPERSET': ['Feat1=Val1', 'Feat1=Val3', 'Feat2=Val2']}}]
    matcher.add('M', [pattern1])
    matcher.add('N', [pattern2])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    assert len(matcher(doc)) == 0
    doc[0].set_morph('Feat2=Val2,Val3|Feat1=Val1')
    assert len(matcher(doc)) == 1
    doc[0].set_morph('Feat1=Val1,Val3|Feat2=Val2')
    assert len(matcher(doc)) == 2

def test_matcher_regex(en_vocab):
    if False:
        print('Hello World!')
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': {'REGEX': '(?:a|an)'}}]
    matcher.add('A_OR_AN', [pattern])
    doc = Doc(en_vocab, words=['an', 'a', 'hi'])
    matches = matcher(doc)
    assert len(matches) == 2
    doc = Doc(en_vocab, words=['bye'])
    matches = matcher(doc)
    assert len(matches) == 0

def test_matcher_regex_set_in(en_vocab):
    if False:
        while True:
            i = 10
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': {'REGEX': {'IN': ['(?:a)', '(?:an)']}}}]
    matcher.add('A_OR_AN', [pattern])
    doc = Doc(en_vocab, words=['an', 'a', 'hi'])
    matches = matcher(doc)
    assert len(matches) == 2
    doc = Doc(en_vocab, words=['bye'])
    matches = matcher(doc)
    assert len(matches) == 0

def test_matcher_regex_set_not_in(en_vocab):
    if False:
        return 10
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': {'REGEX': {'NOT_IN': ['(?:a)', '(?:an)']}}}]
    matcher.add('A_OR_AN', [pattern])
    doc = Doc(en_vocab, words=['an', 'a', 'hi'])
    matches = matcher(doc)
    assert len(matches) == 1
    doc = Doc(en_vocab, words=['bye'])
    matches = matcher(doc)
    assert len(matches) == 1

def test_matcher_regex_shape(en_vocab):
    if False:
        while True:
            i = 10
    matcher = Matcher(en_vocab)
    pattern = [{'SHAPE': {'REGEX': '^[^x]+$'}}]
    matcher.add('NON_ALPHA', [pattern])
    doc = Doc(en_vocab, words=['99', 'problems', '!'])
    matches = matcher(doc)
    assert len(matches) == 2
    doc = Doc(en_vocab, words=['bye'])
    matches = matcher(doc)
    assert len(matches) == 0

@pytest.mark.parametrize('cmp, bad', [('==', ['a', 'aaa']), ('!=', ['aa']), ('>=', ['a']), ('<=', ['aaa']), ('>', ['a', 'aa']), ('<', ['aa', 'aaa'])])
def test_matcher_compare_length(en_vocab, cmp, bad):
    if False:
        return 10
    matcher = Matcher(en_vocab)
    pattern = [{'LENGTH': {cmp: 2}}]
    matcher.add('LENGTH_COMPARE', [pattern])
    doc = Doc(en_vocab, words=['a', 'aa', 'aaa'])
    matches = matcher(doc)
    assert len(matches) == len(doc) - len(bad)
    doc = Doc(en_vocab, words=bad)
    matches = matcher(doc)
    assert len(matches) == 0

def test_matcher_extension_set_membership(en_vocab):
    if False:
        i = 10
        return i + 15
    matcher = Matcher(en_vocab)
    get_reversed = lambda token: ''.join(reversed(token.text))
    Token.set_extension('reversed', getter=get_reversed, force=True)
    pattern = [{'_': {'reversed': {'IN': ['eyb', 'ih']}}}]
    matcher.add('REVERSED', [pattern])
    doc = Doc(en_vocab, words=['hi', 'bye', 'hello'])
    matches = matcher(doc)
    assert len(matches) == 2
    doc = Doc(en_vocab, words=['aardvark'])
    matches = matcher(doc)
    assert len(matches) == 0

def test_matcher_extension_in_set_predicate(en_vocab):
    if False:
        for i in range(10):
            print('nop')
    matcher = Matcher(en_vocab)
    Token.set_extension('ext', default=[])
    pattern = [{'_': {'ext': {'IN': ['A', 'C']}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = ['A', 'B']
    assert len(matcher(doc)) == 0
    doc[0]._.ext = ['A']
    assert len(matcher(doc)) == 0
    doc[0]._.ext = 'A'
    assert len(matcher(doc)) == 1

def test_matcher_basic_check(en_vocab):
    if False:
        i = 10
        return i + 15
    matcher = Matcher(en_vocab)
    pattern = [{'TEXT': 'hello'}, {'TEXT': 'world'}]
    with pytest.raises(ValueError):
        matcher.add('TEST', pattern)

def test_attr_pipeline_checks(en_vocab):
    if False:
        while True:
            i = 10
    doc1 = Doc(en_vocab, words=['Test'])
    doc1[0].dep_ = 'ROOT'
    doc2 = Doc(en_vocab, words=['Test'])
    doc2[0].tag_ = 'TAG'
    doc2[0].pos_ = 'X'
    doc2[0].set_morph('Feat=Val')
    doc2[0].lemma_ = 'LEMMA'
    doc3 = Doc(en_vocab, words=['Test'])
    matcher = Matcher(en_vocab)
    matcher.add('TEST', [[{'DEP': 'a'}]])
    matcher(doc1)
    with pytest.raises(ValueError):
        matcher(doc2)
    with pytest.raises(ValueError):
        matcher(doc3)
    matcher(doc2, allow_missing=True)
    matcher(doc3, allow_missing=True)
    for attr in ('TAG', 'POS', 'LEMMA'):
        matcher = Matcher(en_vocab)
        matcher.add('TEST', [[{attr: 'a'}]])
        matcher(doc2)
        with pytest.raises(ValueError):
            matcher(doc1)
        with pytest.raises(ValueError):
            matcher(doc3)
    matcher = Matcher(en_vocab)
    matcher.add('TEST', [[{'ORTH': 'a'}]])
    matcher(doc1)
    matcher(doc2)
    matcher(doc3)
    matcher = Matcher(en_vocab)
    matcher.add('TEST', [[{'TEXT': 'a'}]])
    matcher(doc1)
    matcher(doc2)
    matcher(doc3)

@pytest.mark.parametrize('pattern,text', [([{'IS_ALPHA': True}], 'a'), ([{'IS_ASCII': True}], 'a'), ([{'IS_DIGIT': True}], '1'), ([{'IS_LOWER': True}], 'a'), ([{'IS_UPPER': True}], 'A'), ([{'IS_TITLE': True}], 'Aaaa'), ([{'IS_PUNCT': True}], '.'), ([{'IS_SPACE': True}], '\n'), ([{'IS_BRACKET': True}], '['), ([{'IS_QUOTE': True}], '"'), ([{'IS_LEFT_PUNCT': True}], '``'), ([{'IS_RIGHT_PUNCT': True}], "''"), ([{'IS_STOP': True}], 'the'), ([{'SPACY': True}], 'the'), ([{'LIKE_NUM': True}], '1'), ([{'LIKE_URL': True}], 'http://example.com'), ([{'LIKE_EMAIL': True}], 'mail@example.com')])
def test_matcher_schema_token_attributes(en_vocab, pattern, text):
    if False:
        i = 10
        return i + 15
    matcher = Matcher(en_vocab)
    doc = Doc(en_vocab, words=text.split(' '))
    matcher.add('Rule', [pattern])
    assert len(matcher) == 1
    matches = matcher(doc)
    assert len(matches) == 1

@pytest.mark.filterwarnings('ignore:\\[W036')
def test_matcher_valid_callback(en_vocab):
    if False:
        for i in range(10):
            print('nop')
    'Test that on_match can only be None or callable.'
    matcher = Matcher(en_vocab)
    with pytest.raises(ValueError):
        matcher.add('TEST', [[{'TEXT': 'test'}]], on_match=[])
    matcher(Doc(en_vocab, words=['test']))

def test_matcher_callback(en_vocab):
    if False:
        while True:
            i = 10
    mock = Mock()
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'test'}]
    matcher.add('Rule', [pattern], on_match=mock)
    doc = Doc(en_vocab, words=['This', 'is', 'a', 'test', '.'])
    matches = matcher(doc)
    mock.assert_called_once_with(matcher, doc, 0, matches)

def test_matcher_callback_with_alignments(en_vocab):
    if False:
        return 10
    mock = Mock()
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'test'}]
    matcher.add('Rule', [pattern], on_match=mock)
    doc = Doc(en_vocab, words=['This', 'is', 'a', 'test', '.'])
    matches = matcher(doc, with_alignments=True)
    mock.assert_called_once_with(matcher, doc, 0, matches)

def test_matcher_span(matcher):
    if False:
        i = 10
        return i + 15
    text = 'JavaScript is good but Java is better'
    doc = Doc(matcher.vocab, words=text.split())
    span_js = doc[:3]
    span_java = doc[4:]
    assert len(matcher(doc)) == 2
    assert len(matcher(span_js)) == 1
    assert len(matcher(span_java)) == 1

def test_matcher_as_spans(matcher):
    if False:
        for i in range(10):
            print('nop')
    'Test the new as_spans=True API.'
    text = 'JavaScript is good but Java is better'
    doc = Doc(matcher.vocab, words=text.split())
    matches = matcher(doc, as_spans=True)
    assert len(matches) == 2
    assert isinstance(matches[0], Span)
    assert matches[0].text == 'JavaScript'
    assert matches[0].label_ == 'JS'
    assert isinstance(matches[1], Span)
    assert matches[1].text == 'Java'
    assert matches[1].label_ == 'Java'
    matches = matcher(doc[1:], as_spans=True)
    assert len(matches) == 1
    assert isinstance(matches[0], Span)
    assert matches[0].text == 'Java'
    assert matches[0].label_ == 'Java'

def test_matcher_deprecated(matcher):
    if False:
        for i in range(10):
            print('nop')
    doc = Doc(matcher.vocab, words=['hello', 'world'])
    with pytest.warns(DeprecationWarning) as record:
        for _ in matcher.pipe([doc]):
            pass
        assert record.list
        assert 'spaCy v3.0' in str(record.list[0].message)

def test_matcher_remove_zero_operator(en_vocab):
    if False:
        for i in range(10):
            print('nop')
    matcher = Matcher(en_vocab)
    pattern = [{'OP': '!'}]
    matcher.add('Rule', [pattern])
    doc = Doc(en_vocab, words=['This', 'is', 'a', 'test', '.'])
    matches = matcher(doc)
    assert len(matches) == 0
    assert 'Rule' in matcher
    matcher.remove('Rule')
    assert 'Rule' not in matcher

def test_matcher_no_zero_length(en_vocab):
    if False:
        print('Hello World!')
    doc = Doc(en_vocab, words=['a', 'b'], tags=['A', 'B'])
    matcher = Matcher(en_vocab)
    matcher.add('TEST', [[{'TAG': 'C', 'OP': '?'}]])
    assert len(matcher(doc)) == 0

def test_matcher_ent_iob_key(en_vocab):
    if False:
        for i in range(10):
            print('nop')
    'Test that patterns with ent_iob works correctly.'
    matcher = Matcher(en_vocab)
    matcher.add('Rule', [[{'ENT_IOB': 'I'}]])
    doc1 = Doc(en_vocab, words=['I', 'visited', 'New', 'York', 'and', 'California'])
    doc1.ents = [Span(doc1, 2, 4, label='GPE'), Span(doc1, 5, 6, label='GPE')]
    doc2 = Doc(en_vocab, words=['I', 'visited', 'my', 'friend', 'Alicia'])
    doc2.ents = [Span(doc2, 4, 5, label='PERSON')]
    matches1 = [doc1[start:end].text for (_, start, end) in matcher(doc1)]
    matches2 = [doc2[start:end].text for (_, start, end) in matcher(doc2)]
    assert len(matches1) == 1
    assert matches1[0] == 'York'
    assert len(matches2) == 0
    matcher = Matcher(en_vocab)
    matcher.add('Rule', [[{'ENT_IOB': 'I', 'OP': '+'}]])
    doc = Doc(en_vocab, words=['I', 'visited', 'my', 'friend', 'Anna', 'Maria', 'Esperanza'])
    doc.ents = [Span(doc, 4, 7, label='PERSON')]
    matches = [doc[start:end].text for (_, start, end) in matcher(doc)]
    assert len(matches) == 3
    assert matches[0] == 'Maria'
    assert matches[1] == 'Maria Esperanza'
    assert matches[2] == 'Esperanza'

def test_matcher_min_max_operator(en_vocab):
    if False:
        for i in range(10):
            print('nop')
    doc = Doc(en_vocab, words=['foo', 'bar', 'foo', 'foo', 'bar', 'foo', 'foo', 'foo', 'bar', 'bar'])
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'foo', 'OP': '{3}'}]
    matcher.add('TEST', [pattern])
    matches1 = [doc[start:end].text for (_, start, end) in matcher(doc)]
    assert len(matches1) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'foo', 'OP': '{2,}'}]
    matcher.add('TEST', [pattern])
    matches2 = [doc[start:end].text for (_, start, end) in matcher(doc)]
    assert len(matches2) == 4
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'foo', 'OP': '{,2}'}]
    matcher.add('TEST', [pattern])
    matches3 = [doc[start:end].text for (_, start, end) in matcher(doc)]
    assert len(matches3) == 9
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'foo', 'OP': '{2,3}'}]
    matcher.add('TEST', [pattern])
    matches4 = [doc[start:end].text for (_, start, end) in matcher(doc)]
    assert len(matches4) == 4