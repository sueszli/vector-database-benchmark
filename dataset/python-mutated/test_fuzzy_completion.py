import pytest

@pytest.fixture
def completer():
    if False:
        print('Hello World!')
    import pgcli.pgcompleter as pgcompleter
    return pgcompleter.PGCompleter()

def test_ranking_ignores_identifier_quotes(completer):
    if False:
        for i in range(10):
            print('nop')
    'When calculating result rank, identifier quotes should be ignored.\n\n    The result ranking algorithm ignores identifier quotes. Without this\n    correction, the match "user", which Postgres requires to be quoted\n    since it is also a reserved word, would incorrectly fall below the\n    match user_action because the literal quotation marks in "user"\n    alter the position of the match.\n\n    This test checks that the fuzzy ranking algorithm correctly ignores\n    quotation marks when computing match ranks.\n\n    '
    text = 'user'
    collection = ['user_action', '"user"']
    matches = completer.find_matches(text, collection)
    assert len(matches) == 2

def test_ranking_based_on_shortest_match(completer):
    if False:
        for i in range(10):
            print('nop')
    "Fuzzy result rank should be based on shortest match.\n\n    Result ranking in fuzzy searching is partially based on the length\n    of matches: shorter matches are considered more relevant than\n    longer ones. When searching for the text 'user', the length\n    component of the match 'user_group' could be either 4 ('user') or\n    7 ('user_gr').\n\n    This test checks that the fuzzy ranking algorithm uses the shorter\n    match when calculating result rank.\n\n    "
    text = 'user'
    collection = ['api_user', 'user_group']
    matches = completer.find_matches(text, collection)
    assert matches[1].priority > matches[0].priority

@pytest.mark.parametrize('collection', [['user_action', 'user'], ['user_group', 'user'], ['user_group', 'user_action']])
def test_should_break_ties_using_lexical_order(completer, collection):
    if False:
        return 10
    "Fuzzy result rank should use lexical order to break ties.\n\n    When fuzzy matching, if multiple matches have the same match length and\n    start position, present them in lexical (rather than arbitrary) order. For\n    example, if we have tables 'user', 'user_action', and 'user_group', a\n    search for the text 'user' should present these tables in this order.\n\n    The input collections to this test are out of order; each run checks that\n    the search text 'user' results in the input tables being reordered\n    lexically.\n\n    "
    text = 'user'
    matches = completer.find_matches(text, collection)
    assert matches[1].priority > matches[0].priority

def test_matching_should_be_case_insensitive(completer):
    if False:
        print('Hello World!')
    "Fuzzy matching should keep matches even if letter casing doesn't match.\n\n    This test checks that variations of the text which have different casing\n    are still matched.\n    "
    text = 'foo'
    collection = ['Foo', 'FOO', 'fOO']
    matches = completer.find_matches(text, collection)
    assert len(matches) == 3