import unicodedata
from hypothesis import given, settings
from hypothesis.strategies import text

@given(text(min_size=1, max_size=1))
@settings(max_examples=2000)
def test_does_not_generate_surrogates(t):
    if False:
        while True:
            i = 10
    assert unicodedata.category(t) != 'Cs'