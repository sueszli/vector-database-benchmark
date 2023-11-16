import pytest

@pytest.mark.parametrize('text,norms,lemmas', [('ім.', ['імені'], ["ім'я"]), ('проф.', ['професор'], ['професор'])])
def test_uk_tokenizer_abbrev_exceptions(uk_tokenizer, text, norms, lemmas):
    if False:
        print('Hello World!')
    tokens = uk_tokenizer(text)
    assert len(tokens) == 1
    assert [token.norm_ for token in tokens] == norms