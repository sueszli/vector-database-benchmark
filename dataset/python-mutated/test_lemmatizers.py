import pytest
from spacy import registry
from spacy.lookups import Lookups
from spacy.util import get_lang_class
LANGUAGES = ['bn', 'ca', 'el', 'en', 'fa', 'fr', 'nb', 'nl', 'sv']

@pytest.mark.parametrize('lang', LANGUAGES)
def test_lemmatizer_initialize(lang, capfd):
    if False:
        while True:
            i = 10

    @registry.misc('lemmatizer_init_lookups')
    def lemmatizer_init_lookups():
        if False:
            while True:
                i = 10
        lookups = Lookups()
        lookups.add_table('lemma_lookup', {'cope': 'cope', 'x': 'y'})
        lookups.add_table('lemma_index', {'verb': ('cope', 'cop')})
        lookups.add_table('lemma_exc', {'verb': {'coping': ('cope',)}})
        lookups.add_table('lemma_rules', {'verb': [['ing', '']]})
        return lookups
    lang_cls = get_lang_class(lang)
    nlp = lang_cls()
    lemmatizer = nlp.add_pipe('lemmatizer', config={'mode': 'lookup'})
    assert not lemmatizer.lookups.tables
    nlp.config['initialize']['components']['lemmatizer'] = {'lookups': {'@misc': 'lemmatizer_init_lookups'}}
    with pytest.raises(ValueError):
        nlp('x')
    nlp.initialize()
    assert lemmatizer.lookups.tables
    doc = nlp('x')
    captured = capfd.readouterr()
    assert not captured.out
    assert doc[0].lemma_ == 'y'
    nlp = lang_cls()
    lemmatizer = nlp.add_pipe('lemmatizer', config={'mode': 'lookup'})
    lemmatizer.initialize(lookups=lemmatizer_init_lookups())
    assert nlp('x')[0].lemma_ == 'y'
    for mode in ('rule', 'lookup', 'pos_lookup'):
        (required, optional) = lemmatizer.get_lookups_config(mode)
        assert isinstance(required, list)
        assert isinstance(optional, list)