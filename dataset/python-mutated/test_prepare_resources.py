import pytest
import stanza
import stanza.resources.prepare_resources as prepare_resources
from stanza.tests import *
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_split_model_name():
    if False:
        return 10
    (lang, package, processor) = prepare_resources.split_model_name('ro_nonstandard_tagger.pt')
    assert lang == 'ro'
    assert package == 'nonstandard'
    assert processor == 'pos'
    (lang, package, processor) = prepare_resources.split_model_name('en_ncbi_disease_nertagger.pt')
    assert lang == 'en'
    assert package == 'ncbi_disease'
    assert processor == 'ner'
    (lang, package, processor) = prepare_resources.split_model_name('en_pubmed_forward_charlm.pt')
    assert lang == 'en'
    assert package == 'pubmed'
    assert processor == 'forward_charlm'