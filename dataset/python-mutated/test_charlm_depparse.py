import pytest
from stanza.resources.default_packages import default_charlms, depparse_charlms
from stanza.resources.print_charlm_depparse import list_depparse

def test_list_depparse():
    if False:
        print('Hello World!')
    models = list_depparse()
    assert 'af' not in depparse_charlms
    assert 'af' in default_charlms
    assert 'af_afribooms_charlm' in models
    assert 'af_afribooms_nocharlm' in models
    assert 'en' in depparse_charlms
    assert 'en' in default_charlms
    assert 'ewt' not in depparse_charlms['en']
    assert 'craft' in depparse_charlms['en']
    assert 'mimic' in depparse_charlms['en']
    assert 'en_ewt_charlm' in models
    assert 'en_ewt_nocharlm' in models
    assert 'en_mimic_charlm' in models
    assert 'en_mimic_nocharlm' not in models
    assert 'en_craft_charlm' not in models
    assert 'en_craft_nocharlm' in models