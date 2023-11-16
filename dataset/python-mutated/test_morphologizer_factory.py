import pytest
from spacy.lang.ja import Japanese

def test_ja_morphologizer_factory():
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('sudachipy')
    nlp = Japanese()
    morphologizer = nlp.add_pipe('morphologizer')
    assert morphologizer.cfg['extend'] is True