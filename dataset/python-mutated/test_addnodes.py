"""Test the non-trivial features in the :mod:`sphinx.addnodes` module."""
from __future__ import annotations
import pytest
from sphinx import addnodes

@pytest.fixture()
def sig_elements() -> set[type[addnodes.desc_sig_element]]:
    if False:
        print('Hello World!')
    'Fixture returning the current ``addnodes.SIG_ELEMENTS`` set.'
    original = addnodes.SIG_ELEMENTS.copy()
    yield {*addnodes.SIG_ELEMENTS}
    addnodes.SIG_ELEMENTS = original

def test_desc_sig_element_nodes(sig_elements):
    if False:
        return 10
    'Test the registration of ``desc_sig_element`` subclasses.'
    EXPECTED_SIG_ELEMENTS = {addnodes.desc_sig_space, addnodes.desc_sig_name, addnodes.desc_sig_operator, addnodes.desc_sig_punctuation, addnodes.desc_sig_keyword, addnodes.desc_sig_keyword_type, addnodes.desc_sig_literal_number, addnodes.desc_sig_literal_string, addnodes.desc_sig_literal_char}
    assert addnodes.SIG_ELEMENTS == EXPECTED_SIG_ELEMENTS

    class BuiltInSigElementLikeNode(addnodes.desc_sig_element, _sig_element=True):
        pass

    class Custom1SigElementLikeNode(addnodes.desc_sig_element):
        pass

    class Custom2SigElementLikeNode(addnodes.desc_sig_element, _sig_element=False):
        pass
    assert BuiltInSigElementLikeNode in addnodes.SIG_ELEMENTS
    assert Custom1SigElementLikeNode not in addnodes.SIG_ELEMENTS
    assert Custom2SigElementLikeNode not in addnodes.SIG_ELEMENTS