"""
Test some of the functions used for converting an AMT json to a Stanza json
"""
import os
import pytest
import stanza
from stanza.utils.datasets.ner import convert_amt
from stanza.tests import TEST_MODELS_DIR
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]
TEXT = "Jennifer Sh'reyan has lovely antennae."

def fake_label(label, start_char, end_char):
    if False:
        while True:
            i = 10
    return {'label': label, 'startOffset': start_char, 'endOffset': end_char}
LABELS = [fake_label('Person', 0, 8), fake_label('Person', 9, 17), fake_label('Person', 0, 17), fake_label('Andorian', 0, 8), fake_label('Appendage', 29, 37), fake_label('Person', 1, 8), fake_label('Person', 0, 7), fake_label('Person', 0, 9), fake_label('Appendage', 29, 38)]

def fake_labels(*indices):
    if False:
        i = 10
        return i + 15
    return [LABELS[x] for x in indices]

def fake_docs(*indices):
    if False:
        i = 10
        return i + 15
    return [(TEXT, fake_labels(*indices))]

def test_remove_nesting():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test a few orders on nested items to make sure the desired results are coming back\n    '
    result = convert_amt.remove_nesting(fake_docs(0, 1))
    assert result == fake_docs(0, 1)
    result = convert_amt.remove_nesting(fake_docs(0, 4, 1))
    assert result == fake_docs(0, 1, 4)
    result = convert_amt.remove_nesting(fake_docs(0, 0))
    assert result == fake_docs(0)
    result = convert_amt.remove_nesting(fake_docs(0, 2))
    assert result == fake_docs(2)
    result = convert_amt.remove_nesting(fake_docs(1, 2))
    assert result == fake_docs(2)
    result = convert_amt.remove_nesting(fake_docs(5, 2))
    assert result == fake_docs(2)
    result = convert_amt.remove_nesting(fake_docs(0, 4, 2))
    assert result == fake_docs(2, 4)
    result = convert_amt.remove_nesting(fake_docs(2, 4, 0))
    assert result == fake_docs(2, 4)
    result = convert_amt.remove_nesting(fake_docs(0, 3))
    assert result == fake_docs(0)
    result = convert_amt.remove_nesting(fake_docs(3, 0))
    assert result == fake_docs(3)

def test_process_doc():
    if False:
        i = 10
        return i + 15
    nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, processors='tokenize', download_method=None)

    def check_results(doc, *expected):
        if False:
            while True:
                i = 10
        ner = [x[1] for x in doc[0]]
        assert ner == list(expected)
    doc = convert_amt.process_doc(TEXT, fake_labels(2, 4), nlp)
    check_results(doc, 'B-Person', 'I-Person', 'O', 'O', 'B-Appendage', 'O')
    doc = convert_amt.process_doc(TEXT, fake_labels(5, 1, 4), nlp)
    check_results(doc, 'B-Person', 'B-Person', 'O', 'O', 'B-Appendage', 'O')
    doc = convert_amt.process_doc(TEXT, fake_labels(6, 1, 4), nlp)
    check_results(doc, 'B-Person', 'B-Person', 'O', 'O', 'B-Appendage', 'O')
    doc = convert_amt.process_doc(TEXT, fake_labels(7, 4), nlp)
    check_results(doc, 'B-Person', 'O', 'O', 'O', 'B-Appendage', 'O')
    doc = convert_amt.process_doc(TEXT, fake_labels(7, 8), nlp)
    check_results(doc, 'B-Person', 'O', 'O', 'O', 'B-Appendage', 'O')