"""Simple tests to make sure all stemmers share the same API."""
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from sumy.nlp.stemmers import null_stemmer, Stemmer

def test_missing_stemmer_language():
    if False:
        return 10
    with pytest.raises(LookupError):
        Stemmer('klingon')

def test_null_stemmer():
    if False:
        while True:
            i = 10
    assert 'ľščťžýáíé' == null_stemmer('ľŠčŤžÝáÍé')

def test_english_stemmer():
    if False:
        i = 10
        return i + 15
    english_stemmer = Stemmer('english')
    assert 'beauti' == english_stemmer('beautiful')

def test_german_stemmer():
    if False:
        for i in range(10):
            print('nop')
    german_stemmer = Stemmer('german')
    assert 'sterb' == german_stemmer('sterben')

def test_czech_stemmer():
    if False:
        return 10
    czech_stemmer = Stemmer('czech')
    assert 'pěkn' == czech_stemmer('pěkný')

def test_french_stemmer():
    if False:
        for i in range(10):
            print('nop')
    french_stemmer = Stemmer('czech')
    assert 'jol' == french_stemmer('jolies')

def test_slovak_stemmer():
    if False:
        print('Hello World!')
    expected = Stemmer('czech')
    actual = Stemmer('slovak')
    assert type(actual) is type(expected)
    assert expected.__dict__ == actual.__dict__

def test_greek_stemmer():
    if False:
        print('Hello World!')
    greek_stemmer = Stemmer('greek')
    assert 'οτ' == greek_stemmer('όταν')
    assert 'εργαζ' == greek_stemmer('εργαζόμενος')

def test_swedish_stemmer():
    if False:
        print('Hello World!')
    swedish_stemmer = Stemmer('swedish')
    assert 'sov' == swedish_stemmer('sover')