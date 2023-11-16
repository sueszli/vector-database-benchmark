from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import pytest
from pytest import approx
from sumy.evaluation import precision, recall, f_score

def test_precision_empty_evaluated():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        precision((), ('s1', 's2', 's3', 's4', 's5'))

def test_precision_empty_reference():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        precision(('s1', 's2', 's3', 's4', 's5'), ())

def test_precision_no_match():
    if False:
        i = 10
        return i + 15
    result = precision(('s1', 's2', 's3', 's4', 's5'), ('s6', 's7', 's8'))
    assert result == 0.0

def test_precision_reference_smaller():
    if False:
        for i in range(10):
            print('nop')
    result = precision(('s1', 's2', 's3', 's4', 's5'), ('s1',))
    assert result == approx(0.2)

def test_precision_evaluated_smaller():
    if False:
        print('Hello World!')
    result = precision(('s1',), ('s1', 's2', 's3', 's4', 's5'))
    assert result == approx(1.0)

def test_precision_equals():
    if False:
        return 10
    sentences = ('s1', 's2', 's3', 's4', 's5')
    result = precision(sentences, sentences)
    assert result == approx(1.0)

def test_recall_empty_evaluated():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        recall((), ('s1', 's2', 's3', 's4', 's5'))

def test_recall_empty_reference():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        recall(('s1', 's2', 's3', 's4', 's5'), ())

def test_recall_no_match():
    if False:
        while True:
            i = 10
    result = recall(('s1', 's2', 's3', 's4', 's5'), ('s6', 's7', 's8'))
    assert result == 0.0

def test_recall_reference_smaller():
    if False:
        print('Hello World!')
    result = recall(('s1', 's2', 's3', 's4', 's5'), ('s1',))
    assert result == approx(1.0)

def test_recall_evaluated_smaller():
    if False:
        i = 10
        return i + 15
    result = recall(('s1',), ('s1', 's2', 's3', 's4', 's5'))
    assert result == approx(0.2)

def test_recall_equals():
    if False:
        i = 10
        return i + 15
    sentences = ('s1', 's2', 's3', 's4', 's5')
    result = recall(sentences, sentences)
    assert result == approx(1.0)

def test_basic_f_score_empty_evaluated():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        f_score((), ('s1', 's2', 's3', 's4', 's5'))

def test_basic_f_score_empty_reference():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        f_score(('s1', 's2', 's3', 's4', 's5'), ())

def test_basic_f_score_no_match():
    if False:
        i = 10
        return i + 15
    result = f_score(('s1', 's2', 's3', 's4', 's5'), ('s6', 's7', 's8'))
    assert result == 0.0

def test_basic_f_score_reference_smaller():
    if False:
        return 10
    result = f_score(('s1', 's2', 's3', 's4', 's5'), ('s1',))
    assert result == approx(1 / 3)

def test_basic_f_score_evaluated_smaller():
    if False:
        for i in range(10):
            print('nop')
    result = f_score(('s1',), ('s1', 's2', 's3', 's4', 's5'))
    assert result == approx(1 / 3)

def test_basic_f_score_equals():
    if False:
        i = 10
        return i + 15
    sentences = ('s1', 's2', 's3', 's4', 's5')
    result = f_score(sentences, sentences)
    assert result == approx(1.0)

def test_f_score_1():
    if False:
        for i in range(10):
            print('nop')
    sentences = (('s1',), ('s1', 's2', 's3', 's4', 's5'))
    result = f_score(*sentences, weight=2.0)
    p = 1 / 1
    r = 1 / 5
    expected = 5 * p * r / (4 * p + r)
    assert result == approx(expected)

def test_f_score_2():
    if False:
        while True:
            i = 10
    sentences = (('s1', 's3', 's6'), ('s1', 's2', 's3', 's4', 's5'))
    result = f_score(*sentences, weight=0.5)
    p = 2 / 3
    r = 2 / 5
    expected = 1.25 * p * r / (0.25 * p + r)
    assert result == approx(expected)