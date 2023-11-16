"""Unit tests to provide coverage not easily obtained from integration tests."""
from __future__ import annotations
from ansible.module_utils.basic import _get_available_hash_algorithms

def test_unavailable_algorithm(mocker):
    if False:
        i = 10
        return i + 15
    "Simulate an available algorithm that isn't."
    expected_algorithms = {'sha256', 'sha512'}
    mocker.patch('hashlib.algorithms_available', expected_algorithms | {'not_actually_available'})
    available_algorithms = _get_available_hash_algorithms()
    assert sorted(expected_algorithms) == sorted(available_algorithms)

def test_fips_mode(mocker):
    if False:
        for i in range(10):
            print('nop')
    'Simulate running in FIPS mode on Python 2.7.9 or later.'
    expected_algorithms = {'sha256', 'sha512'}
    mocker.patch('hashlib.algorithms_available', expected_algorithms | {'md5'})
    mocker.patch('hashlib.md5').side_effect = ValueError()
    available_algorithms = _get_available_hash_algorithms()
    assert sorted(expected_algorithms) == sorted(available_algorithms)