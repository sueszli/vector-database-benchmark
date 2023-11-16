"""Unit tests for :func:`~workflow.util.atomic_writer`."""
from __future__ import print_function
import json
import os
import pytest
from util import DEFAULT_SETTINGS
from workflow.util import atomic_writer

def _settings(tempdir):
    if False:
        i = 10
        return i + 15
    'Path to ``settings.json`` file.'
    return os.path.join(tempdir, 'settings.json')

def test_write_file_succeed(tempdir):
    if False:
        print('Hello World!')
    'Succeed, no temp file left'
    p = _settings(tempdir)
    with atomic_writer(p, 'wb') as fp:
        json.dump(DEFAULT_SETTINGS, fp)
    assert len(os.listdir(tempdir)) == 1
    assert os.path.exists(p)

def test_failed_before_writing(tempdir):
    if False:
        for i in range(10):
            print('nop')
    'Exception before writing'
    p = _settings(tempdir)

    def write():
        if False:
            print('Hello World!')
        with atomic_writer(p, 'wb'):
            raise Exception()
    with pytest.raises(Exception):
        write()
    assert not os.listdir(tempdir)

def test_failed_after_writing(tempdir):
    if False:
        return 10
    'Exception after writing'
    p = _settings(tempdir)

    def write():
        if False:
            for i in range(10):
                print('nop')
        with atomic_writer(p, 'wb') as fp:
            json.dump(DEFAULT_SETTINGS, fp)
            raise Exception()
    with pytest.raises(Exception):
        write()
    assert not os.listdir(tempdir)

def test_failed_without_overwriting(tempdir):
    if False:
        while True:
            i = 10
    "AtomicWriter: Exception after writing won't overwrite the old file"
    p = _settings(tempdir)
    mockSettings = {}

    def write():
        if False:
            return 10
        with atomic_writer(p, 'wb') as fp:
            json.dump(mockSettings, fp)
            raise Exception()
    with atomic_writer(p, 'wb') as fp:
        json.dump(DEFAULT_SETTINGS, fp)
    assert len(os.listdir(tempdir)) == 1
    assert os.path.exists(p)
    with pytest.raises(Exception):
        write()
    assert len(os.listdir(tempdir)) == 1
    assert os.path.exists(p)
    with open(p, 'rb') as fp:
        real_settings = json.load(fp)
    assert DEFAULT_SETTINGS == real_settings
if __name__ == '__main__':
    pytest.main([__file__])