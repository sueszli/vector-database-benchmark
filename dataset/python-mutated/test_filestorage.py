import os
from hypothesis import configuration as fs
previous_home_dir = None

def setup_function(function):
    if False:
        while True:
            i = 10
    global previous_home_dir
    previous_home_dir = fs.storage_directory()
    fs.set_hypothesis_home_dir(None)

def teardown_function(function):
    if False:
        while True:
            i = 10
    global previous_home_dir
    fs.set_hypothesis_home_dir(previous_home_dir)
    previous_home_dir = None

def test_defaults_to_the_default():
    if False:
        for i in range(10):
            print('nop')
    assert fs.storage_directory() == fs.__hypothesis_home_directory_default

def test_can_set_homedir_and_it_will_exist(tmpdir):
    if False:
        print('Hello World!')
    fs.set_hypothesis_home_dir(str(tmpdir.mkdir('kittens')))
    d = fs.storage_directory()
    assert 'kittens' in str(d)
    assert d.exists()

def test_will_pick_up_location_from_env(monkeypatch, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(os, 'environ', {'HYPOTHESIS_STORAGE_DIRECTORY': str(tmpdir)})
    assert fs.storage_directory() == tmpdir

def test_storage_directories_are_not_created_automatically(tmpdir):
    if False:
        print('Hello World!')
    fs.set_hypothesis_home_dir(str(tmpdir))
    assert not fs.storage_directory('badgers').exists()