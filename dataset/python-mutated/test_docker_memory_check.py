import os
from unittest import mock
import pytest
from tools import docker_memory_check

@pytest.mark.parametrize(('option', 'expected'), (('always', True), ('never', False)))
def test_should_use_color_forced(option, expected):
    if False:
        return 10
    assert docker_memory_check.should_use_color(option) is expected

def test_should_use_color_determined_by_CI_variable_missing():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.dict(os.environ, clear=True):
        assert docker_memory_check.should_use_color('auto') is True

def test_should_use_color_determined_by_CI_variable_empty():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.dict(os.environ, {'CI': ''}):
        assert docker_memory_check.should_use_color('auto') is True

def test_should_use_color_determined_by_CI_variable_present():
    if False:
        while True:
            i = 10
    with mock.patch.dict(os.environ, {'CI': ''}):
        assert docker_memory_check.should_use_color('1') is False

def test_color_using_color():
    if False:
        return 10
    ret = docker_memory_check.color('hello hello', '\x1b[33m', use_color=True)
    assert ret == '\x1b[33mhello hello\x1b[m'

def test_color_not_using_color():
    if False:
        print('Hello World!')
    ret = docker_memory_check.color('hello hello', '\x1b[33m', use_color=False)
    assert ret == 'hello hello'

def test_check_ignored_file_does_not_exist(capsys, tmp_path):
    if False:
        i = 10
        return i + 15
    json_file = tmp_path.joinpath('settings.json')
    assert docker_memory_check.main(('--settings-file', str(json_file))) == 0
    (out, err) = capsys.readouterr()
    assert out == err == ''

def test_check_ignored_file_is_not_json(capsys, tmp_path):
    if False:
        while True:
            i = 10
    json_file = tmp_path.joinpath('settings.json')
    json_file.write_text('not json')
    assert docker_memory_check.main(('--settings-file', str(json_file))) == 0
    (out, err) = capsys.readouterr()
    assert out == err == ''

def test_check_ignored_file_missing_field(capsys, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    json_file = tmp_path.joinpath('settings.json')
    json_file.write_text('{}')
    assert docker_memory_check.main(('--settings-file', str(json_file))) == 0
    (out, err) = capsys.readouterr()
    assert out == err == ''

def test_check_ignored_memory_limit_not_int(capsys, tmp_path):
    if False:
        print('Hello World!')
    json_file = tmp_path.joinpath('settings.json')
    json_file.write_text('{"memoryMiB": "lots"}')
    assert docker_memory_check.main(('--settings-file', str(json_file))) == 0
    (out, err) = capsys.readouterr()
    assert out == err == ''

def test_check_has_enough_configured_memory(capsys, tmp_path):
    if False:
        i = 10
        return i + 15
    json_file = tmp_path.joinpath('settings.json')
    json_file.write_text('{"memoryMiB": 9001}')
    args = ('--settings-file', str(json_file), '--memory-minimum', '8092')
    assert docker_memory_check.main(args) == 0
    (out, err) = capsys.readouterr()
    assert out == err == ''

def test_check_insufficient_configured_memory(capsys, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    json_file = tmp_path.joinpath('settings.json')
    json_file.write_text('{"memoryMiB": 7000}')
    args = ('--settings-file', str(json_file), '--memory-minimum=8092', '--color=never')
    assert docker_memory_check.main(args) == 0
    (out, err) = capsys.readouterr()
    assert out == ''
    assert err == 'WARNING: docker is configured with less than the recommended minimum memory!\n- open Docker.app and adjust the memory in Settings -> Resources\n- current memory (MiB): 7000\n- recommended minimum (MiB): 8092\n'