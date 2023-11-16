from __future__ import annotations
import pytest
from ansible.utils.display import Display

@pytest.fixture
def warning_message():
    if False:
        print('Hello World!')
    warning_message = 'bad things will happen'
    expected_warning_message = '[WARNING]: {0}\n'.format(warning_message)
    return (warning_message, expected_warning_message)

def test_warning(capsys, mocker, warning_message):
    if False:
        while True:
            i = 10
    (warning_message, expected_warning_message) = warning_message
    mocker.patch('ansible.utils.color.ANSIBLE_COLOR', True)
    mocker.patch('ansible.utils.color.parsecolor', return_value=u'1;35')
    d = Display()
    d.warning(warning_message)
    (out, err) = capsys.readouterr()
    assert d._warns == {expected_warning_message: 1}
    assert err == '\x1b[1;35m{0}\x1b[0m\n'.format(expected_warning_message.rstrip('\n'))

def test_warning_no_color(capsys, mocker, warning_message):
    if False:
        i = 10
        return i + 15
    (warning_message, expected_warning_message) = warning_message
    mocker.patch('ansible.utils.color.ANSIBLE_COLOR', False)
    d = Display()
    d.warning(warning_message)
    (out, err) = capsys.readouterr()
    assert d._warns == {expected_warning_message: 1}
    assert err == expected_warning_message