"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>

    Test cases for salt.modules.gnomedesktop
"""
import pytest
import salt.modules.gnomedesktop as gnomedesktop
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {gnomedesktop: {}}

def test_ping():
    if False:
        while True:
            i = 10
    '\n    Test for A test to ensure the GNOME module is loaded\n    '
    assert gnomedesktop.ping()

def test_getidledelay():
    if False:
        i = 10
        return i + 15
    '\n    Test for Return the current idle delay setting in seconds\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        with patch.object(gsettings_mock, '_get', return_value=True):
            assert gnomedesktop.getIdleDelay()

def test_setidledelay():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Set the current idle delay setting in seconds\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        with patch.object(gsettings_mock, '_set', return_value=True):
            assert gnomedesktop.setIdleDelay(5)

def test_getclockformat():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Return the current clock format, either 12h or 24h format.\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        with patch.object(gsettings_mock, '_get', return_value=True):
            assert gnomedesktop.getClockFormat()

def test_setclockformat():
    if False:
        return 10
    '\n    Test for Set the clock format, either 12h or 24h format..\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        with patch.object(gsettings_mock, '_set', return_value=True):
            assert gnomedesktop.setClockFormat('12h')
        assert not gnomedesktop.setClockFormat('a')

def test_getclockshowdate():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Return the current setting, if the date is shown in the clock\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        with patch.object(gsettings_mock, '_get', return_value=True):
            assert gnomedesktop.getClockShowDate()

def test_setclockshowdate():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Set whether the date is visible in the clock\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        assert not gnomedesktop.setClockShowDate('kvalue')
        with patch.object(gsettings_mock, '_get', return_value=True):
            assert gnomedesktop.setClockShowDate(True)

def test_getidleactivation():
    if False:
        while True:
            i = 10
    '\n    Test for Get whether the idle activation is enabled\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        with patch.object(gsettings_mock, '_get', return_value=True):
            assert gnomedesktop.getIdleActivation()

def test_setidleactivation():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Set whether the idle activation is enabled\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        assert not gnomedesktop.setIdleActivation('kvalue')
        with patch.object(gsettings_mock, '_set', return_value=True):
            assert gnomedesktop.setIdleActivation(True)

def test_get():
    if False:
        i = 10
        return i + 15
    '\n    Test for Get key in a particular GNOME schema\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        with patch.object(gsettings_mock, '_get', return_value=True):
            assert gnomedesktop.get()

def test_set_():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Set key in a particular GNOME schema.\n    '
    with patch('salt.modules.gnomedesktop._GSettings') as gsettings_mock:
        with patch.object(gsettings_mock, '_get', return_value=True):
            assert gnomedesktop.set_()