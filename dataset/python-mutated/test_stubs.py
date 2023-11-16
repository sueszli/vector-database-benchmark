"""Test test stubs."""
from unittest import mock
import pytest

@pytest.fixture
def timer(stubs):
    if False:
        while True:
            i = 10
    return stubs.FakeTimer()

def test_timeout(timer):
    if False:
        for i in range(10):
            print('nop')
    'Test whether timeout calls the functions.'
    func = mock.Mock()
    func2 = mock.Mock()
    timer.timeout.connect(func)
    timer.timeout.connect(func2)
    func.assert_not_called()
    func2.assert_not_called()
    timer.timeout.emit()
    func.assert_called_once_with()
    func2.assert_called_once_with()

def test_disconnect_all(timer):
    if False:
        print('Hello World!')
    'Test disconnect without arguments.'
    func = mock.Mock()
    timer.timeout.connect(func)
    timer.timeout.disconnect()
    timer.timeout.emit()
    func.assert_not_called()

def test_disconnect_one(timer):
    if False:
        for i in range(10):
            print('nop')
    'Test disconnect with a single argument.'
    func = mock.Mock()
    timer.timeout.connect(func)
    timer.timeout.disconnect(func)
    timer.timeout.emit()
    func.assert_not_called()

def test_disconnect_all_invalid(timer):
    if False:
        print('Hello World!')
    'Test disconnecting with no connections.'
    with pytest.raises(TypeError):
        timer.timeout.disconnect()

def test_disconnect_one_invalid(timer):
    if False:
        i = 10
        return i + 15
    'Test disconnecting with an invalid connection.'
    func1 = mock.Mock()
    func2 = mock.Mock()
    timer.timeout.connect(func1)
    with pytest.raises(TypeError):
        timer.timeout.disconnect(func2)
    func1.assert_not_called()
    func2.assert_not_called()
    timer.timeout.emit()
    func1.assert_called_once_with()

def test_singleshot(timer):
    if False:
        return 10
    'Test setting singleShot.'
    assert not timer.isSingleShot()
    timer.setSingleShot(True)
    assert timer.isSingleShot()
    timer.start()
    assert timer.isActive()
    timer.timeout.emit()
    assert not timer.isActive()

def test_active(timer):
    if False:
        print('Hello World!')
    'Test isActive.'
    assert not timer.isActive()
    timer.start()
    assert timer.isActive()
    timer.stop()
    assert not timer.isActive()

def test_interval(timer):
    if False:
        i = 10
        return i + 15
    'Test setting an interval.'
    assert timer.interval() == 0
    timer.setInterval(1000)
    assert timer.interval() == 1000