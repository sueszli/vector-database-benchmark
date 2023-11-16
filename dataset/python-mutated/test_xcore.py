import pytest
from libqtile.backend import get_core
from libqtile.backend.x11 import core
from test.test_manager import ManagerConfig

def test_get_core_x11(display):
    if False:
        print('Hello World!')
    get_core('x11', display).finalize()

def test_keys(display):
    if False:
        return 10
    assert 'a' in core.get_keys()
    assert 'shift' in core.get_modifiers()

def test_no_two_qtiles(xmanager):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core.ExistingWMException):
        core.Core(xmanager.display).finalize()

def test_color_pixel(xmanager):
    if False:
        i = 10
        return i + 15
    (success, e) = xmanager.c.eval('self.core.conn.color_pixel("ffffff")')
    assert success, e

@pytest.mark.parametrize('xmanager', [ManagerConfig], indirect=True)
def test_net_client_list(xmanager, conn):
    if False:
        print('Hello World!')

    def assert_clients(number):
        if False:
            i = 10
            return i + 15
        clients = conn.default_screen.root.get_property('_NET_CLIENT_LIST', unpack=int)
        assert len(clients) == number
    xmanager.c.eval('self.core.update_client_lists()')
    assert_clients(0)
    one = xmanager.test_window('one')
    assert_clients(1)
    two = xmanager.test_window('two')
    xmanager.c.window.toggle_minimize()
    three = xmanager.test_window('three')
    xmanager.c.screen.next_group()
    assert_clients(3)
    xmanager.kill_window(one)
    xmanager.c.screen.next_group()
    assert_clients(2)
    xmanager.kill_window(three)
    assert_clients(1)
    xmanager.c.screen.next_group()
    one = xmanager.test_window('one')
    assert_clients(2)
    xmanager.c.window.static()
    assert_clients(1)
    xmanager.kill_window(two)
    assert_clients(0)