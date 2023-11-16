import sys
from importlib import reload
from types import ModuleType
import pytest
import libqtile.config
from libqtile.bar import Bar

def no_op(*args, **kwargs):
    if False:
        return 10
    pass

class MockIwlib(ModuleType):
    DATA = {'wlan0': {'NWID': b'Auto', 'Frequency': b'5.18 GHz', 'Access Point': b'12:34:56:78:90:AB', 'BitRate': b'650 Mb/s', 'ESSID': b'QtileNet', 'Mode': b'Managed', 'stats': {'quality': 49, 'level': 190, 'noise': 0, 'updated': 75}}}

    @classmethod
    def get_iwconfig(cls, interface):
        if False:
            print('Hello World!')
        return cls.DATA.get(interface, dict())

@pytest.fixture
def patched_wlan(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setitem(sys.modules, 'iwlib', MockIwlib('iwlib'))
    from libqtile.widget import wlan
    reload(wlan)
    yield wlan

@pytest.mark.parametrize('kwargs,expected', [({}, 'QtileNet 49/70'), ({'format': '{essid} {percent:2.0%}'}, 'QtileNet 70%'), ({'interface': 'wlan1'}, 'Disconnected')])
def test_wlan_display(minimal_conf_noscreen, manager_nospawn, patched_wlan, kwargs, expected):
    if False:
        print('Hello World!')
    widget = patched_wlan.Wlan(**kwargs)
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=Bar([widget], 10))]
    manager_nospawn.start(config)
    text = manager_nospawn.c.bar['top'].info()['widgets'][0]['text']
    assert text == expected