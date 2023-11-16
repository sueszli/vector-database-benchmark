import pytest
import libqtile.bar
import libqtile.config
import libqtile.confreader
import libqtile.layout
import libqtile.widget as widgets
from libqtile.widget.base import ORIENTATION_BOTH, ORIENTATION_VERTICAL
from libqtile.widget.clock import Clock
from libqtile.widget.crashme import _CrashMe
from test.widgets.conftest import FakeBar
overrides = []
extras = [(_CrashMe, {})]
no_test = [widgets.Mirror, widgets.PulseVolume]
exclusive_backend = {widgets.Systray: 'x11'}
parameters = [(getattr(widgets, w), {'dummy_parameter': 1}) for w in widgets.__all__]
for ovr in overrides:
    parameters = [ovr if ovr[0] == w[0] else w for w in parameters]
parameters.extend(extras)
for skipped in no_test:
    parameters = [w for w in parameters if w[0] != skipped]

def no_op(*args, **kwargs):
    if False:
        return 10
    pass

@pytest.mark.parametrize('widget_class,kwargs', parameters)
def test_widget_init_config(manager_nospawn, minimal_conf_noscreen, widget_class, kwargs):
    if False:
        return 10
    if widget_class in exclusive_backend:
        if exclusive_backend[widget_class] != manager_nospawn.backend.name:
            pytest.skip('Unsupported backend')
    widget = widget_class(**kwargs)
    widget.draw = no_op
    for (k, v) in kwargs.items():
        assert getattr(widget, k) == v
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([widget], 10))]
    manager_nospawn.start(config)
    i = manager_nospawn.c.bar['top'].info()
    assert i['widgets'][0]['name'] == widget.name

@pytest.mark.parametrize('widget_class,kwargs', [param for param in parameters if param[0]().orientations in [ORIENTATION_BOTH, ORIENTATION_VERTICAL]])
def test_widget_init_config_vertical_bar(manager_nospawn, minimal_conf_noscreen, widget_class, kwargs):
    if False:
        print('Hello World!')
    if widget_class in exclusive_backend:
        if exclusive_backend[widget_class] != manager_nospawn.backend.name:
            pytest.skip('Unsupported backend')
    widget = widget_class(**kwargs)
    widget.draw = no_op
    for (k, v) in kwargs.items():
        assert getattr(widget, k) == v
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(left=libqtile.bar.Bar([widget], 10))]
    manager_nospawn.start(config)
    i = manager_nospawn.c.bar['left'].info()
    assert i['widgets'][0]['name'] == widget.name

@pytest.mark.parametrize('widget_class,kwargs', parameters)
def test_widget_init_config_set_width(widget_class, kwargs):
    if False:
        for i in range(10):
            print('nop')
    widget = widget_class(width=50)
    assert widget

def test_incompatible_orientation(fake_qtile, fake_window):
    if False:
        print('Hello World!')
    clk1 = Clock()
    clk1.orientations = ORIENTATION_VERTICAL
    fakebar = FakeBar([clk1], window=fake_window)
    with pytest.raises(libqtile.confreader.ConfigError):
        clk1._configure(fake_qtile, fakebar)