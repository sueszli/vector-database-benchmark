import pytest
import libqtile.bar
import libqtile.config
import libqtile.layout
from libqtile.confreader import Config
from libqtile.extension.window_list import WindowList
from libqtile.lazy import lazy

@pytest.fixture
def extension_manager(monkeypatch, manager_nospawn):
    if False:
        print('Hello World!')
    extension = WindowList()

    def fake_popen(cmd, *args, **kwargs):
        if False:
            while True:
                i = 10

        class PopenObj:

            def communicate(self, value_in, *args):
                if False:
                    for i in range(10):
                        print('nop')
                return [value_in, None]
        return PopenObj()
    monkeypatch.setattr('libqtile.extension.base.Popen', fake_popen)

    class ManagerConfig(Config):
        groups = [libqtile.config.Group('a'), libqtile.config.Group('b')]
        layouts = [libqtile.layout.max.Max()]
        keys = [libqtile.config.Key(['control'], 'k', lazy.run_extension(extension))]
        screens = [libqtile.config.Screen(bottom=libqtile.bar.Bar([], 20))]
    manager_nospawn.start(ManagerConfig)
    yield manager_nospawn

def test_window_list(extension_manager):
    if False:
        return 10
    'Test WindowList extension switches group.'
    extension_manager.test_window('one')
    assert len(extension_manager.c.group.info()['windows']) == 1
    extension_manager.c.group['b'].toscreen()
    assert len(extension_manager.c.group.info()['windows']) == 0
    extension_manager.c.simulate_keypress(['control'], 'k')
    assert len(extension_manager.c.group.info()['windows']) == 1
    assert extension_manager.c.group.info()['label'] == 'a'