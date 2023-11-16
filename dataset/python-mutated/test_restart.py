import pickle
import shutil
import textwrap
from multiprocessing import Value
import pytest
import libqtile.bar
import libqtile.config
import libqtile.layout
from libqtile import config, hook, layout
from libqtile.confreader import Config
from libqtile.ipc import IPCError
from libqtile.lazy import lazy
from libqtile.resources import default_config
from libqtile.widget import TextBox
from test.helpers import TestManager as BareManager

class TwoScreenConfig(Config):
    auto_fullscreen = True
    groups = [config.Group('a'), config.Group('b'), config.Group('c'), config.Group('d')]
    layouts = [layout.stack.Stack(num_stacks=1), layout.stack.Stack(num_stacks=2)]
    floating_layout = default_config.floating_layout
    keys = [config.Key(['control'], 'k', lazy.layout.up()), config.Key(['control'], 'j', lazy.layout.down())]
    mouse = []
    follow_mouse_focus = False
    reconfigure_screens = False
    screens = []
    fake_screens = [libqtile.config.Screen(top=libqtile.bar.Bar([TextBox('Qtile Test')], 10), x=0, y=0, width=400, height=600), libqtile.config.Screen(top=libqtile.bar.Bar([TextBox('Qtile Test')], 10), x=400, y=0, width=400, height=600)]

def test_restart_hook_and_state(manager_nospawn, request, backend, backend_name):
    if False:
        i = 10
        return i + 15
    if backend_name == 'wayland':
        pytest.skip('Skipping test on Wayland.')
    manager = manager_nospawn
    inject = textwrap.dedent('\n        from libqtile.core.lifecycle import lifecycle\n\n        def no_op(*args, **kwargs):\n            pass\n\n        self.lifecycle = lifecycle\n        self._do_stop = self._stop\n        self._stop = no_op\n        ')

    def inc_restart_call():
        if False:
            for i in range(10):
                print('nop')
        manager.restart_calls.value += 1
    manager.restart_calls = Value('i', 0)
    hook.subscribe.restart(inc_restart_call)
    manager.start(TwoScreenConfig)
    assert manager.restart_calls.value == 0
    manager.c.group['c'].toscreen(0)
    manager.c.group['d'].toscreen(1)
    manager.test_window('one')
    manager.test_window('two')
    wins = {w['name']: w['id'] for w in manager.c.windows()}
    manager.c.window[wins['one']].togroup('c')
    manager.c.window[wins['two']].togroup('d')
    manager.c.eval(inject)
    manager.c.restart()
    assert manager.restart_calls.value == 1
    (_, state_file) = manager.c.eval('self.lifecycle.state_file')
    assert state_file
    original_state = f'{state_file}-original'
    shutil.copy(state_file, original_state)
    manager.c.eval('self._do_stop()')
    with pytest.raises((IPCError, ConnectionResetError)):
        assert manager.c.status()
    with BareManager(backend, request.config.getoption('--debuglog')) as restarted_manager:
        restarted_manager.start(TwoScreenConfig, state=state_file)
        screen0_info = restarted_manager.c.screen[0].group.info()
        assert screen0_info['name'] == 'c'
        assert screen0_info['screen'] == 0
        screen1_info = restarted_manager.c.screen[1].group.info()
        assert screen1_info['name'] == 'd'
        assert screen1_info['screen'] == 1
        assert len(restarted_manager.c.windows()) == 2
        name_to_group = {w['name']: w['group'] for w in restarted_manager.c.windows()}
        assert name_to_group['one'] == 'c'
        assert name_to_group['two'] == 'd'
        restarted_manager.c.eval(inject)
        restarted_manager.c.restart()
        (_, restarted_state) = restarted_manager.c.eval('self.lifecycle.state_file')
        assert restarted_state
        restarted_manager.c.eval('self._do_stop()')
    with open(original_state, 'rb') as f:
        original = pickle.load(f)
    with open(restarted_state, 'rb') as f:
        restarted = pickle.load(f)
    assert original.groups == restarted.groups
    assert original.screens == restarted.screens
    assert original.current_screen == restarted.current_screen
    assert original.scratchpads == restarted.scratchpads