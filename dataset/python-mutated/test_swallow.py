import pytest
import libqtile
from libqtile import config
from libqtile.backend.x11.core import Core
from libqtile.confreader import Config
from libqtile.lazy import lazy

@lazy.function
def swallow_inc(qtile):
    if False:
        print('Hello World!')
    qtile.test_data += 1
    return True

class SwallowConfig(Config):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

        @libqtile.hook.subscribe.startup
        def _():
            if False:
                i = 10
                return i + 15
            libqtile.qtile.test_data = 0
    keys = [config.Key(['control'], 'k', swallow_inc()), config.Key(['control'], 'j', swallow_inc(), swallow=False), config.Key(['control'], 'i', swallow_inc().when(layout='idonotexist')), config.Key(['control'], 'o', swallow_inc().when(layout='idonotexist'), swallow_inc())]

def send_process_key_event(manager, key):
    if False:
        i = 10
        return i + 15
    (keysym, mask) = Core.lookup_key(None, key)
    output = manager.c.eval(f'self.process_key_event({keysym}, {mask})[1]')
    assert output[0]
    return output[1] == 'True'

def get_test_counter(manager):
    if False:
        while True:
            i = 10
    output = manager.c.eval('self.test_data')
    assert output[0]
    return int(output[1])

@pytest.mark.parametrize('manager', [SwallowConfig], indirect=True)
def test_swallow(manager):
    if False:
        print('Hello World!')
    expectedexecuted = [True, True, False, True]
    expectedswallow = [True, False, False, True]
    prev_counter = 0
    for (index, key) in enumerate(SwallowConfig.keys):
        assert send_process_key_event(manager, key) == expectedswallow[index]
        counter = get_test_counter(manager)
        if expectedexecuted[index]:
            assert counter > prev_counter
        else:
            assert counter == prev_counter
        prev_counter = counter
    not_used_key = config.Key(['control'], 'h', swallow_inc())
    assert not send_process_key_event(manager, not_used_key)
    assert get_test_counter(manager) == prev_counter