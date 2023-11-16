import asyncio
from multiprocessing import Value
import pytest
import libqtile.log_utils
import libqtile.utils
from libqtile import hook
from libqtile.resources import default_config
from test.conftest import BareConfig

class Call:

    def __init__(self, val):
        if False:
            while True:
                i = 10
        self.val = val

    def __call__(self, val):
        if False:
            for i in range(10):
                print('nop')
        self.val = val

class NoArgCall(Call):

    def __call__(self):
        if False:
            while True:
                i = 10
        self.val += 1

@pytest.fixture
def hook_fixture():
    if False:
        i = 10
        return i + 15
    libqtile.log_utils.init_log()
    yield
    hook.clear()

def test_cannot_fire_unknown_event():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(libqtile.utils.QtileError):
        hook.fire('unknown')

@pytest.mark.usefixtures('hook_fixture')
def test_hook_calls_subscriber():
    if False:
        return 10
    test = Call(0)
    hook.subscribe.group_window_add(test)
    hook.fire('group_window_add', 8)
    assert test.val == 8

@pytest.mark.usefixtures('hook_fixture')
def test_hook_calls_subscriber_async():
    if False:
        while True:
            i = 10
    val = 0

    async def co(new_val):
        nonlocal val
        val = new_val
    hook.subscribe.group_window_add(co)
    hook.fire('group_window_add', 8)
    assert val == 8

@pytest.mark.usefixtures('hook_fixture')
def test_hook_calls_subscriber_async_co():
    if False:
        while True:
            i = 10
    val = 0

    async def co(new_val):
        nonlocal val
        val = new_val
    hook.subscribe.group_window_add(co(8))
    hook.fire('group_window_add')
    assert val == 8

@pytest.mark.usefixtures('hook_fixture')
def test_hook_calls_subscriber_async_in_existing_loop():
    if False:
        for i in range(10):
            print('nop')

    async def t():
        val = 0

        async def co(new_val):
            nonlocal val
            val = new_val
        hook.subscribe.group_window_add(co(8))
        hook.fire('group_window_add')
        await asyncio.sleep(0)
        assert val == 8
    asyncio.run(t())

@pytest.mark.usefixtures('hook_fixture')
def test_subscribers_can_be_added_removed():
    if False:
        return 10
    test = Call(0)
    hook.subscribe.group_window_add(test)
    assert hook.subscriptions
    hook.clear()
    assert not hook.subscriptions

@pytest.mark.usefixtures('hook_fixture')
def test_can_unsubscribe_from_hook():
    if False:
        return 10
    test = Call(0)
    hook.subscribe.group_window_add(test)
    hook.fire('group_window_add', 3)
    assert test.val == 3
    hook.unsubscribe.group_window_add(test)
    hook.fire('group_window_add', 4)
    assert test.val == 3

def test_can_subscribe_to_startup_hooks(manager_nospawn):
    if False:
        for i in range(10):
            print('nop')
    config = BareConfig
    for attr in dir(default_config):
        if not hasattr(config, attr):
            setattr(config, attr, getattr(default_config, attr))
    manager = manager_nospawn
    manager.startup_once_calls = Value('i', 0)
    manager.startup_calls = Value('i', 0)
    manager.startup_complete_calls = Value('i', 0)

    def inc_startup_once_calls():
        if False:
            i = 10
            return i + 15
        manager.startup_once_calls.value += 1

    def inc_startup_calls():
        if False:
            i = 10
            return i + 15
        manager.startup_calls.value += 1

    def inc_startup_complete_calls():
        if False:
            print('Hello World!')
        manager.startup_complete_calls.value += 1
    hook.subscribe.startup_once(inc_startup_once_calls)
    hook.subscribe.startup(inc_startup_calls)
    hook.subscribe.startup_complete(inc_startup_complete_calls)
    manager.start(config)
    assert manager.startup_once_calls.value == 1
    assert manager.startup_calls.value == 1
    assert manager.startup_complete_calls.value == 1
    manager.terminate()
    manager.start(config, no_spawn=True)
    assert manager.startup_once_calls.value == 1
    assert manager.startup_calls.value == 2
    assert manager.startup_complete_calls.value == 2

@pytest.mark.usefixtures('hook_fixture')
def test_can_update_by_selection_change(manager):
    if False:
        i = 10
        return i + 15
    test = Call(0)
    hook.subscribe.selection_change(test)
    hook.fire('selection_change', 'hello')
    assert test.val == 'hello'

@pytest.mark.usefixtures('hook_fixture')
def test_can_call_by_selection_notify(manager):
    if False:
        print('Hello World!')
    test = Call(0)
    hook.subscribe.selection_notify(test)
    hook.fire('selection_notify', 'hello')
    assert test.val == 'hello'

@pytest.mark.usefixtures('hook_fixture')
def test_resume_hook(manager):
    if False:
        print('Hello World!')
    test = NoArgCall(0)
    hook.subscribe.resume(test)
    hook.fire('resume')
    assert test.val == 1

@pytest.mark.usefixtures('hook_fixture')
def test_custom_hook_registry():
    if False:
        for i in range(10):
            print('nop')
    'Tests ability to create custom hook registries'
    test = NoArgCall(0)
    custom = hook.Registry('test')
    custom.register_hook(hook.Hook('test_hook'))
    custom.subscribe.test_hook(test)
    assert test.val == 0
    custom.fire('test_hook')
    assert test.val == 1
    with pytest.raises(libqtile.utils.QtileError):
        custom.fire('client_managed')
    with pytest.raises(libqtile.utils.QtileError):
        hook.fire('test_hook')

@pytest.mark.usefixtures('hook_fixture')
def test_user_hook(manager_nospawn):
    if False:
        return 10
    config = BareConfig
    for attr in dir(default_config):
        if not hasattr(config, attr):
            setattr(config, attr, getattr(default_config, attr))
    manager = manager_nospawn
    manager.custom_no_arg_text = Value('u', 'A')
    manager.custom_text = Value('u', 'A')

    def predefined_text():
        if False:
            print('Hello World!')
        with manager.custom_no_arg_text.get_lock():
            manager.custom_no_arg_text.value = 'B'

    def defined_text(text):
        if False:
            i = 10
            return i + 15
        with manager.custom_text.get_lock():
            manager.custom_text.value = text
    hook.subscribe.user('set_text')(predefined_text)
    hook.subscribe.user('define_text')(defined_text)
    manager.start(config)
    assert manager.custom_no_arg_text.value == 'A'
    assert manager.custom_text.value == 'A'
    manager.c.fire_user_hook('set_text')
    assert manager.custom_no_arg_text.value == 'B'
    manager.c.fire_user_hook('define_text', 'C')
    assert manager.custom_text.value == 'C'