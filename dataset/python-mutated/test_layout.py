import asyncio
import gc
import random
import re
from weakref import finalize
from weakref import ref as weakref
import pytest
import reactpy
from reactpy import html
from reactpy.config import REACTPY_DEBUG_MODE
from reactpy.core.component import component
from reactpy.core.hooks import use_effect, use_state
from reactpy.core.layout import Layout
from reactpy.core.types import State
from reactpy.testing import HookCatcher, StaticEventHandler, assert_reactpy_did_log, capture_reactpy_logs
from reactpy.utils import Ref
from tests.tooling import select
from tests.tooling.common import event_message, update_message
from tests.tooling.hooks import use_force_render, use_toggle
from tests.tooling.layout import layout_runner
from tests.tooling.select import element_exists, find_element

@pytest.fixture(autouse=True)
def no_logged_errors():
    if False:
        for i in range(10):
            print('nop')
    with capture_reactpy_logs() as logs:
        yield
        for record in logs:
            if record.exc_info:
                raise record.exc_info[1]

def test_layout_repr():
    if False:
        print('Hello World!')

    @reactpy.component
    def MyComponent():
        if False:
            i = 10
            return i + 15
        ...
    my_component = MyComponent()
    layout = reactpy.Layout(my_component)
    assert str(layout) == f'Layout(MyComponent({id(my_component):02x}))'

def test_layout_expects_abstract_component():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError, match='Expected a ComponentType'):
        reactpy.Layout(None)
    with pytest.raises(TypeError, match='Expected a ComponentType'):
        reactpy.Layout(reactpy.html.div())

async def test_layout_cannot_be_used_outside_context_manager(caplog):

    @reactpy.component
    def Component():
        if False:
            for i in range(10):
                print('nop')
        ...
    component = Component()
    layout = reactpy.Layout(component)
    with pytest.raises(AttributeError):
        await layout.deliver(event_message('something'))
    with pytest.raises(AttributeError):
        await layout.render()

async def test_simple_layout():
    set_state_hook = reactpy.Ref()

    @reactpy.component
    def SimpleComponent():
        if False:
            return 10
        (tag, set_state_hook.current) = reactpy.hooks.use_state('div')
        return reactpy.vdom(tag)
    async with reactpy.Layout(SimpleComponent()) as layout:
        update_1 = await layout.render()
        assert update_1 == update_message(path='', model={'tagName': '', 'children': [{'tagName': 'div'}]})
        set_state_hook.current('table')
        update_2 = await layout.render()
        assert update_2 == update_message(path='', model={'tagName': '', 'children': [{'tagName': 'table'}]})

async def test_component_can_return_none():

    @reactpy.component
    def SomeComponent():
        if False:
            print('Hello World!')
        return None
    async with reactpy.Layout(SomeComponent()) as layout:
        assert (await layout.render())['model'] == {'tagName': ''}

async def test_nested_component_layout():
    parent_set_state = reactpy.Ref(None)
    child_set_state = reactpy.Ref(None)

    @reactpy.component
    def Parent():
        if False:
            i = 10
            return i + 15
        (state, parent_set_state.current) = reactpy.hooks.use_state(0)
        return reactpy.html.div(state, Child())

    @reactpy.component
    def Child():
        if False:
            return 10
        (state, child_set_state.current) = reactpy.hooks.use_state(0)
        return reactpy.html.div(state)

    def make_parent_model(state, model):
        if False:
            print('Hello World!')
        return {'tagName': '', 'children': [{'tagName': 'div', 'children': [str(state), model]}]}

    def make_child_model(state):
        if False:
            while True:
                i = 10
        return {'tagName': '', 'children': [{'tagName': 'div', 'children': [str(state)]}]}
    async with reactpy.Layout(Parent()) as layout:
        update_1 = await layout.render()
        assert update_1 == update_message(path='', model=make_parent_model(0, make_child_model(0)))
        parent_set_state.current(1)
        update_2 = await layout.render()
        assert update_2 == update_message(path='', model=make_parent_model(1, make_child_model(0)))
        child_set_state.current(1)
        update_3 = await layout.render()
        assert update_3 == update_message(path='/children/0/children/1', model=make_child_model(1))

@pytest.mark.skipif(not REACTPY_DEBUG_MODE.current, reason='errors only reported in debug mode')
async def test_layout_render_error_has_partial_update_with_error_message():

    @reactpy.component
    def Main():
        if False:
            for i in range(10):
                print('nop')
        return reactpy.html.div([OkChild(), BadChild(), OkChild()])

    @reactpy.component
    def OkChild():
        if False:
            i = 10
            return i + 15
        return reactpy.html.div(['hello'])

    @reactpy.component
    def BadChild():
        if False:
            return 10
        msg = 'error from bad child'
        raise ValueError(msg)
    with assert_reactpy_did_log(match_error='error from bad child'):
        async with reactpy.Layout(Main()) as layout:
            assert await layout.render() == update_message(path='', model={'tagName': '', 'children': [{'tagName': 'div', 'children': [{'tagName': '', 'children': [{'tagName': 'div', 'children': ['hello']}]}, {'tagName': '', 'error': 'ValueError: error from bad child'}, {'tagName': '', 'children': [{'tagName': 'div', 'children': ['hello']}]}]}]})

@pytest.mark.skipif(REACTPY_DEBUG_MODE.current, reason='errors only reported in debug mode')
async def test_layout_render_error_has_partial_update_without_error_message():

    @reactpy.component
    def Main():
        if False:
            for i in range(10):
                print('nop')
        return reactpy.html.div([OkChild(), BadChild(), OkChild()])

    @reactpy.component
    def OkChild():
        if False:
            print('Hello World!')
        return reactpy.html.div(['hello'])

    @reactpy.component
    def BadChild():
        if False:
            return 10
        msg = 'error from bad child'
        raise ValueError(msg)
    with assert_reactpy_did_log(match_error='error from bad child'):
        async with reactpy.Layout(Main()) as layout:
            assert await layout.render() == update_message(path='', model={'tagName': '', 'children': [{'children': [{'children': [{'children': ['hello'], 'tagName': 'div'}], 'tagName': ''}, {'error': '', 'tagName': ''}, {'children': [{'children': ['hello'], 'tagName': 'div'}], 'tagName': ''}], 'tagName': 'div'}]})

async def test_render_raw_vdom_dict_with_single_component_object_as_children():

    @reactpy.component
    def Main():
        if False:
            print('Hello World!')
        return {'tagName': 'div', 'children': Child()}

    @reactpy.component
    def Child():
        if False:
            while True:
                i = 10
        return {'tagName': 'div', 'children': {'tagName': 'h1'}}
    async with reactpy.Layout(Main()) as layout:
        assert await layout.render() == update_message(path='', model={'tagName': '', 'children': [{'children': [{'children': [{'children': [{'tagName': 'h1'}], 'tagName': 'div'}], 'tagName': ''}], 'tagName': 'div'}]})

async def test_components_are_garbage_collected():
    live_components = set()
    outer_component_hook = HookCatcher()

    def add_to_live_components(constructor):
        if False:
            i = 10
            return i + 15

        def wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            component = constructor(*args, **kwargs)
            component_id = id(component)
            live_components.add(component_id)
            finalize(component, live_components.discard, component_id)
            return component
        return wrapper

    @add_to_live_components
    @reactpy.component
    @outer_component_hook.capture
    def Outer():
        if False:
            print('Hello World!')
        return Inner()

    @add_to_live_components
    @reactpy.component
    def Inner():
        if False:
            while True:
                i = 10
        return reactpy.html.div()
    async with reactpy.Layout(Outer()) as layout:
        await layout.render()
        assert len(live_components) == 2
        last_live_components = live_components.copy()
        outer_component_hook.latest.schedule_render()
        await layout.render()
        assert len(live_components - last_live_components) == 1
    del layout
    del outer_component_hook
    assert not live_components

async def test_root_component_life_cycle_hook_is_garbage_collected():
    live_hooks = set()

    def add_to_live_hooks(constructor):
        if False:
            return 10

        def wrapper(*args, **kwargs):
            if False:
                return 10
            result = constructor(*args, **kwargs)
            hook = reactpy.hooks.current_hook()
            hook_id = id(hook)
            live_hooks.add(hook_id)
            finalize(hook, live_hooks.discard, hook_id)
            return result
        return wrapper

    @reactpy.component
    @add_to_live_hooks
    def Root():
        if False:
            print('Hello World!')
        return reactpy.html.div()
    async with reactpy.Layout(Root()) as layout:
        await layout.render()
        assert len(live_hooks) == 1
    del layout
    assert not live_hooks

async def test_life_cycle_hooks_are_garbage_collected():
    live_hooks = set()
    set_inner_component = None

    def add_to_live_hooks(constructor):
        if False:
            for i in range(10):
                print('nop')

        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            result = constructor(*args, **kwargs)
            hook = reactpy.hooks.current_hook()
            hook_id = id(hook)
            live_hooks.add(hook_id)
            finalize(hook, live_hooks.discard, hook_id)
            return result
        return wrapper

    @reactpy.component
    @add_to_live_hooks
    def Outer():
        if False:
            while True:
                i = 10
        nonlocal set_inner_component
        (inner_component, set_inner_component) = reactpy.hooks.use_state(Inner(key='first'))
        return inner_component

    @reactpy.component
    @add_to_live_hooks
    def Inner():
        if False:
            print('Hello World!')
        return reactpy.html.div()
    async with reactpy.Layout(Outer()) as layout:
        await layout.render()
        assert len(live_hooks) == 2
        last_live_hooks = live_hooks.copy()
        set_inner_component(Inner(key='second'))
        await layout.render()
        assert len(live_hooks - last_live_hooks) == 1
    del layout
    del set_inner_component
    gc.collect()
    assert not live_hooks

async def test_double_updated_component_is_not_double_rendered():
    hook = HookCatcher()
    run_count = reactpy.Ref(0)

    @reactpy.component
    @hook.capture
    def AnyComponent():
        if False:
            print('Hello World!')
        run_count.current += 1
        return reactpy.html.div()
    async with reactpy.Layout(AnyComponent()) as layout:
        await layout.render()
        assert run_count.current == 1
        hook.latest.schedule_render()
        hook.latest.schedule_render()
        await layout.render()
        try:
            await asyncio.wait_for(layout.render(), timeout=0.1)
        except asyncio.TimeoutError:
            pass
        assert run_count.current == 2

async def test_update_path_to_component_that_is_not_direct_child_is_correct():
    hook = HookCatcher()

    @reactpy.component
    def Parent():
        if False:
            i = 10
            return i + 15
        return reactpy.html.div(reactpy.html.div(Child()))

    @reactpy.component
    @hook.capture
    def Child():
        if False:
            print('Hello World!')
        return reactpy.html.div()
    async with reactpy.Layout(Parent()) as layout:
        await layout.render()
        hook.latest.schedule_render()
        update = await layout.render()
        assert update['path'] == '/children/0/children/0/children/0'

async def test_log_on_dispatch_to_missing_event_handler(caplog):

    @reactpy.component
    def SomeComponent():
        if False:
            for i in range(10):
                print('nop')
        return reactpy.html.div()
    async with reactpy.Layout(SomeComponent()) as layout:
        await layout.deliver(event_message('missing'))
    assert re.match("Ignored event - handler 'missing' does not exist or its component unmounted", next(iter(caplog.records)).msg)

async def test_model_key_preserves_callback_identity_for_common_elements(caplog):
    called_good_trigger = reactpy.Ref(False)
    good_handler = StaticEventHandler()
    bad_handler = StaticEventHandler()

    @reactpy.component
    def MyComponent():
        if False:
            print('Hello World!')
        (reverse_children, set_reverse_children) = use_toggle()

        @good_handler.use
        def good_trigger():
            if False:
                while True:
                    i = 10
            called_good_trigger.current = True
            set_reverse_children()

        @bad_handler.use
        def bad_trigger():
            if False:
                for i in range(10):
                    print('nop')
            msg = 'Called bad trigger'
            raise ValueError(msg)
        children = [reactpy.html.button({'on_click': good_trigger, 'id': 'good', 'key': 'good'}, 'good'), reactpy.html.button({'on_click': bad_trigger, 'id': 'bad', 'key': 'bad'}, 'bad')]
        if reverse_children:
            children.reverse()
        return reactpy.html.div(children)
    async with reactpy.Layout(MyComponent()) as layout:
        await layout.render()
        for _i in range(3):
            event = event_message(good_handler.target)
            await layout.deliver(event)
            assert called_good_trigger.current
            called_good_trigger.current = False
            await layout.render()
    assert not caplog.records

async def test_model_key_preserves_callback_identity_for_components():
    called_good_trigger = reactpy.Ref(False)
    good_handler = StaticEventHandler()
    bad_handler = StaticEventHandler()

    @reactpy.component
    def RootComponent():
        if False:
            for i in range(10):
                print('nop')
        (reverse_children, set_reverse_children) = use_toggle()
        children = [Trigger(set_reverse_children, name=name, key=name) for name in ['good', 'bad']]
        if reverse_children:
            children.reverse()
        return reactpy.html.div(children)

    @reactpy.component
    def Trigger(set_reverse_children, name):
        if False:
            return 10
        if name == 'good':

            @good_handler.use
            def callback():
                if False:
                    i = 10
                    return i + 15
                called_good_trigger.current = True
                set_reverse_children()
        else:

            @bad_handler.use
            def callback():
                if False:
                    i = 10
                    return i + 15
                msg = 'Called bad trigger'
                raise ValueError(msg)
        return reactpy.html.button({'on_click': callback, 'id': 'good'}, 'good')
    async with reactpy.Layout(RootComponent()) as layout:
        await layout.render()
        for _ in range(3):
            event = event_message(good_handler.target)
            await layout.deliver(event)
            assert called_good_trigger.current
            called_good_trigger.current = False
            await layout.render()

async def test_component_can_return_another_component_directly():

    @reactpy.component
    def Outer():
        if False:
            for i in range(10):
                print('nop')
        return Inner()

    @reactpy.component
    def Inner():
        if False:
            print('Hello World!')
        return reactpy.html.div('hello')
    async with reactpy.Layout(Outer()) as layout:
        assert await layout.render() == update_message(path='', model={'tagName': '', 'children': [{'children': [{'children': ['hello'], 'tagName': 'div'}], 'tagName': ''}]})

async def test_hooks_for_keyed_components_get_garbage_collected():
    pop_item = reactpy.Ref(None)
    garbage_collect_items = []
    registered_finalizers = set()

    @reactpy.component
    def Outer():
        if False:
            print('Hello World!')
        (items, set_items) = reactpy.hooks.use_state([1, 2, 3])
        pop_item.current = lambda : set_items(items[:-1])
        return reactpy.html.div((Inner(key=k, finalizer_id=k) for k in items))

    @reactpy.component
    def Inner(finalizer_id):
        if False:
            print('Hello World!')
        if finalizer_id not in registered_finalizers:
            hook = reactpy.hooks.current_hook()
            finalize(hook, lambda : garbage_collect_items.append(finalizer_id))
            registered_finalizers.add(finalizer_id)
        return reactpy.html.div(finalizer_id)
    async with reactpy.Layout(Outer()) as layout:
        await layout.render()
        pop_item.current()
        await layout.render()
        assert garbage_collect_items == [3]
        pop_item.current()
        await layout.render()
        assert garbage_collect_items == [3, 2]
        pop_item.current()
        await layout.render()
        assert garbage_collect_items == [3, 2, 1]

async def test_event_handler_at_component_root_is_garbage_collected():
    event_handler = reactpy.Ref()

    @reactpy.component
    def HasEventHandlerAtRoot():
        if False:
            print('Hello World!')
        (value, set_value) = reactpy.hooks.use_state(False)
        set_value(not value)
        event_handler.current = weakref(set_value)
        button = reactpy.html.button({'on_click': set_value}, 'state is: ', value)
        event_handler.current = weakref(button['eventHandlers']['on_click'].function)
        return button
    async with reactpy.Layout(HasEventHandlerAtRoot()) as layout:
        await layout.render()
        for _i in range(3):
            last_event_handler = event_handler.current
            await layout.render()
            assert last_event_handler() is None

async def test_event_handler_deep_in_component_layout_is_garbage_collected():
    event_handler = reactpy.Ref()

    @reactpy.component
    def HasNestedEventHandler():
        if False:
            print('Hello World!')
        (value, set_value) = reactpy.hooks.use_state(False)
        set_value(not value)
        event_handler.current = weakref(set_value)
        button = reactpy.html.button({'on_click': set_value}, 'state is: ', value)
        event_handler.current = weakref(button['eventHandlers']['on_click'].function)
        return reactpy.html.div(reactpy.html.div(button))
    async with reactpy.Layout(HasNestedEventHandler()) as layout:
        await layout.render()
        for _i in range(3):
            last_event_handler = event_handler.current
            await layout.render()
            assert last_event_handler() is None

async def test_duplicate_sibling_keys_causes_error(caplog):
    hook = HookCatcher()
    should_error = True

    @reactpy.component
    @hook.capture
    def ComponentReturnsDuplicateKeys():
        if False:
            for i in range(10):
                print('nop')
        if should_error:
            return reactpy.html.div(reactpy.html.div({'key': 'duplicate'}), reactpy.html.div({'key': 'duplicate'}))
        else:
            return reactpy.html.div()
    async with reactpy.Layout(ComponentReturnsDuplicateKeys()) as layout:
        with assert_reactpy_did_log(error_type=ValueError, match_error="Duplicate keys \\['duplicate'\\] at '/children/0'"):
            await layout.render()
        hook.latest.schedule_render()
        should_error = False
        await layout.render()
        should_error = True
        hook.latest.schedule_render()
        with assert_reactpy_did_log(error_type=ValueError, match_error="Duplicate keys \\['duplicate'\\] at '/children/0'"):
            await layout.render()

async def test_keyed_components_preserve_hook_on_parent_update():
    outer_hook = HookCatcher()
    inner_hook = HookCatcher()

    @reactpy.component
    @outer_hook.capture
    def Outer():
        if False:
            print('Hello World!')
        return Inner(key=1)

    @reactpy.component
    @inner_hook.capture
    def Inner():
        if False:
            return 10
        return reactpy.html.div()
    async with reactpy.Layout(Outer()) as layout:
        await layout.render()
        old_inner_hook = inner_hook.latest
        outer_hook.latest.schedule_render()
        await layout.render()
        assert old_inner_hook is inner_hook.latest

async def test_log_error_on_bad_event_handler():
    bad_handler = StaticEventHandler()

    @reactpy.component
    def ComponentWithBadEventHandler():
        if False:
            while True:
                i = 10

        @bad_handler.use
        def raise_error():
            if False:
                i = 10
                return i + 15
            msg = 'bad event handler'
            raise Exception(msg)
        return reactpy.html.button({'on_click': raise_error})
    with assert_reactpy_did_log(match_error='bad event handler'):
        async with reactpy.Layout(ComponentWithBadEventHandler()) as layout:
            await layout.render()
            event = event_message(bad_handler.target)
            await layout.deliver(event)

async def test_schedule_render_from_unmounted_hook():
    parent_set_state = reactpy.Ref()

    @reactpy.component
    def Parent():
        if False:
            i = 10
            return i + 15
        (state, parent_set_state.current) = reactpy.hooks.use_state(1)
        return Child(key=state, state=state)
    child_hook = HookCatcher()

    @reactpy.component
    @child_hook.capture
    def Child(state):
        if False:
            i = 10
            return i + 15
        return reactpy.html.div(state)
    with assert_reactpy_did_log('Did not render component with model state ID .*? - component already unmounted'):
        async with reactpy.Layout(Parent()) as layout:
            await layout.render()
            old_hook = child_hook.latest
            parent_set_state.current(2)
            await layout.render()
            old_hook.schedule_render()
            parent_set_state.current(3)
            await layout.render()

async def test_elements_and_components_with_the_same_key_can_be_interchanged():
    set_toggle = reactpy.Ref()
    effects = []

    @reactpy.component
    def Root():
        if False:
            print('Hello World!')
        (toggle, set_toggle.current) = use_toggle(True)
        if toggle:
            return SomeComponent('x')
        else:
            return reactpy.html.div(SomeComponent('y'))

    @reactpy.component
    def SomeComponent(name):
        if False:
            while True:
                i = 10

        @use_effect
        def some_effect():
            if False:
                for i in range(10):
                    print('nop')
            effects.append('mount ' + name)
            return lambda : effects.append('unmount ' + name)
        return reactpy.html.div(name)
    async with reactpy.Layout(Root()) as layout:
        await layout.render()
        assert effects == ['mount x']
        set_toggle.current()
        await layout.render()
        assert effects == ['mount x', 'unmount x', 'mount y']
        set_toggle.current()
        await layout.render()
        assert effects == ['mount x', 'unmount x', 'mount y', 'unmount y', 'mount x']

async def test_layout_does_not_copy_element_children_by_key():
    set_items = reactpy.Ref()

    @reactpy.component
    def SomeComponent():
        if False:
            return 10
        (items, set_items.current) = reactpy.use_state([1, 2, 3])
        return reactpy.html.div([reactpy.html.div({'key': i}, reactpy.html.input({'on_change': lambda event: None})) for i in items])
    async with reactpy.Layout(SomeComponent()) as layout:
        await layout.render()
        set_items.current([2, 3])
        await layout.render()
        set_items.current([3])
        await layout.render()
        set_items.current([])
        await layout.render()

async def test_changing_key_of_parent_element_unmounts_children():
    random.seed(0)
    root_hook = HookCatcher()
    state = reactpy.Ref(None)

    @reactpy.component
    @root_hook.capture
    def Root():
        if False:
            return 10
        return reactpy.html.div({'key': str(random.random())}, HasState())

    @reactpy.component
    def HasState():
        if False:
            for i in range(10):
                print('nop')
        state.current = reactpy.hooks.use_state(random.random)[0]
        return reactpy.html.div()
    async with reactpy.Layout(Root()) as layout:
        await layout.render()
        for _i in range(5):
            last_state = state.current
            root_hook.latest.schedule_render()
            await layout.render()
            assert last_state != state.current

async def test_switching_node_type_with_event_handlers():
    toggle_type = reactpy.Ref()
    element_static_handler = StaticEventHandler()
    component_static_handler = StaticEventHandler()

    @reactpy.component
    def Root():
        if False:
            while True:
                i = 10
        (toggle, toggle_type.current) = use_toggle(True)
        handler = element_static_handler.use(lambda : None)
        if toggle:
            return html.div(html.button({'on_event': handler}))
        else:
            return html.div(SomeComponent())

    @reactpy.component
    def SomeComponent():
        if False:
            while True:
                i = 10
        handler = component_static_handler.use(lambda : None)
        return html.button({'on_another_event': handler})
    async with reactpy.Layout(Root()) as layout:
        await layout.render()
        assert element_static_handler.target in layout._event_handlers
        assert component_static_handler.target not in layout._event_handlers
        toggle_type.current()
        await layout.render()
        assert element_static_handler.target not in layout._event_handlers
        assert component_static_handler.target in layout._event_handlers
        toggle_type.current()
        await layout.render()
        assert element_static_handler.target in layout._event_handlers
        assert component_static_handler.target not in layout._event_handlers

async def test_switching_component_definition():
    toggle_component = reactpy.Ref()
    first_used_state = reactpy.Ref(None)
    second_used_state = reactpy.Ref(None)

    @reactpy.component
    def Root():
        if False:
            for i in range(10):
                print('nop')
        (toggle, toggle_component.current) = use_toggle(True)
        if toggle:
            return FirstComponent()
        else:
            return SecondComponent()

    @reactpy.component
    def FirstComponent():
        if False:
            i = 10
            return i + 15
        first_used_state.current = use_state('first')[0]
        use_effect(lambda : lambda : first_used_state.set_current(None))
        return html.div()

    @reactpy.component
    def SecondComponent():
        if False:
            print('Hello World!')
        second_used_state.current = use_state('second')[0]
        use_effect(lambda : lambda : second_used_state.set_current(None))
        return html.div()
    async with reactpy.Layout(Root()) as layout:
        await layout.render()
        assert first_used_state.current == 'first'
        assert second_used_state.current is None
        toggle_component.current()
        await layout.render()
        assert first_used_state.current is None
        assert second_used_state.current == 'second'
        toggle_component.current()
        await layout.render()
        assert first_used_state.current == 'first'
        assert second_used_state.current is None

async def test_element_keys_inside_components_do_not_reset_state_of_component():
    """This is a regression test for a bug.

    You would not expect that calling `set_child_key_num` would trigger state to be
    reset in any `Child()` components but there was a bug where that happened.
    """
    effect_calls_without_state = set()
    set_child_key_num = StaticEventHandler()
    did_call_effect = asyncio.Event()

    @component
    def Parent():
        if False:
            for i in range(10):
                print('nop')
        (state, set_state) = use_state(0)
        return html.div(html.button({'on_click': set_child_key_num.use(lambda : set_state(state + 1))}, 'click me'), Child('some-key'), Child(f'key-{state}'))

    @component
    def Child(child_key):
        if False:
            i = 10
            return i + 15
        (state, set_state) = use_state(0)

        @use_effect
        async def record_if_state_is_reset():
            if state:
                return
            effect_calls_without_state.add(child_key)
            set_state(1)
            did_call_effect.set()
        return html.div({'key': child_key}, child_key)
    async with reactpy.Layout(Parent()) as layout:
        await layout.render()
        await did_call_effect.wait()
        assert effect_calls_without_state == {'some-key', 'key-0'}
        did_call_effect.clear()
        for _i in range(1, 5):
            await layout.deliver(event_message(set_child_key_num.target))
            await layout.render()
            assert effect_calls_without_state == {'some-key', 'key-0'}
            did_call_effect.clear()

async def test_changing_key_of_component_resets_state():
    set_key = Ref()
    did_init_state = Ref(0)
    hook = HookCatcher()

    @component
    @hook.capture
    def Root():
        if False:
            print('Hello World!')
        (key, set_key.current) = use_state('key-1')
        return Child(key=key)

    @component
    def Child():
        if False:
            for i in range(10):
                print('nop')
        use_state(lambda : did_init_state.set_current(did_init_state.current + 1))
    async with Layout(Root()) as layout:
        await layout.render()
        assert did_init_state.current == 1
        set_key.current('key-2')
        await layout.render()
        assert did_init_state.current == 2
        hook.latest.schedule_render()
        await layout.render()
        assert did_init_state.current == 2

async def test_changing_event_handlers_in_the_next_render():
    set_event_name = Ref()
    event_handler = StaticEventHandler()
    did_trigger = Ref(False)

    @component
    def Root():
        if False:
            print('Hello World!')
        (event_name, set_event_name.current) = use_state('first')
        return html.button({event_name: event_handler.use(lambda : did_trigger.set_current(True))})
    async with Layout(Root()) as layout:
        await layout.render()
        await layout.deliver(event_message(event_handler.target))
        assert did_trigger.current
        did_trigger.current = False
        set_event_name.current('second')
        await layout.render()
        await layout.deliver(event_message(event_handler.target))
        assert did_trigger.current
        did_trigger.current = False

async def test_change_element_to_string_causes_unmount():
    set_toggle = Ref()
    did_unmount = Ref(False)

    @component
    def Root():
        if False:
            for i in range(10):
                print('nop')
        (toggle, set_toggle.current) = use_toggle(True)
        if toggle:
            return html.div(Child())
        else:
            return html.div('some-string')

    @component
    def Child():
        if False:
            return 10
        use_effect(lambda : lambda : did_unmount.set_current(True))
    async with Layout(Root()) as layout:
        await layout.render()
        set_toggle.current()
        await layout.render()
        assert did_unmount.current

async def test_does_render_children_after_component():
    """Regression test for bug where layout was appending children to a stale ref

    The stale reference was created when a component got rendered. Thus, everything
    after the component failed to display.
    """

    @reactpy.component
    def Parent():
        if False:
            for i in range(10):
                print('nop')
        return html.div(html.p('first'), Child(), html.p('third'))

    @reactpy.component
    def Child():
        if False:
            while True:
                i = 10
        return html.p('second')
    async with reactpy.Layout(Parent()) as layout:
        update = await layout.render()
        assert update['model'] == {'tagName': '', 'children': [{'tagName': 'div', 'children': [{'tagName': 'p', 'children': ['first']}, {'tagName': '', 'children': [{'tagName': 'p', 'children': ['second']}]}, {'tagName': 'p', 'children': ['third']}]}]}

async def test_render_removed_context_consumer():
    Context = reactpy.create_context(None)
    toggle_remove_child = None
    schedule_removed_child_render = None

    @component
    def Parent():
        if False:
            i = 10
            return i + 15
        nonlocal toggle_remove_child
        (remove_child, toggle_remove_child) = use_toggle()
        return Context(html.div() if remove_child else Child(), value=None)

    @component
    def Child():
        if False:
            return 10
        nonlocal schedule_removed_child_render
        schedule_removed_child_render = use_force_render()
    async with reactpy.Layout(Parent()) as layout:
        await layout.render()
        toggle_remove_child()
        await layout.render()
        schedule_removed_child_render()
        render_task = asyncio.create_task(layout.render())
        (done, pending) = await asyncio.wait([render_task], timeout=0.1)
        assert not done and pending
        render_task.cancel()

async def test_ensure_model_path_udpates():
    """
    This is regression test for a bug in which we failed to update the path of a bug
    that arose when the "path" of a component within the overall model was not updated
    when the component changes position amongst its siblings. This meant that when
    a component whose position had changed would attempt to update the view at its old
    position.
    """

    @component
    def Item(item: str, all_items: State[list[str]]):
        if False:
            return 10
        color = use_state(None)

        def deleteme(event):
            if False:
                while True:
                    i = 10
            all_items.set_value([i for i in all_items.value if i != item])

        def colorize(event):
            if False:
                while True:
                    i = 10
            color.set_value('blue' if not color.value else None)
        return html.div({'id': item, 'color': color.value}, html.button({'on_click': colorize}, f'Color {item}'), html.button({'on_click': deleteme}, f'Delete {item}'))

    @component
    def App():
        if False:
            while True:
                i = 10
        items = use_state(['A', 'B', 'C'])
        return html._([Item(item, items, key=item) for item in items.value])
    async with layout_runner(reactpy.Layout(App())) as runner:
        tree = await runner.render()
        (b, b_info) = find_element(tree, select.id_equals('B'))
        assert b_info.path == (0, 1, 0)
        (b_delete, _) = find_element(b, select.text_equals('Delete B'))
        await runner.trigger(b_delete, 'on_click', {})
        tree = await runner.render()
        assert not element_exists(tree, select.id_equals('B'))
        (c, c_info) = find_element(tree, select.id_equals('C'))
        assert c_info.path == (0, 1, 0)
        (c_color, _) = find_element(c, select.text_equals('Color C'))
        await runner.trigger(c_color, 'on_click', {})
        tree = await runner.render()
        (c, c_info) = find_element(tree, select.id_equals('C'))
        assert c_info.path == (0, 1, 0)
        assert c['attributes']['color'] == 'blue'