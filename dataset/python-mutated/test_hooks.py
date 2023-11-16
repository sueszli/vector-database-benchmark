import asyncio
import pytest
import reactpy
from reactpy import html
from reactpy.config import REACTPY_DEBUG_MODE
from reactpy.core.hooks import COMPONENT_DID_RENDER_EFFECT, LifeCycleHook, current_hook, strictly_equal
from reactpy.core.layout import Layout
from reactpy.testing import DisplayFixture, HookCatcher, assert_reactpy_did_log, poll
from reactpy.testing.logs import assert_reactpy_did_not_log
from reactpy.utils import Ref
from tests.tooling.common import DEFAULT_TYPE_DELAY, update_message

async def test_must_be_rendering_in_layout_to_use_hooks():

    @reactpy.component
    def SimpleComponentWithHook():
        if False:
            i = 10
            return i + 15
        reactpy.hooks.use_state(None)
        return reactpy.html.div()
    with pytest.raises(RuntimeError, match='No life cycle hook is active'):
        await SimpleComponentWithHook().render()
    async with reactpy.Layout(SimpleComponentWithHook()) as layout:
        await layout.render()

async def test_simple_stateful_component():

    @reactpy.component
    def SimpleStatefulComponent():
        if False:
            return 10
        (index, set_index) = reactpy.hooks.use_state(0)
        set_index(index + 1)
        return reactpy.html.div(index)
    sse = SimpleStatefulComponent()
    async with reactpy.Layout(sse) as layout:
        update_1 = await layout.render()
        assert update_1 == update_message(path='', model={'tagName': '', 'children': [{'tagName': 'div', 'children': ['0']}]})
        update_2 = await layout.render()
        assert update_2 == update_message(path='', model={'tagName': '', 'children': [{'tagName': 'div', 'children': ['1']}]})
        update_3 = await layout.render()
        assert update_3 == update_message(path='', model={'tagName': '', 'children': [{'tagName': 'div', 'children': ['2']}]})

async def test_set_state_callback_identity_is_preserved():
    saved_set_state_hooks = []

    @reactpy.component
    def SimpleStatefulComponent():
        if False:
            for i in range(10):
                print('nop')
        (index, set_index) = reactpy.hooks.use_state(0)
        saved_set_state_hooks.append(set_index)
        set_index(index + 1)
        return reactpy.html.div(index)
    sse = SimpleStatefulComponent()
    async with reactpy.Layout(sse) as layout:
        await layout.render()
        await layout.render()
        await layout.render()
        await layout.render()
    first_hook = saved_set_state_hooks[0]
    for h in saved_set_state_hooks[1:]:
        assert first_hook is h

async def test_use_state_with_constructor():
    constructor_call_count = reactpy.Ref(0)
    set_outer_state = reactpy.Ref()
    set_inner_key = reactpy.Ref()
    set_inner_state = reactpy.Ref()

    def make_default():
        if False:
            print('Hello World!')
        constructor_call_count.current += 1
        return 0

    @reactpy.component
    def Outer():
        if False:
            while True:
                i = 10
        (state, set_outer_state.current) = reactpy.use_state(0)
        (inner_key, set_inner_key.current) = reactpy.use_state('first')
        return reactpy.html.div(state, Inner(key=inner_key))

    @reactpy.component
    def Inner():
        if False:
            while True:
                i = 10
        (state, set_inner_state.current) = reactpy.use_state(make_default)
        return reactpy.html.div(state)
    async with reactpy.Layout(Outer()) as layout:
        await layout.render()
        assert constructor_call_count.current == 1
        set_outer_state.current(1)
        await layout.render()
        assert constructor_call_count.current == 1
        set_inner_state.current(1)
        await layout.render()
        assert constructor_call_count.current == 1
        set_inner_key.current('second')
        await layout.render()
        assert constructor_call_count.current == 2

async def test_set_state_with_reducer_instead_of_value():
    count = reactpy.Ref()
    set_count = reactpy.Ref()

    def increment(count):
        if False:
            for i in range(10):
                print('nop')
        return count + 1

    @reactpy.component
    def Counter():
        if False:
            return 10
        (count.current, set_count.current) = reactpy.hooks.use_state(0)
        return reactpy.html.div(count.current)
    async with reactpy.Layout(Counter()) as layout:
        await layout.render()
        for i in range(4):
            assert count.current == i
            set_count.current(increment)
            await layout.render()

async def test_set_state_checks_identity_not_equality(display: DisplayFixture):
    r_1 = reactpy.Ref('value')
    r_2 = reactpy.Ref('value')
    assert r_1 == r_2
    assert r_1 is not r_2
    render_count = reactpy.Ref(0)
    event_count = reactpy.Ref(0)

    def event_count_tracker(function):
        if False:
            i = 10
            return i + 15

        def tracker(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            event_count.current += 1
            return function(*args, **kwargs)
        return tracker

    @reactpy.component
    def TestComponent():
        if False:
            while True:
                i = 10
        (state, set_state) = reactpy.hooks.use_state(r_1)
        render_count.current += 1
        return reactpy.html.div(reactpy.html.button({'id': 'r_1', 'on_click': event_count_tracker(lambda event: set_state(r_1))}, 'r_1'), reactpy.html.button({'id': 'r_2', 'on_click': event_count_tracker(lambda event: set_state(r_2))}, 'r_2'), f"Last state: {('r_1' if state is r_1 else 'r_2')}")
    await display.show(TestComponent)
    client_r_1_button = await display.page.wait_for_selector('#r_1')
    client_r_2_button = await display.page.wait_for_selector('#r_2')
    poll_event_count = poll(lambda : event_count.current)
    poll_render_count = poll(lambda : render_count.current)
    assert render_count.current == 1
    assert event_count.current == 0
    await client_r_1_button.click()
    await poll_event_count.until_equals(1)
    await poll_render_count.until_equals(1)
    await client_r_2_button.click()
    await poll_event_count.until_equals(2)
    await poll_render_count.until_equals(2)
    await client_r_2_button.click()
    await poll_event_count.until_equals(3)
    await poll_render_count.until_equals(2)

async def test_simple_input_with_use_state(display: DisplayFixture):
    message_ref = reactpy.Ref(None)

    @reactpy.component
    def Input(message=None):
        if False:
            return 10
        (message, set_message) = reactpy.hooks.use_state(message)
        message_ref.current = message

        async def on_change(event):
            if event['target']['value'] == 'this is a test':
                set_message(event['target']['value'])
        if message is None:
            return reactpy.html.input({'id': 'input', 'on_change': on_change})
        else:
            return reactpy.html.p({'id': 'complete'}, ['Complete'])
    await display.show(Input)
    button = await display.page.wait_for_selector('#input')
    await button.type('this is a test', delay=DEFAULT_TYPE_DELAY)
    await display.page.wait_for_selector('#complete')
    assert message_ref.current == 'this is a test'

async def test_double_set_state(display: DisplayFixture):

    @reactpy.component
    def SomeComponent():
        if False:
            i = 10
            return i + 15
        (state_1, set_state_1) = reactpy.hooks.use_state(0)
        (state_2, set_state_2) = reactpy.hooks.use_state(0)

        def double_set_state(event):
            if False:
                return 10
            set_state_1(state_1 + 1)
            set_state_2(state_2 + 1)
        return reactpy.html.div(reactpy.html.div({'id': 'first', 'data-value': state_1}, f'value is: {state_1}'), reactpy.html.div({'id': 'second', 'data-value': state_2}, f'value is: {state_2}'), reactpy.html.button({'id': 'button', 'on_click': double_set_state}, 'click me'))
    await display.show(SomeComponent)
    button = await display.page.wait_for_selector('#button')
    first = await display.page.wait_for_selector('#first')
    second = await display.page.wait_for_selector('#second')
    assert await first.get_attribute('data-value') == '0'
    assert await second.get_attribute('data-value') == '0'
    await button.click()
    assert await first.get_attribute('data-value') == '1'
    assert await second.get_attribute('data-value') == '1'
    await button.click()
    assert await first.get_attribute('data-value') == '2'
    assert await second.get_attribute('data-value') == '2'

async def test_use_effect_callback_occurs_after_full_render_is_complete():
    effect_triggered = reactpy.Ref(False)
    effect_triggers_after_final_render = reactpy.Ref(None)

    @reactpy.component
    def OuterComponent():
        if False:
            while True:
                i = 10
        return reactpy.html.div(ComponentWithEffect(), CheckNoEffectYet())

    @reactpy.component
    def ComponentWithEffect():
        if False:
            i = 10
            return i + 15

        @reactpy.hooks.use_effect
        def effect():
            if False:
                print('Hello World!')
            effect_triggered.current = True
        return reactpy.html.div()

    @reactpy.component
    def CheckNoEffectYet():
        if False:
            i = 10
            return i + 15
        effect_triggers_after_final_render.current = not effect_triggered.current
        return reactpy.html.div()
    async with reactpy.Layout(OuterComponent()) as layout:
        await layout.render()
    assert effect_triggered.current
    assert effect_triggers_after_final_render.current is not None
    assert effect_triggers_after_final_render.current

async def test_use_effect_cleanup_occurs_before_next_effect():
    component_hook = HookCatcher()
    cleanup_triggered = reactpy.Ref(False)
    cleanup_triggered_before_next_effect = reactpy.Ref(False)

    @reactpy.component
    @component_hook.capture
    def ComponentWithEffect():
        if False:
            print('Hello World!')

        @reactpy.hooks.use_effect(dependencies=None)
        def effect():
            if False:
                return 10
            if cleanup_triggered.current:
                cleanup_triggered_before_next_effect.current = True

            def cleanup():
                if False:
                    print('Hello World!')
                cleanup_triggered.current = True
            return cleanup
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithEffect()) as layout:
        await layout.render()
        assert not cleanup_triggered.current
        component_hook.latest.schedule_render()
        await layout.render()
        assert cleanup_triggered.current
        assert cleanup_triggered_before_next_effect.current

async def test_use_effect_cleanup_occurs_on_will_unmount():
    set_key = reactpy.Ref()
    component_did_render = reactpy.Ref(False)
    cleanup_triggered = reactpy.Ref(False)
    cleanup_triggered_before_next_render = reactpy.Ref(False)

    @reactpy.component
    def OuterComponent():
        if False:
            while True:
                i = 10
        (key, set_key.current) = reactpy.use_state('first')
        return ComponentWithEffect(key=key)

    @reactpy.component
    def ComponentWithEffect():
        if False:
            print('Hello World!')
        if component_did_render.current and cleanup_triggered.current:
            cleanup_triggered_before_next_render.current = True
        component_did_render.current = True

        @reactpy.hooks.use_effect
        def effect():
            if False:
                return 10

            def cleanup():
                if False:
                    i = 10
                    return i + 15
                cleanup_triggered.current = True
            return cleanup
        return reactpy.html.div()
    async with reactpy.Layout(OuterComponent()) as layout:
        await layout.render()
        assert not cleanup_triggered.current
        set_key.current('second')
        await layout.render()
        assert cleanup_triggered.current
        assert cleanup_triggered_before_next_render.current

async def test_memoized_effect_on_recreated_if_dependencies_change():
    component_hook = HookCatcher()
    set_state_callback = reactpy.Ref(None)
    effect_run_count = reactpy.Ref(0)
    first_value = 1
    second_value = 2

    @reactpy.component
    @component_hook.capture
    def ComponentWithMemoizedEffect():
        if False:
            print('Hello World!')
        (state, set_state_callback.current) = reactpy.hooks.use_state(first_value)

        @reactpy.hooks.use_effect(dependencies=[state])
        def effect():
            if False:
                return 10
            effect_run_count.current += 1
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithMemoizedEffect()) as layout:
        await layout.render()
        assert effect_run_count.current == 1
        component_hook.latest.schedule_render()
        await layout.render()
        assert effect_run_count.current == 1
        set_state_callback.current(second_value)
        await layout.render()
        assert effect_run_count.current == 2
        component_hook.latest.schedule_render()
        await layout.render()
        assert effect_run_count.current == 2

async def test_memoized_effect_cleanup_only_triggered_before_new_effect():
    component_hook = HookCatcher()
    set_state_callback = reactpy.Ref(None)
    cleanup_trigger_count = reactpy.Ref(0)
    first_value = 1
    second_value = 2

    @reactpy.component
    @component_hook.capture
    def ComponentWithEffect():
        if False:
            while True:
                i = 10
        (state, set_state_callback.current) = reactpy.hooks.use_state(first_value)

        @reactpy.hooks.use_effect(dependencies=[state])
        def effect():
            if False:
                print('Hello World!')

            def cleanup():
                if False:
                    for i in range(10):
                        print('nop')
                cleanup_trigger_count.current += 1
            return cleanup
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithEffect()) as layout:
        await layout.render()
        assert cleanup_trigger_count.current == 0
        component_hook.latest.schedule_render()
        await layout.render()
        assert cleanup_trigger_count.current == 0
        set_state_callback.current(second_value)
        await layout.render()
        assert cleanup_trigger_count.current == 1

async def test_use_async_effect():
    effect_ran = asyncio.Event()

    @reactpy.component
    def ComponentWithAsyncEffect():
        if False:
            i = 10
            return i + 15

        @reactpy.hooks.use_effect
        async def effect():
            effect_ran.set()
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithAsyncEffect()) as layout:
        await layout.render()
        await asyncio.wait_for(effect_ran.wait(), 1)

async def test_use_async_effect_cleanup():
    component_hook = HookCatcher()
    effect_ran = asyncio.Event()
    cleanup_ran = asyncio.Event()

    @reactpy.component
    @component_hook.capture
    def ComponentWithAsyncEffect():
        if False:
            while True:
                i = 10

        @reactpy.hooks.use_effect(dependencies=None)
        async def effect():
            effect_ran.set()
            return cleanup_ran.set
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithAsyncEffect()) as layout:
        await layout.render()
        component_hook.latest.schedule_render()
        await layout.render()
    await asyncio.wait_for(cleanup_ran.wait(), 1)

async def test_use_async_effect_cancel(caplog):
    component_hook = HookCatcher()
    effect_ran = asyncio.Event()
    effect_was_cancelled = asyncio.Event()
    event_that_never_occurs = asyncio.Event()

    @reactpy.component
    @component_hook.capture
    def ComponentWithLongWaitingEffect():
        if False:
            for i in range(10):
                print('nop')

        @reactpy.hooks.use_effect(dependencies=None)
        async def effect():
            effect_ran.set()
            try:
                await event_that_never_occurs.wait()
            except asyncio.CancelledError:
                effect_was_cancelled.set()
                raise
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithLongWaitingEffect()) as layout:
        await layout.render()
        await effect_ran.wait()
        component_hook.latest.schedule_render()
        await layout.render()
    await asyncio.wait_for(effect_was_cancelled.wait(), 1)
    event_that_never_occurs.set()

async def test_error_in_effect_is_gracefully_handled(caplog):

    @reactpy.component
    def ComponentWithEffect():
        if False:
            print('Hello World!')

        @reactpy.hooks.use_effect
        def bad_effect():
            if False:
                i = 10
                return i + 15
            msg = 'Something went wong :('
            raise ValueError(msg)
        return reactpy.html.div()
    with assert_reactpy_did_log(match_message='Layout post-render effect .* failed'):
        async with reactpy.Layout(ComponentWithEffect()) as layout:
            await layout.render()

async def test_error_in_effect_pre_unmount_cleanup_is_gracefully_handled():
    set_key = reactpy.Ref()

    @reactpy.component
    def OuterComponent():
        if False:
            i = 10
            return i + 15
        (key, set_key.current) = reactpy.use_state('first')
        return ComponentWithEffect(key=key)

    @reactpy.component
    def ComponentWithEffect():
        if False:
            print('Hello World!')

        @reactpy.hooks.use_effect
        def ok_effect():
            if False:
                while True:
                    i = 10

            def bad_cleanup():
                if False:
                    print('Hello World!')
                msg = 'Something went wong :('
                raise ValueError(msg)
            return bad_cleanup
        return reactpy.html.div()
    with assert_reactpy_did_log(match_message='Pre-unmount effect .*? failed', error_type=ValueError):
        async with reactpy.Layout(OuterComponent()) as layout:
            await layout.render()
            set_key.current('second')
            await layout.render()

async def test_use_reducer():
    saved_count = reactpy.Ref(None)
    saved_dispatch = reactpy.Ref(None)

    def reducer(count, action):
        if False:
            i = 10
            return i + 15
        if action == 'increment':
            return count + 1
        elif action == 'decrement':
            return count - 1
        else:
            msg = f"Unknown action '{action}'"
            raise ValueError(msg)

    @reactpy.component
    def Counter(initial_count):
        if False:
            i = 10
            return i + 15
        (saved_count.current, saved_dispatch.current) = reactpy.hooks.use_reducer(reducer, initial_count)
        return reactpy.html.div()
    async with reactpy.Layout(Counter(0)) as layout:
        await layout.render()
        assert saved_count.current == 0
        saved_dispatch.current('increment')
        await layout.render()
        assert saved_count.current == 1
        saved_dispatch.current('decrement')
        await layout.render()
        assert saved_count.current == 0

async def test_use_reducer_dispatch_callback_identity_is_preserved():
    saved_dispatchers = []

    def reducer(count, action):
        if False:
            return 10
        if action == 'increment':
            return count + 1
        else:
            msg = f"Unknown action '{action}'"
            raise ValueError(msg)

    @reactpy.component
    def ComponentWithUseReduce():
        if False:
            for i in range(10):
                print('nop')
        saved_dispatchers.append(reactpy.hooks.use_reducer(reducer, 0)[1])
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithUseReduce()) as layout:
        for _ in range(3):
            await layout.render()
            saved_dispatchers[-1]('increment')
    first_dispatch = saved_dispatchers[0]
    for d in saved_dispatchers[1:]:
        assert first_dispatch is d

async def test_use_callback_identity():
    component_hook = HookCatcher()
    used_callbacks = []

    @reactpy.component
    @component_hook.capture
    def ComponentWithRef():
        if False:
            for i in range(10):
                print('nop')
        used_callbacks.append(reactpy.hooks.use_callback(lambda : None))
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithRef()) as layout:
        await layout.render()
        component_hook.latest.schedule_render()
        await layout.render()
    assert used_callbacks[0] is used_callbacks[1]
    assert len(used_callbacks) == 2

async def test_use_callback_memoization():
    component_hook = HookCatcher()
    set_state_hook = reactpy.Ref(None)
    used_callbacks = []

    @reactpy.component
    @component_hook.capture
    def ComponentWithRef():
        if False:
            return 10
        (state, set_state_hook.current) = reactpy.hooks.use_state(0)

        @reactpy.hooks.use_callback(dependencies=[state])
        def cb():
            if False:
                for i in range(10):
                    print('nop')
            return None
        used_callbacks.append(cb)
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithRef()) as layout:
        await layout.render()
        set_state_hook.current(1)
        await layout.render()
        component_hook.latest.schedule_render()
        await layout.render()
    assert used_callbacks[0] is not used_callbacks[1]
    assert used_callbacks[1] is used_callbacks[2]
    assert len(used_callbacks) == 3

async def test_use_memo():
    component_hook = HookCatcher()
    set_state_hook = reactpy.Ref(None)
    used_values = []

    @reactpy.component
    @component_hook.capture
    def ComponentWithMemo():
        if False:
            return 10
        (state, set_state_hook.current) = reactpy.hooks.use_state(0)
        value = reactpy.hooks.use_memo(lambda : reactpy.Ref(state), [state])
        used_values.append(value)
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithMemo()) as layout:
        await layout.render()
        set_state_hook.current(1)
        await layout.render()
        component_hook.latest.schedule_render()
        await layout.render()
    assert used_values[0] is not used_values[1]
    assert used_values[1] is used_values[2]
    assert len(used_values) == 3

async def test_use_memo_always_runs_if_dependencies_are_none():
    component_hook = HookCatcher()
    used_values = []
    iter_values = iter([1, 2, 3])

    @reactpy.component
    @component_hook.capture
    def ComponentWithMemo():
        if False:
            print('Hello World!')
        value = reactpy.hooks.use_memo(lambda : next(iter_values), dependencies=None)
        used_values.append(value)
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithMemo()) as layout:
        await layout.render()
        component_hook.latest.schedule_render()
        await layout.render()
        component_hook.latest.schedule_render()
        await layout.render()
    assert used_values == [1, 2, 3]

async def test_use_memo_with_stored_deps_is_empty_tuple_after_deps_are_none():
    component_hook = HookCatcher()
    used_values = []
    iter_values = iter([1, 2, 3])
    deps_used_in_memo = reactpy.Ref(())

    @reactpy.component
    @component_hook.capture
    def ComponentWithMemo():
        if False:
            for i in range(10):
                print('nop')
        value = reactpy.hooks.use_memo(lambda : next(iter_values), deps_used_in_memo.current)
        used_values.append(value)
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithMemo()) as layout:
        await layout.render()
        component_hook.latest.schedule_render()
        deps_used_in_memo.current = None
        await layout.render()
        component_hook.latest.schedule_render()
        deps_used_in_memo.current = ()
        await layout.render()
    assert used_values == [1, 2, 2]

async def test_use_memo_never_runs_if_deps_is_empty_list():
    component_hook = HookCatcher()
    used_values = []
    iter_values = iter([1, 2, 3])

    @reactpy.component
    @component_hook.capture
    def ComponentWithMemo():
        if False:
            return 10
        value = reactpy.hooks.use_memo(lambda : next(iter_values), ())
        used_values.append(value)
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithMemo()) as layout:
        await layout.render()
        component_hook.latest.schedule_render()
        await layout.render()
        component_hook.latest.schedule_render()
        await layout.render()
    assert used_values == [1, 1, 1]

async def test_use_ref():
    component_hook = HookCatcher()
    used_refs = []

    @reactpy.component
    @component_hook.capture
    def ComponentWithRef():
        if False:
            while True:
                i = 10
        used_refs.append(reactpy.hooks.use_ref(1))
        return reactpy.html.div()
    async with reactpy.Layout(ComponentWithRef()) as layout:
        await layout.render()
        component_hook.latest.schedule_render()
        await layout.render()
    assert used_refs[0] is used_refs[1]
    assert len(used_refs) == 2

def test_bad_schedule_render_callback():
    if False:
        while True:
            i = 10

    def bad_callback():
        if False:
            for i in range(10):
                print('nop')
        msg = 'something went wrong'
        raise ValueError(msg)
    with assert_reactpy_did_log(match_message=f'Failed to schedule render via {bad_callback}'):
        LifeCycleHook(bad_callback).schedule_render()

async def test_use_effect_automatically_infers_closure_values():
    set_count = reactpy.Ref()
    did_effect = asyncio.Event()

    @reactpy.component
    def CounterWithEffect():
        if False:
            i = 10
            return i + 15
        (count, set_count.current) = reactpy.hooks.use_state(0)

        @reactpy.hooks.use_effect
        def some_effect_that_uses_count():
            if False:
                print('Hello World!')
            'should automatically trigger on count change'
            _ = count
            did_effect.set()
        return reactpy.html.div()
    async with reactpy.Layout(CounterWithEffect()) as layout:
        await layout.render()
        await did_effect.wait()
        did_effect.clear()
        for i in range(1, 3):
            set_count.current(i)
            await layout.render()
            await did_effect.wait()
            did_effect.clear()

async def test_use_memo_automatically_infers_closure_values():
    set_count = reactpy.Ref()
    did_memo = asyncio.Event()

    @reactpy.component
    def CounterWithEffect():
        if False:
            print('Hello World!')
        (count, set_count.current) = reactpy.hooks.use_state(0)

        @reactpy.hooks.use_memo
        def some_memo_func_that_uses_count():
            if False:
                i = 10
                return i + 15
            'should automatically trigger on count change'
            _ = count
            did_memo.set()
        return reactpy.html.div()
    async with reactpy.Layout(CounterWithEffect()) as layout:
        await layout.render()
        await did_memo.wait()
        did_memo.clear()
        for i in range(1, 3):
            set_count.current(i)
            await layout.render()
            await did_memo.wait()
            did_memo.clear()

async def test_use_context_default_value():
    Context = reactpy.create_context('something')
    value = reactpy.Ref()

    @reactpy.component
    def ComponentProvidesContext():
        if False:
            for i in range(10):
                print('nop')
        return Context(ComponentUsesContext())

    @reactpy.component
    def ComponentUsesContext():
        if False:
            i = 10
            return i + 15
        value.current = reactpy.use_context(Context)
        return html.div()
    async with reactpy.Layout(ComponentProvidesContext()) as layout:
        await layout.render()
        assert value.current == 'something'

    @reactpy.component
    def ComponentUsesContext2():
        if False:
            i = 10
            return i + 15
        value.current = reactpy.use_context(Context)
        return html.div()
    async with reactpy.Layout(ComponentUsesContext2()) as layout:
        await layout.render()
        assert value.current == 'something'

def test_context_repr():
    if False:
        for i in range(10):
            print('nop')
    sample_context = reactpy.create_context(None)
    assert repr(sample_context()) == f'ContextProvider({sample_context})'

async def test_use_context_updates_components_even_if_memoized():
    Context = reactpy.create_context(None)
    value = reactpy.Ref(None)
    render_count = reactpy.Ref(0)
    set_state = reactpy.Ref()

    @reactpy.component
    def ComponentProvidesContext():
        if False:
            for i in range(10):
                print('nop')
        (state, set_state.current) = reactpy.use_state(0)
        return Context(ComponentInContext(), value=state)

    @reactpy.component
    def ComponentInContext():
        if False:
            return 10
        return reactpy.use_memo(MemoizedComponentUsesContext)

    @reactpy.component
    def MemoizedComponentUsesContext():
        if False:
            for i in range(10):
                print('nop')
        value.current = reactpy.use_context(Context)
        render_count.current += 1
        return html.div()
    async with reactpy.Layout(ComponentProvidesContext()) as layout:
        await layout.render()
        assert render_count.current == 1
        assert value.current == 0
        set_state.current(1)
        await layout.render()
        assert render_count.current == 2
        assert value.current == 1
        set_state.current(2)
        await layout.render()
        assert render_count.current == 3
        assert value.current == 2

async def test_context_values_are_scoped():
    Context = reactpy.create_context(None)

    @reactpy.component
    def Parent():
        if False:
            for i in range(10):
                print('nop')
        return html._(Context(Context(Child1(), value=1), value='something-else'), Context(Child2(), value=2))

    @reactpy.component
    def Child1():
        if False:
            return 10
        assert reactpy.use_context(Context) == 1

    @reactpy.component
    def Child2():
        if False:
            for i in range(10):
                print('nop')
        assert reactpy.use_context(Context) == 2
    async with Layout(Parent()) as layout:
        await layout.render()

async def test_error_in_layout_effect_cleanup_is_gracefully_handled():
    component_hook = HookCatcher()

    @reactpy.component
    @component_hook.capture
    def ComponentWithEffect():
        if False:
            i = 10
            return i + 15

        @reactpy.hooks.use_effect(dependencies=None)
        def bad_effect():
            if False:
                i = 10
                return i + 15
            msg = 'The error message'
            raise ValueError(msg)
        return reactpy.html.div()
    with assert_reactpy_did_log(match_message='post-render effect .*? failed', error_type=ValueError, match_error='The error message'):
        async with reactpy.Layout(ComponentWithEffect()) as layout:
            await layout.render()
            component_hook.latest.schedule_render()
            await layout.render()

async def test_set_state_during_render():
    render_count = Ref(0)

    @reactpy.component
    def SetStateDuringRender():
        if False:
            return 10
        render_count.current += 1
        (state, set_state) = reactpy.use_state(0)
        if not state:
            set_state(state + 1)
        return html.div(state)
    async with Layout(SetStateDuringRender()) as layout:
        await layout.render()
        assert render_count.current == 1
        await layout.render()
        assert render_count.current == 2
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(layout.render(), timeout=0.1)

@pytest.mark.skipif(not REACTPY_DEBUG_MODE.current, reason='only logs in debug mode')
async def test_use_debug_mode():
    set_message = reactpy.Ref()
    component_hook = HookCatcher()

    @reactpy.component
    @component_hook.capture
    def SomeComponent():
        if False:
            return 10
        (message, set_message.current) = reactpy.use_state('hello')
        reactpy.use_debug_value(f'message is {message!r}')
        return reactpy.html.div()
    async with reactpy.Layout(SomeComponent()) as layout:
        with assert_reactpy_did_log("SomeComponent\\(.*?\\) message is 'hello'"):
            await layout.render()
        set_message.current('bye')
        with assert_reactpy_did_log("SomeComponent\\(.*?\\) message is 'bye'"):
            await layout.render()
        component_hook.latest.schedule_render()
        with assert_reactpy_did_not_log("SomeComponent\\(.*?\\) message is 'bye'"):
            await layout.render()

@pytest.mark.skipif(not REACTPY_DEBUG_MODE.current, reason='only logs in debug mode')
async def test_use_debug_mode_with_factory():
    set_message = reactpy.Ref()
    component_hook = HookCatcher()

    @reactpy.component
    @component_hook.capture
    def SomeComponent():
        if False:
            i = 10
            return i + 15
        (message, set_message.current) = reactpy.use_state('hello')
        reactpy.use_debug_value(lambda : f'message is {message!r}')
        return reactpy.html.div()
    async with reactpy.Layout(SomeComponent()) as layout:
        with assert_reactpy_did_log("SomeComponent\\(.*?\\) message is 'hello'"):
            await layout.render()
        set_message.current('bye')
        with assert_reactpy_did_log("SomeComponent\\(.*?\\) message is 'bye'"):
            await layout.render()
        component_hook.latest.schedule_render()
        with assert_reactpy_did_not_log("SomeComponent\\(.*?\\) message is 'bye'"):
            await layout.render()

@pytest.mark.skipif(REACTPY_DEBUG_MODE.current, reason='logs in debug mode')
async def test_use_debug_mode_does_not_log_if_not_in_debug_mode():
    set_message = reactpy.Ref()

    @reactpy.component
    def SomeComponent():
        if False:
            print('Hello World!')
        (message, set_message.current) = reactpy.use_state('hello')
        reactpy.use_debug_value(lambda : f'message is {message!r}')
        return reactpy.html.div()
    async with reactpy.Layout(SomeComponent()) as layout:
        with assert_reactpy_did_not_log("SomeComponent\\(.*?\\) message is 'hello'"):
            await layout.render()
        set_message.current('bye')
        with assert_reactpy_did_not_log("SomeComponent\\(.*?\\) message is 'bye'"):
            await layout.render()

async def test_conditionally_rendered_components_can_use_context():
    set_state = reactpy.Ref()
    used_context_values = []
    some_context = reactpy.create_context(None)

    @reactpy.component
    def SomeComponent():
        if False:
            for i in range(10):
                print('nop')
        (state, set_state.current) = reactpy.use_state(True)
        if state:
            return FirstCondition()
        else:
            return SecondCondition()

    @reactpy.component
    def FirstCondition():
        if False:
            for i in range(10):
                print('nop')
        used_context_values.append(reactpy.use_context(some_context) + '-1')

    @reactpy.component
    def SecondCondition():
        if False:
            print('Hello World!')
        used_context_values.append(reactpy.use_context(some_context) + '-2')
    async with reactpy.Layout(some_context(SomeComponent(), value='the-value')) as layout:
        await layout.render()
        assert used_context_values == ['the-value-1']
        set_state.current(False)
        await layout.render()
        assert used_context_values == ['the-value-1', 'the-value-2']

@pytest.mark.parametrize('x, y, result', [('text', 'text', True), ('text', 'not-text', False), (b'text', b'text', True), (b'text', b'not-text', False), (bytearray([1, 2, 3]), bytearray([1, 2, 3]), True), (bytearray([1, 2, 3]), bytearray([1, 2, 3, 4]), False), (1.0, 1.0, True), (1.0, 2.0, False), (1j, 1j, True), (1j, 2j, False), (-100000, -100000, True), (100000, 100000, True), (123, 456, False)])
def test_strictly_equal(x, y, result):
    if False:
        print('Hello World!')
    assert strictly_equal(x, y) is result
STRICT_EQUALITY_VALUE_CONSTRUCTORS = [lambda : 'string-text', lambda : b'byte-text', lambda : bytearray([1, 2, 3]), lambda : bytearray([1, 2, 3]), lambda : 1.0, lambda : 10000000, lambda : 1j]

@pytest.mark.parametrize('get_value', STRICT_EQUALITY_VALUE_CONSTRUCTORS)
async def test_use_state_compares_with_strict_equality(get_value):
    render_count = reactpy.Ref(0)
    set_state = reactpy.Ref()

    @reactpy.component
    def SomeComponent():
        if False:
            print('Hello World!')
        (_, set_state.current) = reactpy.use_state(get_value())
        render_count.current += 1
    async with reactpy.Layout(SomeComponent()) as layout:
        await layout.render()
        assert render_count.current == 1
        set_state.current(get_value())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(layout.render(), timeout=0.1)

@pytest.mark.parametrize('get_value', STRICT_EQUALITY_VALUE_CONSTRUCTORS)
async def test_use_effect_compares_with_strict_equality(get_value):
    effect_count = reactpy.Ref(0)
    value = reactpy.Ref('string')
    hook = HookCatcher()

    @reactpy.component
    @hook.capture
    def SomeComponent():
        if False:
            for i in range(10):
                print('nop')

        @reactpy.use_effect(dependencies=[value.current])
        def incr_effect_count():
            if False:
                return 10
            effect_count.current += 1
    async with reactpy.Layout(SomeComponent()) as layout:
        await layout.render()
        assert effect_count.current == 1
        value.current = 'string'
        hook.latest.schedule_render()
        await layout.render()
        assert effect_count.current == 1

async def test_use_state_named_tuple():
    state = reactpy.Ref()

    @reactpy.component
    def some_component():
        if False:
            print('Hello World!')
        state.current = reactpy.use_state(1)
    async with reactpy.Layout(some_component()) as layout:
        await layout.render()
        assert state.current.value == 1
        state.current.set_value(2)
        await layout.render()
        assert state.current.value == 2

async def test_error_in_component_effect_cleanup_is_gracefully_handled():
    component_hook = HookCatcher()

    @reactpy.component
    @component_hook.capture
    def ComponentWithEffect():
        if False:
            return 10
        hook = current_hook()

        def bad_effect():
            if False:
                i = 10
                return i + 15
            raise ValueError('The error message')
        hook.add_effect(COMPONENT_DID_RENDER_EFFECT, bad_effect)
        return reactpy.html.div()
    with assert_reactpy_did_log(match_message='Component post-render effect .*? failed', error_type=ValueError, match_error='The error message'):
        async with reactpy.Layout(ComponentWithEffect()) as layout:
            await layout.render()
            component_hook.latest.schedule_render()
            await layout.render()