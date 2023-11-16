from __future__ import annotations
import asyncio
import pytest
from textual.app import App, ComposeResult
from textual.reactive import Reactive, TooManyComputesError, reactive, var
from textual.widget import Widget
OLD_VALUE = 5000
NEW_VALUE = 1000000

async def test_watch():
    """Test that changes to a watched reactive attribute happen immediately."""

    class WatchApp(App):
        count = reactive(0, init=False)
        watcher_call_count = 0

        def watch_count(self, value: int) -> None:
            if False:
                i = 10
                return i + 15
            self.watcher_call_count = value
    app = WatchApp()
    async with app.run_test():
        app.count += 1
        assert app.watcher_call_count == 1
        app.count += 1
        assert app.watcher_call_count == 2
        app.count -= 1
        assert app.watcher_call_count == 1
        app.count -= 1
        assert app.watcher_call_count == 0

async def test_watch_async_init_false():
    """Ensure that async watchers are called eventually when set by user code"""

    class WatchAsyncApp(App):
        count = reactive(OLD_VALUE, init=False)
        watcher_old_value = None
        watcher_new_value = None
        watcher_called_event = asyncio.Event()

        async def watch_count(self, old_value: int, new_value: int) -> None:
            self.watcher_old_value = old_value
            self.watcher_new_value = new_value
            self.watcher_called_event.set()
    app = WatchAsyncApp()
    async with app.run_test():
        app.count = NEW_VALUE
        assert app.count == NEW_VALUE
        try:
            await asyncio.wait_for(app.watcher_called_event.wait(), timeout=0.05)
        except TimeoutError:
            pytest.fail("Async watch method (watch_count) wasn't called within timeout")
        assert app.count == NEW_VALUE
        assert app.watcher_old_value == OLD_VALUE
        assert app.watcher_new_value == NEW_VALUE

async def test_watch_async_init_true():
    """Ensure that when init is True in a reactive, its async watcher gets called
    by Textual eventually, even when the user does not set the value themselves."""

    class WatchAsyncApp(App):
        count = reactive(OLD_VALUE, init=True)
        watcher_called_event = asyncio.Event()
        watcher_old_value = None
        watcher_new_value = None

        async def watch_count(self, old_value: int, new_value: int) -> None:
            self.watcher_old_value = old_value
            self.watcher_new_value = new_value
            self.watcher_called_event.set()
    app = WatchAsyncApp()
    async with app.run_test():
        try:
            await asyncio.wait_for(app.watcher_called_event.wait(), timeout=0.05)
        except TimeoutError:
            pytest.fail("Async watcher wasn't called within timeout when reactive init = True")
    assert app.count == OLD_VALUE
    assert app.watcher_old_value == OLD_VALUE
    assert app.watcher_new_value == OLD_VALUE

async def test_watch_init_false_always_update_false():

    class WatcherInitFalse(App):
        count = reactive(0, init=False)
        watcher_call_count = 0

        def watch_count(self, new_value: int) -> None:
            if False:
                while True:
                    i = 10
            self.watcher_call_count += 1
    app = WatcherInitFalse()
    async with app.run_test():
        app.count = 0
        assert app.watcher_call_count == 0
        app.count = 0
        assert app.watcher_call_count == 0
        app.count = 1
        assert app.watcher_call_count == 1

async def test_watch_init_true():

    class WatcherInitTrue(App):
        count = var(OLD_VALUE)
        watcher_call_count = 0

        def watch_count(self, new_value: int) -> None:
            if False:
                i = 10
                return i + 15
            self.watcher_call_count += 1
    app = WatcherInitTrue()
    async with app.run_test():
        assert app.count == OLD_VALUE
        assert app.watcher_call_count == 1
        app.count = NEW_VALUE
        assert app.watcher_call_count == 2
        app.count = NEW_VALUE
        assert app.watcher_call_count == 2

async def test_reactive_always_update():
    calls = []

    class AlwaysUpdate(App):
        first_name = reactive('Darren', init=False, always_update=True)
        last_name = reactive('Burns', init=False)

        def watch_first_name(self, value):
            if False:
                while True:
                    i = 10
            calls.append(f'first_name {value}')

        def watch_last_name(self, value):
            if False:
                while True:
                    i = 10
            calls.append(f'last_name {value}')
    app = AlwaysUpdate()
    async with app.run_test():
        app.first_name = 'Darren'
        assert calls == ['first_name Darren']
        app.last_name = 'Burns'
        assert calls == ['first_name Darren']
        app.first_name = 'abc'
        app.last_name = 'def'
        assert calls == ['first_name Darren', 'first_name abc', 'last_name def']

async def test_reactive_with_callable_default():
    """A callable can be supplied as the default value for a reactive.
    Textual will call it in order to retrieve the default value."""

    class ReactiveCallable(App):
        value = reactive(lambda : 123)
        watcher_called_with = None

        def watch_value(self, new_value):
            if False:
                for i in range(10):
                    print('nop')
            self.watcher_called_with = new_value
    app = ReactiveCallable()
    async with app.run_test():
        assert app.value == 123
        assert app.watcher_called_with == 123

async def test_validate_init_true():
    """When init is True for a reactive attribute, Textual should call the validator
    AND the watch method when the app starts."""
    validator_call_count = 0

    class ValidatorInitTrue(App):
        count = var(5, init=True)

        def validate_count(self, value: int) -> int:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal validator_call_count
            validator_call_count += 1
            return value + 1
    app = ValidatorInitTrue()
    async with app.run_test():
        app.count = 5
        assert app.count == 6
        assert validator_call_count == 1

async def test_validate_init_true_set_before_dom_ready():
    """When init is True for a reactive attribute, Textual should call the validator
    AND the watch method when the app starts."""
    validator_call_count = 0

    class ValidatorInitTrue(App):
        count = var(5, init=True)

        def validate_count(self, value: int) -> int:
            if False:
                i = 10
                return i + 15
            nonlocal validator_call_count
            validator_call_count += 1
            return value + 1
    app = ValidatorInitTrue()
    app.count = 5
    async with app.run_test():
        assert app.count == 6
        assert validator_call_count == 1

async def test_reactive_compute_first_time_set():

    class ReactiveComputeFirstTimeSet(App):
        number = reactive(1)
        double_number = reactive(None)

        def compute_double_number(self):
            if False:
                return 10
            return self.number * 2
    app = ReactiveComputeFirstTimeSet()
    async with app.run_test():
        assert app.double_number == 2

async def test_reactive_method_call_order():

    class CallOrder(App):
        count = reactive(OLD_VALUE, init=False)
        count_times_ten = reactive(OLD_VALUE * 10, init=False)
        calls = []

        def validate_count(self, value: int) -> int:
            if False:
                return 10
            self.calls.append(f'validate {value}')
            return value + 1

        def watch_count(self, value: int) -> None:
            if False:
                print('Hello World!')
            self.calls.append(f'watch {value}')

        def compute_count_times_ten(self) -> int:
            if False:
                print('Hello World!')
            self.calls.append(f'compute {self.count}')
            return self.count * 10
    app = CallOrder()
    async with app.run_test():
        app.count = NEW_VALUE
        assert app.calls == [f'validate {NEW_VALUE}', f'watch {NEW_VALUE + 1}', f'compute {NEW_VALUE + 1}']
        assert app.count == NEW_VALUE + 1
        assert app.count_times_ten == (NEW_VALUE + 1) * 10

async def test_premature_reactive_call():
    watcher_called = False

    class BrokenWidget(Widget):
        foo = reactive(1)

        def __init__(self) -> None:
            if False:
                print('Hello World!')
            super().__init__()
            self.foo = 'bar'

        async def watch_foo(self) -> None:
            nonlocal watcher_called
            watcher_called = True

    class PrematureApp(App):

        def compose(self) -> ComposeResult:
            if False:
                i = 10
                return i + 15
            yield BrokenWidget()
    app = PrematureApp()
    async with app.run_test() as pilot:
        assert watcher_called
        app.exit()

async def test_reactive_inheritance():
    """Check that inheritance works as expected for reactives."""

    class Primary(App):
        foo = reactive(1)
        bar = reactive('bar')

    class Secondary(Primary):
        foo = reactive(2)
        egg = reactive('egg')

    class Tertiary(Secondary):
        baz = reactive('baz')
    primary = Primary()
    secondary = Secondary()
    tertiary = Tertiary()
    primary_reactive_count = len(primary._reactives)
    assert len(secondary._reactives) == primary_reactive_count + 1
    Reactive._initialize_object(primary)
    Reactive._initialize_object(secondary)
    Reactive._initialize_object(tertiary)
    with pytest.raises(AttributeError):
        assert primary.egg
    assert primary.foo == 1
    assert secondary.foo == 2
    assert tertiary.foo == 2
    with pytest.raises(AttributeError):
        secondary.baz
    assert tertiary.baz == 'baz'

async def test_compute():
    """Check compute method is called."""

    class ComputeApp(App):
        count = var(0)
        count_double = var(0)

        def __init__(self) -> None:
            if False:
                while True:
                    i = 10
            self.start = 0
            super().__init__()

        def compute_count_double(self) -> int:
            if False:
                while True:
                    i = 10
            return self.start + self.count * 2
    app = ComputeApp()
    async with app.run_test():
        assert app.count_double == 0
        app.count = 1
        assert app.count_double == 2
        assert app.count_double == 2
        app.count = 2
        assert app.count_double == 4
        app.start = 10
        assert app.count_double == 14
        with pytest.raises(AttributeError):
            app.count_double = 100

async def test_watch_compute():
    """Check that watching a computed attribute works."""
    watch_called: list[bool] = []

    class Calculator(App):
        numbers = var('0')
        show_ac = var(True)
        value = var('')

        def compute_show_ac(self) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return self.value in ('', '0') and self.numbers == '0'

        def watch_show_ac(self, show_ac: bool) -> None:
            if False:
                print('Hello World!')
            'Called when show_ac changes.'
            watch_called.append(show_ac)
    app = Calculator()
    async with app.run_test():
        assert app.show_ac is True
        app.value = '1'
        assert app.show_ac is False
        app.value = '0'
        assert app.show_ac is True
        app.numbers = '123'
        assert app.show_ac is False
    assert watch_called == [True, True, False, False, True, True, False, False]

async def test_public_and_private_watch() -> None:
    """If a reactive/var has public and private watches both should get called."""
    calls: dict[str, bool] = {'private': False, 'public': False}

    class PrivateWatchTest(App):
        counter = var(0, init=False)

        def watch_counter(self) -> None:
            if False:
                return 10
            calls['public'] = True

        def _watch_counter(self) -> None:
            if False:
                print('Hello World!')
            calls['private'] = True
    async with PrivateWatchTest().run_test() as pilot:
        assert calls['private'] is False
        assert calls['public'] is False
        pilot.app.counter += 1
        assert calls['private'] is True
        assert calls['public'] is True

async def test_private_validate() -> None:
    calls: dict[str, bool] = {'private': False}

    class PrivateValidateTest(App):
        counter = var(0, init=False)

        def _validate_counter(self, _: int) -> None:
            if False:
                return 10
            calls['private'] = True
    async with PrivateValidateTest().run_test() as pilot:
        assert calls['private'] is False
        pilot.app.counter += 1
        assert calls['private'] is True

async def test_public_and_private_validate() -> None:
    """If a reactive/var has public and private validate both should get called."""
    calls: dict[str, bool] = {'private': False, 'public': False}

    class PrivateValidateTest(App):
        counter = var(0, init=False)

        def validate_counter(self, _: int) -> None:
            if False:
                while True:
                    i = 10
            calls['public'] = True

        def _validate_counter(self, _: int) -> None:
            if False:
                print('Hello World!')
            calls['private'] = True
    async with PrivateValidateTest().run_test() as pilot:
        assert calls['private'] is False
        assert calls['public'] is False
        pilot.app.counter += 1
        assert calls['private'] is True
        assert calls['public'] is True

async def test_public_and_private_validate_order() -> None:
    """The private validate should be called first."""

    class ValidateOrderTest(App):
        value = var(0, init=False)

        def validate_value(self, value: int) -> int:
            if False:
                return 10
            if value < 0:
                return 42
            return value

        def _validate_value(self, value: int) -> int:
            if False:
                while True:
                    i = 10
            if value < 0:
                return 73
            return value
    async with ValidateOrderTest().run_test() as pilot:
        pilot.app.value = -10
        assert pilot.app.value == 73

async def test_public_and_private_compute() -> None:
    """If a reactive/var has public and private compute both should get called."""
    with pytest.raises(TooManyComputesError):

        class PublicAndPrivateComputeTest(App):
            counter = var(0, init=False)

            def compute_counter(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def _compute_counter(self):
                if False:
                    print('Hello World!')
                pass

async def test_private_compute() -> None:

    class PrivateComputeTest(App):
        double = var(0, init=False)
        base = var(0, init=False)

        def _compute_double(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return 2 * self.base
    async with PrivateComputeTest().run_test() as pilot:
        pilot.app.base = 5
        assert pilot.app.double == 10

async def test_async_reactive_watch_callbacks_go_on_the_watcher():
    """Regression test for https://github.com/Textualize/textual/issues/3036.

    This makes sure that async callbacks are called.
    See the next test for sync callbacks.
    """
    from_app = False
    from_holder = False

    class Holder(Widget):
        attr = var(None)

        def watch_attr(self):
            if False:
                print('Hello World!')
            nonlocal from_holder
            from_holder = True

    class MyApp(App):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.holder = Holder()

        def on_mount(self):
            if False:
                i = 10
                return i + 15
            self.watch(self.holder, 'attr', self.callback)

        def update(self):
            if False:
                print('Hello World!')
            self.holder.attr = 'hello world'

        async def callback(self):
            nonlocal from_app
            from_app = True
    async with MyApp().run_test() as pilot:
        pilot.app.update()
        await pilot.pause()
        assert from_holder
        assert from_app

async def test_sync_reactive_watch_callbacks_go_on_the_watcher():
    """Regression test for https://github.com/Textualize/textual/issues/3036.

    This makes sure that sync callbacks are called.
    See the previous test for async callbacks.
    """
    from_app = False
    from_holder = False

    class Holder(Widget):
        attr = var(None)

        def watch_attr(self):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal from_holder
            from_holder = True

    class MyApp(App):

        def __init__(self):
            if False:
                return 10
            super().__init__()
            self.holder = Holder()

        def on_mount(self):
            if False:
                print('Hello World!')
            self.watch(self.holder, 'attr', self.callback)

        def update(self):
            if False:
                while True:
                    i = 10
            self.holder.attr = 'hello world'

        def callback(self):
            if False:
                return 10
            nonlocal from_app
            from_app = True
    async with MyApp().run_test() as pilot:
        pilot.app.update()
        await pilot.pause()
        assert from_holder
        assert from_app