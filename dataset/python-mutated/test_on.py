from __future__ import annotations
from dataclasses import dataclass
import pytest
from textual import on
from textual._on import OnDecoratorError
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, TabbedContent, TabPane

async def test_on_button_pressed() -> None:
    """Test handlers with @on decorator."""
    pressed: list[str] = []

    class ButtonApp(App):

        def compose(self) -> ComposeResult:
            if False:
                print('Hello World!')
            yield Button('OK', id='ok')
            yield Button('Cancel', classes='exit cancel')
            yield Button('Quit', classes='exit quit')

        @on(Button.Pressed, '#ok')
        def ok(self):
            if False:
                for i in range(10):
                    print('nop')
            pressed.append('ok')

        @on(Button.Pressed, '.exit')
        def exit(self):
            if False:
                while True:
                    i = 10
            pressed.append('exit')

        @on(Button.Pressed, '.exit.quit')
        def _(self):
            if False:
                i = 10
                return i + 15
            pressed.append('quit')

        def on_button_pressed(self):
            if False:
                i = 10
                return i + 15
            pressed.append('default')
    app = ButtonApp()
    async with app.run_test() as pilot:
        await pilot.press('enter', 'tab', 'enter', 'tab', 'enter')
        await pilot.pause()
    assert pressed == ['ok', 'default', 'exit', 'default', 'exit', 'quit', 'default']

async def test_on_inheritance() -> None:
    """Test on decorator and inheritance."""
    pressed: list[str] = []

    class MyWidget(Widget):

        def compose(self) -> ComposeResult:
            if False:
                return 10
            yield Button('OK', id='ok')

        @on(Button.Pressed, '#ok')
        def ok(self):
            if False:
                while True:
                    i = 10
            pressed.append('MyWidget.ok base')

    class DerivedWidget(MyWidget):

        @on(Button.Pressed, '#ok')
        def ok(self):
            if False:
                for i in range(10):
                    print('nop')
            pressed.append('MyWidget.ok derived')

    class ButtonApp(App):

        def compose(self) -> ComposeResult:
            if False:
                return 10
            yield DerivedWidget()
    app = ButtonApp()
    async with app.run_test() as pilot:
        await pilot.press('tab', 'enter')
    expected = ['MyWidget.ok derived', 'MyWidget.ok base']
    assert pressed == expected

def test_on_bad_selector() -> None:
    if False:
        while True:
            i = 10
    'Check bad selectors raise an error.'
    with pytest.raises(OnDecoratorError):

        @on(Button.Pressed, '@')
        def foo():
            if False:
                i = 10
                return i + 15
            pass

def test_on_no_control() -> None:
    if False:
        for i in range(10):
            print('nop')
    "Check messages with no 'control' attribute raise an error."

    class CustomMessage(Message):
        pass
    with pytest.raises(OnDecoratorError):

        @on(CustomMessage, '#foo')
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            pass

def test_on_attribute_not_listed() -> None:
    if False:
        print('Hello World!')
    'Check `on` checks if the attribute is in ALLOW_SELECTOR_MATCH.'

    class CustomMessage(Message):
        pass
    with pytest.raises(OnDecoratorError):

        @on(CustomMessage, foo='bar')
        def foo():
            if False:
                return 10
            pass

async def test_on_arbitrary_attributes() -> None:
    log: list[str] = []

    class OnArbitraryAttributesApp(App[None]):

        def compose(self) -> ComposeResult:
            if False:
                print('Hello World!')
            with TabbedContent():
                yield TabPane('One', id='one')
                yield TabPane('Two', id='two')
                yield TabPane('Three', id='three')

        def on_mount(self) -> None:
            if False:
                i = 10
                return i + 15
            self.query_one(TabbedContent).add_class('tabs')

        @on(TabbedContent.TabActivated, tab='#one')
        def one(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            log.append('one')

        @on(TabbedContent.TabActivated, '.tabs', tab='#two')
        def two(self) -> None:
            if False:
                print('Hello World!')
            log.append('two')
    app = OnArbitraryAttributesApp()
    async with app.run_test() as pilot:
        await pilot.press('tab', 'right', 'right')
    assert log == ['one', 'two']

class MessageSender(Widget):

    @dataclass
    class Parent(Message):
        sender: MessageSender

        @property
        def control(self) -> MessageSender:
            if False:
                i = 10
                return i + 15
            return self.sender

    class Child(Parent):
        pass

    def post_parent(self) -> None:
        if False:
            return 10
        self.post_message(self.Parent(self))

    def post_child(self) -> None:
        if False:
            return 10
        self.post_message(self.Child(self))

async def test_fire_on_inherited_message() -> None:
    """Handlers should fire when descendant messages are posted."""
    posted: list[str] = []

    class InheritTestApp(App[None]):

        def compose(self) -> ComposeResult:
            if False:
                return 10
            yield MessageSender()

        @on(MessageSender.Parent)
        def catch_parent(self) -> None:
            if False:
                while True:
                    i = 10
            posted.append('parent')

        @on(MessageSender.Child)
        def catch_child(self) -> None:
            if False:
                print('Hello World!')
            posted.append('child')

        def on_mount(self) -> None:
            if False:
                i = 10
                return i + 15
            self.query_one(MessageSender).post_parent()
            self.query_one(MessageSender).post_child()
    async with InheritTestApp().run_test():
        pass
    assert posted == ['parent', 'child', 'parent']

async def test_fire_inherited_on_single_handler() -> None:
    """Test having parent/child messages on a single handler."""
    posted: list[str] = []

    class InheritTestApp(App[None]):

        def compose(self) -> ComposeResult:
            if False:
                print('Hello World!')
            yield MessageSender()

        @on(MessageSender.Parent)
        @on(MessageSender.Child)
        def catch_either(self, event: MessageSender.Parent) -> None:
            if False:
                print('Hello World!')
            posted.append(f'either {event.__class__.__name__}')

        def on_mount(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.query_one(MessageSender).post_parent()
            self.query_one(MessageSender).post_child()
    async with InheritTestApp().run_test():
        pass
    assert posted == ['either Parent', 'either Child']

async def test_fire_inherited_on_single_handler_multi_selector() -> None:
    """Test having parent/child messages on a single handler but with different selectors."""
    posted: list[str] = []

    class InheritTestApp(App[None]):

        def compose(self) -> ComposeResult:
            if False:
                i = 10
                return i + 15
            yield MessageSender(classes='a b')

        @on(MessageSender.Parent, '.y')
        @on(MessageSender.Child, '.y')
        @on(MessageSender.Parent, '.a.b')
        @on(MessageSender.Child, '.a.b')
        @on(MessageSender.Parent, '.a')
        @on(MessageSender.Child, '.a')
        @on(MessageSender.Parent, '.b')
        @on(MessageSender.Child, '.b')
        @on(MessageSender.Parent, '.x')
        @on(MessageSender.Child, '.x')
        def catch_either(self, event: MessageSender.Parent) -> None:
            if False:
                for i in range(10):
                    print('nop')
            posted.append(f'either {event.__class__.__name__}')

        @on(MessageSender.Child, '.a, .x')
        def catch_selector_list_one_miss(self, event: MessageSender.Parent) -> None:
            if False:
                i = 10
                return i + 15
            posted.append(f'selector list one miss {event.__class__.__name__}')

        @on(MessageSender.Child, '.a, .b')
        def catch_selector_list_two_hits(self, event: MessageSender.Parent) -> None:
            if False:
                return 10
            posted.append(f'selector list two hits {event.__class__.__name__}')

        @on(MessageSender.Child, '.a.b')
        def catch_selector_combined_hits(self, event: MessageSender.Parent) -> None:
            if False:
                while True:
                    i = 10
            posted.append(f'combined hits {event.__class__.__name__}')

        @on(MessageSender.Child, '.a.x')
        def catch_selector_combined_miss(self, event: MessageSender.Parent) -> None:
            if False:
                return 10
            posted.append(f'combined miss {event.__class__.__name__}')

        def on_mount(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.query_one(MessageSender).post_parent()
            self.query_one(MessageSender).post_child()
    async with InheritTestApp().run_test():
        pass
    assert posted == ['either Parent', 'either Child', 'selector list one miss Child', 'selector list two hits Child', 'combined hits Child']

async def test_fire_inherited_and_on_methods() -> None:
    posted: list[str] = []

    class OnAndOnTestApp(App[None]):

        def compose(self) -> ComposeResult:
            if False:
                for i in range(10):
                    print('nop')
            yield MessageSender()

        def on_message_sender_parent(self) -> None:
            if False:
                return 10
            posted.append('on_message_sender_parent')

        @on(MessageSender.Parent)
        def catch_parent(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            posted.append('catch_parent')

        def on_message_sender_child(self) -> None:
            if False:
                return 10
            posted.append('on_message_sender_child')

        @on(MessageSender.Child)
        def catch_child(self) -> None:
            if False:
                while True:
                    i = 10
            posted.append('catch_child')

        def on_mount(self) -> None:
            if False:
                i = 10
                return i + 15
            self.query_one(MessageSender).post_parent()
            self.query_one(MessageSender).post_child()
    async with OnAndOnTestApp().run_test():
        pass
    assert posted == ['catch_parent', 'on_message_sender_parent', 'catch_child', 'catch_parent', 'on_message_sender_child']

class MixinMessageSender(Widget):

    class Parent(Message):
        pass

    class JustSomeRandomMixin:
        pass

    class Child(JustSomeRandomMixin, Parent):
        pass

    def post_parent(self) -> None:
        if False:
            while True:
                i = 10
        self.post_message(self.Parent())

    def post_child(self) -> None:
        if False:
            print('Hello World!')
        self.post_message(self.Child())

async def test_fire_on_inherited_message_plus_mixins() -> None:
    """Handlers should fire when descendant messages are posted, without mixins messing things up."""
    posted: list[str] = []

    class InheritTestApp(App[None]):

        def compose(self) -> ComposeResult:
            if False:
                for i in range(10):
                    print('nop')
            yield MixinMessageSender()

        @on(MixinMessageSender.Parent)
        def catch_parent(self) -> None:
            if False:
                while True:
                    i = 10
            posted.append('parent')

        @on(MixinMessageSender.Child)
        def catch_child(self) -> None:
            if False:
                print('Hello World!')
            posted.append('child')

        def on_mount(self) -> None:
            if False:
                i = 10
                return i + 15
            self.query_one(MixinMessageSender).post_parent()
            self.query_one(MixinMessageSender).post_child()
    async with InheritTestApp().run_test():
        pass
    assert posted == ['parent', 'child', 'parent']