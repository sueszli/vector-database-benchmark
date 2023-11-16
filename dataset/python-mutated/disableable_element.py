from typing import Any, Callable, cast
from typing_extensions import Self
from ...binding import BindableProperty, bind, bind_from, bind_to
from ...element import Element

class DisableableElement(Element):
    enabled = BindableProperty(on_change=lambda sender, value: cast(Self, sender)._handle_enabled_change(value))

    def __init__(self, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.enabled = True
        self.ignores_events_when_disabled = True

    @property
    def is_ignoring_events(self) -> bool:
        if False:
            while True:
                i = 10
        'Return whether the element is currently ignoring events.'
        if super().is_ignoring_events:
            return True
        return not self.enabled and self.ignores_events_when_disabled

    def enable(self) -> None:
        if False:
            return 10
        'Enable the element.'
        self.enabled = True

    def disable(self) -> None:
        if False:
            i = 10
            return i + 15
        'Disable the element.'
        self.enabled = False

    def bind_enabled_to(self, target_object: Any, target_name: str='enabled', forward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            i = 10
            return i + 15
        "Bind the enabled state of this element to the target object's target_name property.\n\n        The binding works one way only, from this element to the target.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        "
        bind_to(self, 'enabled', target_object, target_name, forward)
        return self

    def bind_enabled_from(self, target_object: Any, target_name: str='enabled', backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            return 10
        "Bind the enabled state of this element from the target object's target_name property.\n\n        The binding works one way only, from the target to this element.\n\n        :param target_object: The object to bind from.\n        :param target_name: The name of the property to bind from.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind_from(self, 'enabled', target_object, target_name, backward)
        return self

    def bind_enabled(self, target_object: Any, target_name: str='enabled', *, forward: Callable[..., Any]=lambda x: x, backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            while True:
                i = 10
        "Bind the enabled state of this element to the target object's target_name property.\n\n        The binding works both ways, from this element to the target and from the target to this element.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind(self, 'enabled', target_object, target_name, forward=forward, backward=backward)
        return self

    def set_enabled(self, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the enabled state of the element.'
        self.enabled = value

    def _handle_enabled_change(self, enabled: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Called when the element is enabled or disabled.\n\n        :param enabled: The new state.\n        '
        self._props['disable'] = not enabled
        self.update()