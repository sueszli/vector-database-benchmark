from typing import Any, Callable, Optional, cast
from typing_extensions import Self
from ...binding import BindableProperty, bind, bind_from, bind_to
from ...element import Element
from ...events import GenericEventArguments, ValueChangeEventArguments, handle_event

class ValueElement(Element):
    VALUE_PROP: str = 'model-value'
    LOOPBACK: bool = True
    value = BindableProperty(on_change=lambda sender, value: cast(Self, sender)._handle_value_change(value))

    def __init__(self, *, value: Any, on_value_change: Optional[Callable[..., Any]], throttle: float=0, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.set_value(value)
        self._props[self.VALUE_PROP] = self._value_to_model_value(value)
        self._props['loopback'] = self.LOOPBACK
        self._send_update_on_value_change = True
        self._change_handler = on_value_change

        def handle_change(e: GenericEventArguments) -> None:
            if False:
                i = 10
                return i + 15
            self._send_update_on_value_change = self.LOOPBACK
            self.set_value(self._event_args_to_value(e))
            self._send_update_on_value_change = True
        self.on(f'update:{self.VALUE_PROP}', handle_change, [None], throttle=throttle)

    def bind_value_to(self, target_object: Any, target_name: str='value', forward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            return 10
        "Bind the value of this element to the target object's target_name property.\n\n        The binding works one way only, from this element to the target.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        "
        bind_to(self, 'value', target_object, target_name, forward)
        return self

    def bind_value_from(self, target_object: Any, target_name: str='value', backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            i = 10
            return i + 15
        "Bind the value of this element from the target object's target_name property.\n\n        The binding works one way only, from the target to this element.\n\n        :param target_object: The object to bind from.\n        :param target_name: The name of the property to bind from.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind_from(self, 'value', target_object, target_name, backward)
        return self

    def bind_value(self, target_object: Any, target_name: str='value', *, forward: Callable[..., Any]=lambda x: x, backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            return 10
        "Bind the value of this element to the target object's target_name property.\n\n        The binding works both ways, from this element to the target and from the target to this element.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind(self, 'value', target_object, target_name, forward=forward, backward=backward)
        return self

    def set_value(self, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the value of this element.\n\n        :param value: The value to set.\n        '
        self.value = value

    def _handle_value_change(self, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._props[self.VALUE_PROP] = self._value_to_model_value(value)
        if self._send_update_on_value_change:
            self.update()
        args = ValueChangeEventArguments(sender=self, client=self.client, value=self._value_to_event_value(value))
        handle_event(self._change_handler, args)

    def _event_args_to_value(self, e: GenericEventArguments) -> Any:
        if False:
            i = 10
            return i + 15
        return e.args

    def _value_to_model_value(self, value: Any) -> Any:
        if False:
            i = 10
            return i + 15
        return value

    def _value_to_event_value(self, value: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return value