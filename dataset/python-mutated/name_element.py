from typing import Any, Callable, cast
from typing_extensions import Self
from ...binding import BindableProperty, bind, bind_from, bind_to
from ...element import Element

class NameElement(Element):
    name = BindableProperty(on_change=lambda sender, name: cast(Self, sender)._handle_name_change(name))

    def __init__(self, *, name: str, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.name = name
        self._props['name'] = name

    def bind_name_to(self, target_object: Any, target_name: str='name', forward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            return 10
        "Bind the name of this element to the target object's target_name property.\n\n        The binding works one way only, from this element to the target.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        "
        bind_to(self, 'name', target_object, target_name, forward)
        return self

    def bind_name_from(self, target_object: Any, target_name: str='name', backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            print('Hello World!')
        "Bind the name of this element from the target object's target_name property.\n\n        The binding works one way only, from the target to this element.\n\n        :param target_object: The object to bind from.\n        :param target_name: The name of the property to bind from.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind_from(self, 'name', target_object, target_name, backward)
        return self

    def bind_name(self, target_object: Any, target_name: str='name', *, forward: Callable[..., Any]=lambda x: x, backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            for i in range(10):
                print('nop')
        "Bind the name of this element to the target object's target_name property.\n\n        The binding works both ways, from this element to the target and from the target to this element.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind(self, 'name', target_object, target_name, forward=forward, backward=backward)
        return self

    def set_name(self, name: str) -> None:
        if False:
            i = 10
            return i + 15
        'Set the name of this element.\n\n        :param name: The new name.\n        '
        self.name = name

    def _handle_name_change(self, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Called when the name of this element changes.\n\n        :param name: The new name.\n        '
        self._props['name'] = name
        self.update()