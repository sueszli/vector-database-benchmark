from typing import Any, Callable, Optional, cast
from typing_extensions import Self
from ...binding import BindableProperty, bind, bind_from, bind_to
from ...element import Element

class FilterElement(Element):
    FILTER_PROP = 'filter'
    filter = BindableProperty(on_change=lambda sender, filter: cast(Self, sender)._handle_filter_change(filter))

    def __init__(self, *, filter: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.filter = filter
        self._props[self.FILTER_PROP] = filter

    def bind_filter_to(self, target_object: Any, target_name: str='filter', forward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            i = 10
            return i + 15
        "Bind the filter of this element to the target object's target_name property.\n\n        The binding works one way only, from this element to the target.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        "
        bind_to(self, 'filter', target_object, target_name, forward)
        return self

    def bind_filter_from(self, target_object: Any, target_name: str='filter', backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            i = 10
            return i + 15
        "Bind the filter of this element from the target object's target_name property.\n\n        The binding works one way only, from the target to this element.\n\n        :param target_object: The object to bind from.\n        :param target_name: The name of the property to bind from.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind_from(self, 'filter', target_object, target_name, backward)
        return self

    def bind_filter(self, target_object: Any, target_name: str='filter', *, forward: Callable[..., Any]=lambda x: x, backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            for i in range(10):
                print('nop')
        "Bind the filter of this element to the target object's target_name property.\n\n        The binding works both ways, from this element to the target and from the target to this element.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind(self, 'filter', target_object, target_name, forward=forward, backward=backward)
        return self

    def set_filter(self, filter_: str) -> None:
        if False:
            return 10
        'Set the filter of this element.\n\n        :param filter: The new filter.\n        '
        self.filter = filter_

    def _handle_filter_change(self, filter_: str) -> None:
        if False:
            return 10
        'Called when the filter of this element changes.\n\n        :param filter: The new filter.\n        '
        self._props[self.FILTER_PROP] = filter_
        self.update()