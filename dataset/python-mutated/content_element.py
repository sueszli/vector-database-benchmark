from typing import Any, Callable, cast
from typing_extensions import Self
from ...binding import BindableProperty, bind, bind_from, bind_to
from ...element import Element

class ContentElement(Element):
    CONTENT_PROP = 'innerHTML'
    content = BindableProperty(on_change=lambda sender, content: cast(Self, sender)._handle_content_change(content))

    def __init__(self, *, content: str, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.content = content
        self._handle_content_change(content)

    def bind_content_to(self, target_object: Any, target_name: str='content', forward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            while True:
                i = 10
        "Bind the content of this element to the target object's target_name property.\n\n        The binding works one way only, from this element to the target.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        "
        bind_to(self, 'content', target_object, target_name, forward)
        return self

    def bind_content_from(self, target_object: Any, target_name: str='content', backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            i = 10
            return i + 15
        "Bind the content of this element from the target object's target_name property.\n\n        The binding works one way only, from the target to this element.\n\n        :param target_object: The object to bind from.\n        :param target_name: The name of the property to bind from.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind_from(self, 'content', target_object, target_name, backward)
        return self

    def bind_content(self, target_object: Any, target_name: str='content', *, forward: Callable[..., Any]=lambda x: x, backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            while True:
                i = 10
        "Bind the content of this element to the target object's target_name property.\n\n        The binding works both ways, from this element to the target and from the target to this element.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind(self, 'content', target_object, target_name, forward=forward, backward=backward)
        return self

    def set_content(self, content: str) -> None:
        if False:
            while True:
                i = 10
        'Set the content of this element.\n\n        :param content: The new content.\n        '
        self.content = content

    def _handle_content_change(self, content: str) -> None:
        if False:
            while True:
                i = 10
        'Called when the content of this element changes.\n\n        :param content: The new content.\n        '
        if self.CONTENT_PROP == 'innerHTML' and '</script>' in content:
            raise ValueError('HTML elements must not contain <script> tags. Use ui.add_body_html() instead.')
        self._props[self.CONTENT_PROP] = content
        self.update()