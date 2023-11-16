from typing import Any, Callable, cast
from typing_extensions import Self
from ...binding import BindableProperty, bind, bind_from, bind_to
from ...element import Element

class TextElement(Element):
    text = BindableProperty(on_change=lambda sender, text: cast(Self, sender)._handle_text_change(text))

    def __init__(self, *, text: str, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.text = text
        self._text_to_model_text(text)

    def bind_text_to(self, target_object: Any, target_name: str='text', forward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            for i in range(10):
                print('nop')
        "Bind the text of this element to the target object's target_name property.\n\n        The binding works one way only, from this element to the target.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        "
        bind_to(self, 'text', target_object, target_name, forward)
        return self

    def bind_text_from(self, target_object: Any, target_name: str='text', backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            print('Hello World!')
        "Bind the text of this element from the target object's target_name property.\n\n        The binding works one way only, from the target to this element.\n\n        :param target_object: The object to bind from.\n        :param target_name: The name of the property to bind from.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind_from(self, 'text', target_object, target_name, backward)
        return self

    def bind_text(self, target_object: Any, target_name: str='text', *, forward: Callable[..., Any]=lambda x: x, backward: Callable[..., Any]=lambda x: x) -> Self:
        if False:
            i = 10
            return i + 15
        "Bind the text of this element to the target object's target_name property.\n\n        The binding works both ways, from this element to the target and from the target to this element.\n\n        :param target_object: The object to bind to.\n        :param target_name: The name of the property to bind to.\n        :param forward: A function to apply to the value before applying it to the target.\n        :param backward: A function to apply to the value before applying it to this element.\n        "
        bind(self, 'text', target_object, target_name, forward=forward, backward=backward)
        return self

    def set_text(self, text: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the text of this element.\n\n        :param text: The new text.\n        '
        self.text = text

    def _handle_text_change(self, text: str) -> None:
        if False:
            return 10
        'Called when the text of this element changes.\n\n        :param text: The new text.\n        '
        self._text_to_model_text(text)
        self.update()

    def _text_to_model_text(self, text: str) -> None:
        if False:
            while True:
                i = 10
        self._text = text