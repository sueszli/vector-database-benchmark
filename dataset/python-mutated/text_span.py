from typing import Any, List, Optional
from flet_core.inline_span import InlineSpan
from flet_core.text_style import TextStyle

class TextSpan(InlineSpan):

    def __init__(self, text: Optional[str]=None, style: Optional[TextStyle]=None, spans: Optional[List[InlineSpan]]=None, url: Optional[str]=None, url_target: Optional[str]=None, on_click=None, on_enter=None, on_exit=None, ref=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None):
        if False:
            print('Hello World!')
        InlineSpan.__init__(self, ref=ref, visible=visible, disabled=disabled, data=data)
        self.text = text
        self.style = style
        self.spans = spans
        self.url = url
        self.url_target = url_target
        self.on_click = on_click
        self.on_enter = on_enter
        self.on_exit = on_exit

    def _get_control_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'textspan'

    def _get_children(self):
        if False:
            while True:
                i = 10
        children = []
        children.extend(self.__spans)
        return children

    def _before_build_command(self):
        if False:
            return 10
        super()._before_build_command()
        self._set_attr_json('style', self.__style)

    @property
    def text(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('text')

    @text.setter
    def text(self, value: Optional[str]):
        if False:
            i = 10
            return i + 15
        self._set_attr('text', value)

    @property
    def style(self) -> Optional[TextStyle]:
        if False:
            return 10
        return self.__style

    @style.setter
    def style(self, value: Optional[TextStyle]):
        if False:
            i = 10
            return i + 15
        self.__style = value

    @property
    def spans(self) -> Optional[List[InlineSpan]]:
        if False:
            for i in range(10):
                print('nop')
        return self.__spans

    @spans.setter
    def spans(self, value: Optional[List[InlineSpan]]):
        if False:
            print('Hello World!')
        self.__spans = value if value is not None else []

    @property
    def url(self):
        if False:
            print('Hello World!')
        return self._get_attr('url')

    @url.setter
    def url(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('url', value)

    @property
    def url_target(self):
        if False:
            print('Hello World!')
        return self._get_attr('urlTarget')

    @url_target.setter
    def url_target(self, value):
        if False:
            print('Hello World!')
        self._set_attr('urlTarget', value)

    @property
    def on_click(self):
        if False:
            i = 10
            return i + 15
        return self._get_event_handler('click')

    @on_click.setter
    def on_click(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self._add_event_handler('click', handler)
        self._set_attr('onClick', True if handler is not None else None)

    @property
    def on_enter(self):
        if False:
            while True:
                i = 10
        return self._get_event_handler('enter')

    @on_enter.setter
    def on_enter(self, handler):
        if False:
            return 10
        self._add_event_handler('enter', handler)
        self._set_attr('onEnter', True if handler is not None else None)

    @property
    def on_exit(self):
        if False:
            while True:
                i = 10
        return self._get_event_handler('exit')

    @on_exit.setter
    def on_exit(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self._add_event_handler('exit', handler)
        self._set_attr('onExit', True if handler is not None else None)