import json
import time
from typing import Any, Optional
from flet_core.animation import AnimationCurve
from flet_core.control import Control, OptionalNumber
from flet_core.control_event import ControlEvent
from flet_core.event_handler import EventHandler
from flet_core.types import ScrollMode, ScrollModeString

class ScrollableControl(Control):

    def __init__(self, scroll: Optional[ScrollMode]=None, auto_scroll: Optional[bool]=None, on_scroll_interval: OptionalNumber=None, on_scroll: Any=None):
        if False:
            return 10

        def convert_on_scroll_event_data(e):
            if False:
                i = 10
                return i + 15
            d = json.loads(e.data)
            return OnScrollEvent(**d)
        self.__on_scroll = EventHandler(convert_on_scroll_event_data)
        self._add_event_handler('onScroll', self.__on_scroll.get_handler())
        self.scroll = scroll
        self.auto_scroll = auto_scroll
        self.on_scroll_interval = on_scroll_interval
        self.on_scroll = on_scroll

    def scroll_to(self, offset: Optional[float]=None, delta: Optional[float]=None, key: Optional[str]=None, duration: Optional[int]=None, curve: Optional[AnimationCurve]=None):
        if False:
            return 10
        m = {'n': 'scroll_to', 'i': str(time.time()), 'p': {'offset': offset, 'delta': delta, 'key': key, 'duration': duration, 'curve': curve.value if curve is not None else None}}
        self._set_attr_json('method', m)
        self.update()

    async def scroll_to_async(self, offset: Optional[float]=None, delta: Optional[float]=None, key: Optional[str]=None, duration: Optional[int]=None, curve: Optional[AnimationCurve]=None):
        m = {'n': 'scroll_to', 'i': str(time.time()), 'p': {'offset': offset, 'delta': delta, 'key': key, 'duration': duration, 'curve': curve.value if curve is not None else None}}
        self._set_attr_json('method', m)
        await self.update_async()

    @property
    def scroll(self) -> Optional[ScrollMode]:
        if False:
            while True:
                i = 10
        return self.__scroll

    @scroll.setter
    def scroll(self, value: Optional[ScrollMode]):
        if False:
            return 10
        self.__scroll = value
        if isinstance(value, ScrollMode):
            self._set_attr('scroll', value.value)
        else:
            self.__set_scroll(value)

    def __set_scroll(self, value: Optional[ScrollModeString]):
        if False:
            i = 10
            return i + 15
        if value is True:
            value = 'auto'
        elif value is False:
            value = None
        self._set_attr('scroll', value)

    @property
    def auto_scroll(self) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('autoScroll')

    @auto_scroll.setter
    def auto_scroll(self, value: Optional[bool]):
        if False:
            while True:
                i = 10
        self._set_attr('autoScroll', value)

    @property
    def on_scroll_interval(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('onScrollInterval')

    @on_scroll_interval.setter
    def on_scroll_interval(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('onScrollInterval', value)

    @property
    def on_scroll(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__on_scroll

    @on_scroll.setter
    def on_scroll(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self.__on_scroll.subscribe(handler)
        self._set_attr('onScroll', True if handler is not None else None)

class OnScrollEvent(ControlEvent):

    def __init__(self, t, p, minse, maxse, vd, sd=None, dir=None, os=None, v=None) -> None:
        if False:
            while True:
                i = 10
        self.event_type: str = t
        self.pixels: float = p
        self.min_scroll_extent: float = minse
        self.max_scroll_extent: float = maxse
        self.viewport_dimension: float = vd
        self.scroll_delta: Optional[float] = sd
        self.direction: Optional[str] = dir
        self.overscroll: Optional[float] = os
        self.velocity: Optional[float] = v

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        attrs = {}
        return f'{self.event_type}: pixels={self.pixels}, min_scroll_extent={self.min_scroll_extent}, max_scroll_extent={self.max_scroll_extent}, viewport_dimension={self.viewport_dimension}, scroll_delta={self.scroll_delta}, direction={self.direction}, overscroll={self.overscroll}, velocity={self.velocity}'