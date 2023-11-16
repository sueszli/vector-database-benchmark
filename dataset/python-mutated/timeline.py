from typing import Literal, Optional
from nicegui.element import Element

class Timeline(Element):

    def __init__(self, *, side: Literal['left', 'right']='left', layout: Literal['dense', 'comfortable', 'loose']='dense', color: Optional[str]=None) -> None:
        if False:
            return 10
        'Timeline\n\n        This element represents `Quasar\'s QTimeline <https://quasar.dev/vue-components/timeline#qtimeline-api>`_ component.\n\n        :param side: Side ("left" or "right"; default: "left").\n        :param layout: Layout ("dense", "comfortable" or "loose"; default: "dense").\n        :param color: Color of the icons.\n        '
        super().__init__('q-timeline')
        self._props['side'] = side
        self._props['layout'] = layout
        if color is not None:
            self._props['color'] = color

class TimelineEntry(Element):

    def __init__(self, body: Optional[str]=None, *, side: Literal['left', 'right']='left', heading: bool=False, tag: Optional[str]=None, icon: Optional[str]=None, avatar: Optional[str]=None, title: Optional[str]=None, subtitle: Optional[str]=None, color: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Timeline Entry\n\n        This element represents `Quasar\'s QTimelineEntry <https://quasar.dev/vue-components/timeline#qtimelineentry-api>`_ component.\n\n        :param body: Body text.\n        :param side: Side ("left" or "right"; default: "left").\n        :param heading: Whether the timeline entry is a heading.\n        :param tag: HTML tag name to be used if it is a heading.\n        :param icon: Icon name.\n        :param avatar: Avatar URL.\n        :param title: Title text.\n        :param subtitle: Subtitle text.\n        :param color: Color or the timeline.\n        '
        super().__init__('q-timeline-entry')
        if body is not None:
            self._props['body'] = body
        self._props['side'] = side
        self._props['heading'] = heading
        if tag is not None:
            self._props['tag'] = tag
        if color is not None:
            self._props['color'] = color
        if icon is not None:
            self._props['icon'] = icon
        if avatar is not None:
            self._props['avatar'] = avatar
        if title is not None:
            self._props['title'] = title
        if subtitle is not None:
            self._props['subtitle'] = subtitle
        self._classes.append('nicegui-timeline-entry')