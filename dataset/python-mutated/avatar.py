from typing import Optional
from .mixins.color_elements import BackgroundColorElement, TextColorElement

class Avatar(BackgroundColorElement, TextColorElement):
    TEXT_COLOR_PROP = 'text-color'

    def __init__(self, icon: Optional[str]=None, *, color: Optional[str]='primary', text_color: Optional[str]=None, size: Optional[str]=None, font_size: Optional[str]=None, square: bool=False, rounded: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Avatar\n\n        A avatar element wrapping Quasar\'s\n        `QAvatar <https://quasar.dev/vue-components/avatar>`_ component.\n\n        :param icon: name of the icon or image path with "img:" prefix (e.g. "map", "img:path/to/image.png")\n        :param color: background color (either a Quasar, Tailwind, or CSS color or `None`, default: "primary")\n        :param text_color: color name from the Quasar Color Palette (e.g. "primary", "teal-10")\n        :param size: size in CSS units, including unit name or standard size name (xs|sm|md|lg|xl) (e.g. "16px", "2rem")\n        :param font_size: size in CSS units, including unit name, of the content (icon, text) (e.g. "18px", "2rem")\n        :param square: removes border-radius so borders are squared (default: False)\n        :param rounded: applies a small standard border-radius for a squared shape of the component (default: False)\n        '
        super().__init__(tag='q-avatar', background_color=color, text_color=text_color)
        if icon is not None:
            self._props['icon'] = icon
        self._props['square'] = square
        self._props['rounded'] = rounded
        if size is not None:
            self._props['size'] = size
        if font_size is not None:
            self._props['font-size'] = font_size