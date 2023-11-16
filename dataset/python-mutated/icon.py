from typing import Optional
from .mixins.color_elements import TextColorElement
from .mixins.name_element import NameElement

class Icon(NameElement, TextColorElement):

    def __init__(self, name: str, *, size: Optional[str]=None, color: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Icon\n\n        This element is based on Quasar's `QIcon <https://quasar.dev/vue-components/icon>`_ component.\n\n        `Here <https://fonts.google.com/icons?icon.set=Material+Icons>`_ is a reference of possible names.\n\n        :param name: name of the icon (snake case, e.g. `add_circle`)\n        :param size: size in CSS units, including unit name or standard size name (xs|sm|md|lg|xl), examples: 16px, 2rem\n        :param color: icon color (either a Quasar, Tailwind, or CSS color or `None`, default: `None`)\n        "
        super().__init__(tag='q-icon', name=name, text_color=color)
        if size:
            self._props['size'] = size