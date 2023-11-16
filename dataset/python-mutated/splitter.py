from typing import Any, Callable, Optional, Tuple
from .mixins.disableable_element import DisableableElement
from .mixins.value_element import ValueElement

class Splitter(ValueElement, DisableableElement):

    def __init__(self, *, horizontal: Optional[bool]=False, reverse: Optional[bool]=False, limits: Optional[Tuple[float, float]]=(0, 100), value: Optional[float]=50, on_change: Optional[Callable[..., Any]]=None) -> None:
        if False:
            while True:
                i = 10
        "Splitter\n\n        The `ui.splitter` element divides the screen space into resizable sections, \n        allowing for flexible and responsive layouts in your application.\n\n        Based on Quasar's Splitter component:\n        `Splitter <https://quasar.dev/vue-components/splitter>`_\n\n        It provides three customizable slots, ``before``, ``after``, and ``separator``,\n        which can be used to embed other elements within the splitter.\n\n        :param horizontal: Whether to split horizontally instead of vertically\n        :param limits: Two numbers representing the minimum and maximum split size of the two panels\n        :param value: Size of the first panel (or second if using reverse)\n        :param reverse: Whether to apply the model size to the second panel instead of the first\n        :param on_change: callback which is invoked when the user releases the splitter\n        "
        super().__init__(tag='q-splitter', value=value, on_value_change=on_change, throttle=0.05)
        self._props['horizontal'] = horizontal
        self._props['limits'] = limits
        self._props['reverse'] = reverse
        self._classes.append('nicegui-splitter')
        self.before = self.add_slot('before')
        self.after = self.add_slot('after')
        self.separator = self.add_slot('separator')