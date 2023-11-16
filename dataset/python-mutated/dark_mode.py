from typing import Any, Callable, Optional
from .mixins.value_element import ValueElement

class DarkMode(ValueElement, component='dark_mode.js'):
    VALUE_PROP = 'value'

    def __init__(self, value: Optional[bool]=False, *, on_change: Optional[Callable[..., Any]]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Dark mode\n\n        You can use this element to enable, disable or toggle dark mode on the page.\n        The value `None` represents auto mode, which uses the client's system preference.\n\n        Note that this element overrides the `dark` parameter of the `ui.run` function and page decorators.\n\n        :param value: Whether dark mode is enabled. If None, dark mode is set to auto.\n        :param on_change: Callback that is invoked when the value changes.\n        "
        super().__init__(value=value, on_value_change=on_change)

    def enable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Enable dark mode.'
        self.value = True

    def disable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Disable dark mode.'
        self.value = False

    def toggle(self) -> None:
        if False:
            return 10
        'Toggle dark mode.\n\n        This method will raise a ValueError if dark mode is set to auto.\n        '
        if self.value is None:
            raise ValueError('Cannot toggle dark mode when it is set to auto.')
        self.value = not self.value

    def auto(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Set dark mode to auto.\n\n        This will use the client's system preference.\n        "
        self.value = None