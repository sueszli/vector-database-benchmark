from typing import Any, Callable, Dict, List, Optional, Union
from .mixins.value_element import ValueElement

class ChoiceElement(ValueElement):

    def __init__(self, *, tag: Optional[str]=None, options: Union[List, Dict], value: Any, on_change: Optional[Callable[..., Any]]=None) -> None:
        if False:
            while True:
                i = 10
        self.options = options
        self._values: List[str] = []
        self._labels: List[str] = []
        self._update_values_and_labels()
        super().__init__(tag=tag, value=value, on_value_change=on_change)
        self._update_options()

    def _update_values_and_labels(self) -> None:
        if False:
            i = 10
            return i + 15
        self._values = self.options if isinstance(self.options, list) else list(self.options.keys())
        self._labels = self.options if isinstance(self.options, list) else list(self.options.values())

    def _update_options(self) -> None:
        if False:
            i = 10
            return i + 15
        before_value = self.value
        self._props['options'] = [{'value': index, 'label': option} for (index, option) in enumerate(self._labels)]
        if not isinstance(before_value, list):
            self._props[self.VALUE_PROP] = self._value_to_model_value(before_value)
            self.value = before_value if before_value in self._values else None

    def update(self) -> None:
        if False:
            return 10
        self._update_values_and_labels()
        self._update_options()
        super().update()

    def set_options(self, options: Union[List, Dict], *, value: Any=None) -> None:
        if False:
            print('Hello World!')
        'Set the options of this choice element.\n\n        :param options: The new options.\n        :param value: The new value. If not given, the current value is kept.\n        '
        self.options = options
        if value is not None:
            self.value = value
        self.update()