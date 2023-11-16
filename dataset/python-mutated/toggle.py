from typing import Any, Callable, Dict, List, Optional, Union
from ..events import GenericEventArguments
from .choice_element import ChoiceElement
from .mixins.disableable_element import DisableableElement

class Toggle(ChoiceElement, DisableableElement):

    def __init__(self, options: Union[List, Dict], *, value: Any=None, on_change: Optional[Callable[..., Any]]=None, clearable: bool=False) -> None:
        if False:
            while True:
                i = 10
        "Toggle\n\n        This element is based on Quasar's `QBtnToggle <https://quasar.dev/vue-components/button-toggle>`_ component.\n\n        The options can be specified as a list of values, or as a dictionary mapping values to labels.\n        After manipulating the options, call `update()` to update the options in the UI.\n\n        :param options: a list ['value1', ...] or dictionary `{'value1':'label1', ...}` specifying the options\n        :param value: the initial value\n        :param on_change: callback to execute when selection changes\n        :param clearable: whether the toggle can be cleared by clicking the selected option\n        "
        super().__init__(tag='q-btn-toggle', options=options, value=value, on_change=on_change)
        self._props['clearable'] = clearable

    def _event_args_to_value(self, e: GenericEventArguments) -> Any:
        if False:
            print('Hello World!')
        return self._values[e.args] if e.args is not None else None

    def _value_to_model_value(self, value: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return self._values.index(value) if value in self._values else None