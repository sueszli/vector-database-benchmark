from typing import Any, Callable, Dict, Optional
from .input import Input

class Textarea(Input, component='input.js'):

    def __init__(self, label: Optional[str]=None, *, placeholder: Optional[str]=None, value: str='', on_change: Optional[Callable[..., Any]]=None, validation: Dict[str, Callable[..., bool]]={}) -> None:
        if False:
            print('Hello World!')
        "Textarea\n\n        This element is based on Quasar's `QInput <https://quasar.dev/vue-components/input>`_ component.\n        The ``type`` is set to ``textarea`` to create a multi-line text input.\n\n        You can use the `validation` parameter to define a dictionary of validation rules.\n        The key of the first rule that fails will be displayed as an error message.\n\n        :param label: displayed name for the textarea\n        :param placeholder: text to show if no value is entered\n        :param value: the initial value of the field\n        :param on_change: callback to execute when the value changes\n        :param validation: dictionary of validation rules, e.g. ``{'Too long!': lambda value: len(value) < 3}``\n        "
        super().__init__(label, placeholder=placeholder, value=value, on_change=on_change, validation=validation)
        self._props['type'] = 'textarea'