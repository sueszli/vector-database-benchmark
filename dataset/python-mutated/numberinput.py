"""A number input component."""
from numbers import Number
from typing import Any, Dict
from reflex.components.component import Component
from reflex.components.libs.chakra import ChakraComponent, LiteralInputVariant
from reflex.constants import EventTriggers
from reflex.vars import Var

class NumberInput(ChakraComponent):
    """The wrapper that provides context and logic to the components."""
    tag = 'NumberInput'
    value: Var[Number]
    allow_mouse_wheel: Var[bool]
    clamped_value_on_blur: Var[bool]
    default_value: Var[Number]
    error_border_color: Var[str]
    focus_border_color: Var[str]
    focus_input_on_change: Var[bool]
    is_disabled: Var[bool]
    is_invalid: Var[bool]
    is_read_only: Var[bool]
    is_required: Var[bool]
    is_valid_character: Var[str]
    keep_within_range: Var[bool]
    max_: Var[Number]
    min_: Var[Number]
    variant: Var[LiteralInputVariant]
    name: Var[str]

    def get_event_triggers(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        "Get the event triggers that pass the component's value to the handler.\n\n        Returns:\n            A dict mapping the event trigger to the var that is passed to the handler.\n        "
        return {**super().get_event_triggers(), EventTriggers.ON_CHANGE: lambda e0: [e0]}

    @classmethod
    def create(cls, *children, **props) -> Component:
        if False:
            while True:
                i = 10
        'Create a number input component.\n\n        If no children are provided, a default stepper will be used.\n\n        Args:\n            *children: The children of the component.\n            **props: The props of the component.\n\n        Returns:\n            The component.\n        '
        if len(children) == 0:
            _id = props.pop('id', None)
            children = [NumberInputField.create(id=_id) if _id is not None else NumberInputField.create(), NumberInputStepper.create(NumberIncrementStepper.create(), NumberDecrementStepper.create())]
        return super().create(*children, **props)

class NumberInputField(ChakraComponent):
    """The input field itself."""
    tag = 'NumberInputField'

class NumberInputStepper(ChakraComponent):
    """The wrapper for the input's stepper buttons."""
    tag = 'NumberInputStepper'

class NumberIncrementStepper(ChakraComponent):
    """The button to increment the value of the input."""
    tag = 'NumberIncrementStepper'

class NumberDecrementStepper(ChakraComponent):
    """The button to decrement the value of the input."""
    tag = 'NumberDecrementStepper'