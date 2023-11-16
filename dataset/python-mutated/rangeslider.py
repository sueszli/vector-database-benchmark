"""A range slider component."""
from __future__ import annotations
from typing import Any, List, Optional, Union
from reflex.components.component import Component
from reflex.components.libs.chakra import ChakraComponent, LiteralChakraDirection
from reflex.constants import EventTriggers
from reflex.utils import format
from reflex.vars import Var

class RangeSlider(ChakraComponent):
    """The RangeSlider is a multi thumb slider used to select a range of related values. A common use-case of this component is a price range picker that allows a user to set the minimum and maximum price."""
    tag = 'RangeSlider'
    value: Var[List[int]]
    default_value: Var[List[int]]
    direction: Var[LiteralChakraDirection]
    focus_thumb_on_change: Var[bool]
    is_disabled: Var[bool]
    is_read_only: Var[bool]
    is_reversed: Var[bool]
    min_: Var[int]
    max_: Var[int]
    min_steps_between_thumbs: Var[int]
    name: Var[str]

    def get_event_triggers(self) -> dict[str, Union[Var, Any]]:
        if False:
            for i in range(10):
                print('nop')
        "Get the event triggers that pass the component's value to the handler.\n\n        Returns:\n            A dict mapping the event trigger to the var that is passed to the handler.\n        "
        return {**super().get_event_triggers(), EventTriggers.ON_CHANGE: lambda e0: [e0], EventTriggers.ON_CHANGE_END: lambda e0: [e0], EventTriggers.ON_CHANGE_START: lambda e0: [e0]}

    def get_ref(self):
        if False:
            print('Hello World!')
        'Get the ref of the component.\n\n        Returns:\n            The ref of the component.\n        '
        return None

    def _get_ref_hook(self) -> Optional[str]:
        if False:
            return 10
        'Override the base _get_ref_hook to handle array refs.\n\n        Returns:\n            The overrided hooks.\n        '
        if self.id:
            ref = format.format_array_ref(self.id, None)
            if ref:
                return f'const {ref} = Array.from({{length:2}}, () => useRef(null));'
            return super()._get_ref_hook()

    @classmethod
    def create(cls, *children, **props) -> Component:
        if False:
            i = 10
            return i + 15
        'Create a RangeSlider component.\n\n        If no children are provided, a default RangeSlider will be created.\n\n        Args:\n            *children: The children of the component.\n            **props: The properties of the component.\n\n        Returns:\n            The RangeSlider component.\n        '
        if len(children) == 0:
            _id = props.get('id', None)
            if _id:
                children = [RangeSliderTrack.create(RangeSliderFilledTrack.create()), RangeSliderThumb.create(index=0, id=_id), RangeSliderThumb.create(index=1, id=_id)]
            else:
                children = [RangeSliderTrack.create(RangeSliderFilledTrack.create()), RangeSliderThumb.create(index=0), RangeSliderThumb.create(index=1)]
        return super().create(*children, **props)

class RangeSliderTrack(ChakraComponent):
    """A range slider track."""
    tag = 'RangeSliderTrack'

class RangeSliderFilledTrack(ChakraComponent):
    """A filled range slider track."""
    tag = 'RangeSliderFilledTrack'

class RangeSliderThumb(ChakraComponent):
    """A range slider thumb."""
    tag = 'RangeSliderThumb'
    index: Var[int]

    def _get_ref_hook(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return None

    def get_ref(self):
        if False:
            print('Hello World!')
        'Get an array ref for the range slider thumb.\n\n        Returns:\n            The array ref.\n        '
        if self.id:
            return format.format_array_ref(self.id, self.index)