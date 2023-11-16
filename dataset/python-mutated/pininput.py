"""A pin input component."""
from __future__ import annotations
from typing import Any, Optional, Union
from reflex.components.component import Component
from reflex.components.libs.chakra import ChakraComponent, LiteralInputVariant
from reflex.components.tags.tag import Tag
from reflex.constants import EventTriggers
from reflex.utils import format
from reflex.utils.imports import ImportDict, merge_imports
from reflex.vars import Var

class PinInput(ChakraComponent):
    """The component that provides context to all the pin-input fields."""
    tag = 'PinInput'
    value: Var[str]
    auto_focus: Var[bool]
    default_value: Var[str]
    error_border_color: Var[str]
    focus_border_color: Var[str]
    id_: Var[str]
    length: Var[int]
    is_disabled: Var[bool]
    is_invalid: Var[bool]
    manage_focus: Var[bool]
    mask: Var[bool]
    placeholder: Var[str]
    type_: Var[str]
    variant: Var[LiteralInputVariant]
    name: Var[str]

    def _get_imports(self) -> ImportDict:
        if False:
            while True:
                i = 10
        'Include PinInputField explicitly because it may not be a child component at compile time.\n\n        Returns:\n            The merged import dict.\n        '
        return merge_imports(super()._get_imports(), PinInputField().get_imports())

    def get_event_triggers(self) -> dict[str, Union[Var, Any]]:
        if False:
            print('Hello World!')
        "Get the event triggers that pass the component's value to the handler.\n\n        Returns:\n            A dict mapping the event trigger to the var that is passed to the handler.\n        "
        return {**super().get_event_triggers(), EventTriggers.ON_CHANGE: lambda e0: [e0], EventTriggers.ON_COMPLETE: lambda e0: [e0]}

    def get_ref(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        "Override ref handling to handle array refs.\n\n        PinInputFields may be created dynamically, so it's not possible\n        to compute their ref at compile time, so we return a cheating\n        guess if the id is specified.\n\n        The `ref` for this outer component will always be stripped off, so what\n        is returned here only matters for form ref collection purposes.\n\n        Returns:\n            None.\n        "
        if any((isinstance(c, PinInputField) for c in self.children)):
            return None
        if self.id:
            return format.format_array_ref(self.id, idx=self.length)
        return super().get_ref()

    def _get_ref_hook(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        'Override the base _get_ref_hook to handle array refs.\n\n        Returns:\n            The overrided hooks.\n        '
        if self.id:
            ref = format.format_array_ref(self.id, None)
            refs_declaration = Var.range(self.length).foreach(lambda : Var.create_safe('useRef(null)', _var_is_string=False))
            refs_declaration._var_is_local = True
            if ref:
                return f'const {ref} = {refs_declaration}'
            return super()._get_ref_hook()

    def _render(self) -> Tag:
        if False:
            i = 10
            return i + 15
        'Override the base _render to remove the fake get_ref.\n\n        Returns:\n            The rendered component.\n        '
        return super()._render().remove_props('ref')

    @classmethod
    def create(cls, *children, **props) -> Component:
        if False:
            for i in range(10):
                print('nop')
        'Create a pin input component.\n\n        If no children are passed in, the component will create a default pin input\n        based on the length prop.\n\n        Args:\n            *children: The children of the component.\n            **props: The props of the component.\n\n        Returns:\n            The pin input component.\n        '
        if children:
            props.pop('length', None)
        elif 'length' in props:
            field_props = {}
            if 'id' in props:
                field_props['id'] = props['id']
            if 'name' in props:
                field_props['name'] = props['name']
            children = [PinInputField.for_length(props['length'], **field_props)]
        return super().create(*children, **props)

class PinInputField(ChakraComponent):
    """The text field that user types in - must be a direct child of PinInput."""
    tag = 'PinInputField'
    index: Optional[Var[int]] = None
    name: Var[str]

    @classmethod
    def for_length(cls, length: Var | int, **props) -> Var:
        if False:
            i = 10
            return i + 15
        'Create a PinInputField for a PinInput with a given length.\n\n        Args:\n            length: The length of the PinInput.\n            props: The props of each PinInputField (name will become indexed).\n\n        Returns:\n            The PinInputField.\n        '
        name = props.get('name')

        def _create(i):
            if False:
                while True:
                    i = 10
            if name is not None:
                props['name'] = f'{name}-{i}'
            return PinInputField.create(**props, index=i, key=i)
        return Var.range(length).foreach(_create)

    def _get_ref_hook(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return None

    def get_ref(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the array ref for the pin input.\n\n        Returns:\n            The array ref.\n        '
        if self.id:
            return format.format_array_ref(self.id, self.index)