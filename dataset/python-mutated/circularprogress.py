"""Container to stack elements with spacing."""
from typing import Union
from reflex.components.component import Component
from reflex.components.libs.chakra import ChakraComponent
from reflex.vars import Var

class CircularProgress(ChakraComponent):
    """The CircularProgress component is used to indicate the progress for determinate and indeterminate processes."""
    tag = 'CircularProgress'
    cap_is_round: Var[bool]
    is_indeterminate: Var[bool]
    max_: Var[int]
    min_: Var[int]
    thickness: Var[Union[str, int]]
    track_color: Var[str]
    value: Var[int]
    value_text: Var[str]
    color: Var[str]
    size: Var[str]

    @classmethod
    def create(cls, *children, label=None, **props) -> Component:
        if False:
            i = 10
            return i + 15
        'Create a circular progress component.\n\n        Args:\n            *children: the children of the component.\n            label: A label to add in the circular progress. Defaults to None.\n            **props: the props of the component.\n\n        Returns:\n            The circular progress component.\n        '
        if len(children) == 0:
            children = []
            if label is not None:
                children.append(CircularProgressLabel.create(label))
        return super().create(*children, **props)

class CircularProgressLabel(ChakraComponent):
    """Label of CircularProcess."""
    tag = 'CircularProgressLabel'