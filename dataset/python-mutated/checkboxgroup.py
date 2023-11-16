"""gr.CheckboxGroup() component"""
from __future__ import annotations
from typing import Any, Callable, Literal
from gradio_client.documentation import document, set_documentation_group
from gradio.components.base import FormComponent
from gradio.events import Events
set_documentation_group('component')

@document()
class CheckboxGroup(FormComponent):
    """
    Creates a set of checkboxes of which a subset can be checked.
    Preprocessing: passes the list of checked checkboxes as a {List[str | int | float]} or their indices as a {List[int]} into the function, depending on `type`.
    Postprocessing: expects a {List[str | int | float]}, each element of which becomes a checked checkbox.
    Examples-format: a {List[str | int | float]} representing the values to be checked.
    Demos: sentence_builder, titanic_survival
    """
    EVENTS = [Events.change, Events.input, Events.select]

    def __init__(self, choices: list[str | int | float | tuple[str, str | int | float]] | None=None, *, value: list[str | float | int] | str | float | int | Callable | None=None, type: Literal['value', 'index']='value', label: str | None=None, info: str | None=None, every: float | None=None, show_label: bool | None=None, container: bool=True, scale: int | None=None, min_width: int=160, interactive: bool | None=None, visible: bool=True, elem_id: str | None=None, elem_classes: list[str] | str | None=None, render: bool=True):
        if False:
            return 10
        '\n        Parameters:\n            choices: A list of string or numeric options to select from. An option can also be a tuple of the form (name, value), where name is the displayed name of the checkbox button and value is the value to be passed to the function, or returned by the function.\n            value: Default selected list of options. If a single choice is selected, it can be passed in as a string or numeric type. If callable, the function will be called whenever the app loads to set the initial value of the component.\n            type: Type of value to be returned by component. "value" returns the list of strings of the choices selected, "index" returns the list of indices of the choices selected.\n            label: The label for this component. Appears above the component and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component is assigned to.\n            info: Additional component description.\n            every: If `value` is a callable, run the function \'every\' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component\'s .load_event attribute.\n            show_label: If True, will display label.\n            container: If True, will place the component in a container - providing some extra padding around the border.\n            scale: Relative width compared to adjacent Components in a Row. For example, if Component A has scale=2, and Component B has scale=1, A will be twice as wide as B. Should be an integer.\n            min_width: Minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.\n            interactive: If True, choices in this checkbox group will be checkable; if False, checking will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.\n            visible: If False, component will be hidden.\n            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.\n            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.\n            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.\n        '
        self.choices = [tuple(c) if isinstance(c, (tuple, list)) else (str(c), c) for c in choices] if choices else []
        valid_types = ['value', 'index']
        if type not in valid_types:
            raise ValueError(f'Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}')
        self.type = type
        super().__init__(label=label, info=info, every=every, show_label=show_label, container=container, scale=scale, min_width=min_width, interactive=interactive, visible=visible, elem_id=elem_id, elem_classes=elem_classes, render=render, value=value)

    def example_inputs(self) -> Any:
        if False:
            return 10
        return [self.choices[0][1]] if self.choices else None

    def api_info(self) -> dict[str, Any]:
        if False:
            return 10
        return {'items': {'enum': [c[1] for c in self.choices], 'type': 'string'}, 'title': 'Checkbox Group', 'type': 'array'}

    def preprocess(self, payload: list[str | int | float]) -> list[str | int | float] | list[int | None]:
        if False:
            while True:
                i = 10
        if self.type == 'value':
            return payload
        elif self.type == 'index':
            choice_values = [value for (_, value) in self.choices]
            return [choice_values.index(choice) if choice in choice_values else None for choice in payload]
        else:
            raise ValueError(f"Unknown type: {self.type}. Please choose from: 'value', 'index'.")

    def postprocess(self, value: list[str | int | float] | str | int | float | None) -> list[str | int | float]:
        if False:
            while True:
                i = 10
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return value

    def as_example(self, input_data):
        if False:
            while True:
                i = 10
        if input_data is None:
            return None
        elif not isinstance(input_data, list):
            input_data = [input_data]
        for data in input_data:
            if data not in [c[0] for c in self.choices]:
                raise ValueError(f'Example {data} provided not a valid choice.')
        return [next((c[0] for c in self.choices if c[1] == data), None) for data in input_data]