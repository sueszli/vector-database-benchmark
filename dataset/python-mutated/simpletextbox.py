from __future__ import annotations
from typing import Any, Callable
from gradio.components.base import FormComponent
from gradio.events import Events

class SimpleTextbox(FormComponent):
    """
    Creates a very simple textbox for user to enter string input or display string output.
    Preprocessing: passes textbox value as a {str} into the function.
    Postprocessing: expects a {str} returned from function and sets textbox value to it.
    Examples-format: a {str} representing the textbox input.
    """
    EVENTS = [Events.change, Events.input, Events.submit]

    def __init__(self, value: str | Callable | None='', *, placeholder: str | None=None, label: str | None=None, every: float | None=None, show_label: bool | None=None, scale: int | None=None, min_width: int=160, interactive: bool | None=None, visible: bool=True, rtl: bool=False, elem_id: str | None=None, elem_classes: list[str] | str | None=None, render: bool=True):
        if False:
            while True:
                i = 10
        '\n        Parameters:\n            value: default text to provide in textbox. If callable, the function will be called whenever the app loads to set the initial value of the component.\n            placeholder: placeholder hint to provide behind textbox.\n            label: component name in interface.\n            every: If `value` is a callable, run the function \'every\' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component\'s .load_event attribute.\n            show_label: if True, will display label.\n            scale: relative width compared to adjacent Components in a Row. For example, if Component A has scale=2, and Component B has scale=1, A will be twice as wide as B. Should be an integer.\n            min_width: minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.\n            interactive: if True, will be rendered as an editable textbox; if False, editing will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.\n            visible: If False, component will be hidden.\n            rtl: If True and `type` is "text", sets the direction of the text to right-to-left (cursor appears on the left of the text). Default is False, which renders cursor on the right.\n            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.\n            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.\n            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.\n        '
        self.placeholder = placeholder
        self.rtl = rtl
        super().__init__(label=label, every=every, show_label=show_label, scale=scale, min_width=min_width, interactive=interactive, visible=visible, elem_id=elem_id, elem_classes=elem_classes, value=value, render=render)

    def preprocess(self, x: str | None) -> str | None:
        if False:
            print('Hello World!')
        '\n        Preprocesses input (converts it to a string) before passing it to the function.\n        Parameters:\n            x: text\n        Returns:\n            text\n        '
        return None if x is None else str(x)

    def postprocess(self, y: str | None) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Postproccess the function output y by converting it to a str before passing it to the frontend.\n        Parameters:\n            y: function output to postprocess.\n        Returns:\n            text\n        '
        return None if y is None else str(y)

    def api_info(self) -> dict[str, Any]:
        if False:
            return 10
        return {'type': 'string'}

    def example_inputs(self) -> Any:
        if False:
            while True:
                i = 10
        return 'Hello!!'