"""gr.Slider() component."""
from __future__ import annotations
import math
import random
from typing import Any, Callable
from gradio_client.documentation import document, set_documentation_group
from gradio.components.base import FormComponent
from gradio.events import Events
set_documentation_group('component')

@document()
class Slider(FormComponent):
    """
    Creates a slider that ranges from `minimum` to `maximum` with a step size of `step`.
    Preprocessing: passes slider value as a {float} into the function.
    Postprocessing: expects an {int} or {float} returned from function and sets slider value to it as long as it is within range.
    Examples-format: A {float} or {int} representing the slider's value.

    Demos: sentence_builder, slider_release, generate_tone, titanic_survival, interface_random_slider, blocks_random_slider
    Guides: create-your-own-friends-with-a-gan
    """
    EVENTS = [Events.change, Events.input, Events.release]

    def __init__(self, minimum: float=0, maximum: float=100, value: float | Callable | None=None, *, step: float | None=None, label: str | None=None, info: str | None=None, every: float | None=None, show_label: bool | None=None, container: bool=True, scale: int | None=None, min_width: int=160, interactive: bool | None=None, visible: bool=True, elem_id: str | None=None, elem_classes: list[str] | str | None=None, render: bool=True, randomize: bool=False):
        if False:
            while True:
                i = 10
        "\n        Parameters:\n            minimum: minimum value for slider.\n            maximum: maximum value for slider.\n            value: default value. If callable, the function will be called whenever the app loads to set the initial value of the component. Ignored if randomized=True.\n            step: increment between slider values.\n            label: The label for this component. Appears above the component and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component is assigned to.\n            info: additional component description.\n            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.\n            show_label: if True, will display label.\n            container: If True, will place the component in a container - providing some extra padding around the border.\n            scale: relative width compared to adjacent Components in a Row. For example, if Component A has scale=2, and Component B has scale=1, A will be twice as wide as B. Should be an integer.\n            min_width: minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.\n            interactive: if True, slider will be adjustable; if False, adjusting will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.\n            visible: If False, component will be hidden.\n            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.\n            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.\n            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.\n            randomize: If True, the value of the slider when the app loads is taken uniformly at random from the range given by the minimum and maximum.\n        "
        self.minimum = minimum
        self.maximum = maximum
        if step is None:
            difference = maximum - minimum
            power = math.floor(math.log10(difference) - 2)
            self.step = 10 ** power
        else:
            self.step = step
        if randomize:
            value = self.get_random_value
        super().__init__(label=label, info=info, every=every, show_label=show_label, container=container, scale=scale, min_width=min_width, interactive=interactive, visible=visible, elem_id=elem_id, elem_classes=elem_classes, render=render, value=value)

    def api_info(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        return {'type': 'number', 'description': f'numeric value between {self.minimum} and {self.maximum}'}

    def example_inputs(self) -> Any:
        if False:
            while True:
                i = 10
        return self.minimum

    def get_random_value(self):
        if False:
            for i in range(10):
                print('nop')
        n_steps = int((self.maximum - self.minimum) / self.step)
        step = random.randint(0, n_steps)
        value = self.minimum + step * self.step
        n_decimals = max(str(self.step)[::-1].find('.'), 0)
        if n_decimals:
            value = round(value, n_decimals)
        return value

    def postprocess(self, value: float | None) -> float:
        if False:
            return 10
        return self.minimum if value is None else value

    def preprocess(self, payload: float) -> float:
        if False:
            print('Hello World!')
        return payload