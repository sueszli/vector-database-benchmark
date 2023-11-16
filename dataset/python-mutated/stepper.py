from __future__ import annotations
from typing import Any, Callable, Optional, Union, cast
from .. import context
from ..element import Element
from .mixins.disableable_element import DisableableElement
from .mixins.value_element import ValueElement

class Stepper(ValueElement):

    def __init__(self, *, value: Union[str, Step, None]=None, on_value_change: Optional[Callable[..., Any]]=None, keep_alive: bool=True) -> None:
        if False:
            print('Hello World!')
        "Stepper\n\n        This element represents `Quasar's QStepper <https://quasar.dev/vue-components/stepper#qstepper-api>`_ component.\n        It contains individual steps.\n\n        To avoid issues with dynamic elements when switching steps,\n        this element uses Vue's `keep-alive <https://vuejs.org/guide/built-ins/keep-alive.html>`_ component.\n        If client-side performance is an issue, you can disable this feature.\n\n        :param value: `ui.step` or name of the step to be initially selected (default: `None` meaning the first step)\n        :param on_value_change: callback to be executed when the selected step changes\n        :param keep_alive: whether to use Vue's keep-alive component on the content (default: `True`)\n        "
        super().__init__(tag='q-stepper', value=value, on_value_change=on_value_change)
        self._props['keep-alive'] = keep_alive
        self._classes.append('nicegui-stepper')

    def _value_to_model_value(self, value: Any) -> Any:
        if False:
            return 10
        return value._props['name'] if isinstance(value, Step) else value

    def _handle_value_change(self, value: Any) -> None:
        if False:
            return 10
        super()._handle_value_change(value)
        names = [step._props['name'] for step in self]
        for (i, step) in enumerate(self):
            done = i < names.index(value) if value in names else False
            step.props(f':done={done}')

    def next(self) -> None:
        if False:
            return 10
        'Show the next step.'
        self.run_method('next')

    def previous(self) -> None:
        if False:
            i = 10
            return i + 15
        'Show the previous step.'
        self.run_method('previous')

class Step(DisableableElement):

    def __init__(self, name: str, title: Optional[str]=None, icon: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        "Step\n\n        This element represents `Quasar's QStep <https://quasar.dev/vue-components/stepper#qstep-api>`_ component.\n        It is a child of a `ui.stepper` element.\n\n        :param name: name of the step (will be the value of the `ui.stepper` element)\n        :param title: title of the step (default: `None`, meaning the same as `name`)\n        :param icon: icon of the step (default: `None`)\n        "
        super().__init__(tag='q-step')
        self._props['name'] = name
        self._props['title'] = title if title is not None else name
        self._classes.append('nicegui-step')
        if icon:
            self._props['icon'] = icon
        self.stepper = cast(ValueElement, context.get_slot().parent)
        if self.stepper.value is None:
            self.stepper.value = name

class StepperNavigation(Element):

    def __init__(self, *, wrap: bool=True) -> None:
        if False:
            print('Hello World!')
        "Stepper Navigation\n\n        This element represents `Quasar's QStepperNavigation https://quasar.dev/vue-components/stepper#qsteppernavigation-api>`_ component.\n\n        :param wrap: whether to wrap the content (default: `True`)\n        "
        super().__init__('q-stepper-navigation')
        if wrap:
            self._classes.append('wrap')