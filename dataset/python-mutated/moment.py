"""Moment component for humanized date rendering."""
from typing import Any, Dict, List
from reflex.components.component import Component, NoSSRComponent
from reflex.utils import imports
from reflex.vars import ImportVar, Var

class Moment(NoSSRComponent):
    """The Moment component."""
    tag: str = 'Moment'
    is_default = True
    library: str = 'react-moment'
    lib_dependencies: List[str] = ['moment']
    interval: Var[int]
    format: Var[str]
    trim: Var[bool]
    parse: Var[str]
    from_now: Var[bool]
    from_now_during: Var[int]
    to_now: Var[bool]
    with_title: Var[bool]
    title_format: Var[str]
    diff: Var[str]
    decimal: Var[bool]
    unit: Var[str]
    duration: Var[str]
    date: Var[str]
    duration_from_now: Var[bool]
    unix: Var[bool]
    local: Var[bool]
    tz: Var[str]

    def _get_imports(self) -> imports.ImportDict:
        if False:
            for i in range(10):
                print('nop')
        merged_imports = super()._get_imports()
        if self.tz is not None:
            merged_imports = imports.merge_imports(merged_imports, {'moment-timezone': {ImportVar(tag='')}})
        return merged_imports

    def get_event_triggers(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Get the events triggers signatures for the component.\n\n        Returns:\n            The signatures of the event triggers.\n        '
        return {**super().get_event_triggers(), 'on_change': lambda date: [date]}

    @classmethod
    def create(cls, *children, **props) -> Component:
        if False:
            return 10
        'Create a Moment component.\n\n        Args:\n            *children: The children of the component.\n            **props: The properties of the component.\n\n        Returns:\n            The Moment Component.\n        '
        comp = super().create(*children, **props)
        if 'tz' in props:
            comp.lib_dependencies.append('moment-timezone')
        return comp