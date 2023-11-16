"""Tag to loop through a list of components."""
from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Callable, List
from reflex.components.tags.tag import Tag
from reflex.vars import BaseVar, Var
if TYPE_CHECKING:
    from reflex.components.component import Component

class IterTag(Tag):
    """An iterator tag."""
    iterable: Var[List]
    render_fn: Callable
    index_var_name: str = 'i'

    def get_index_var(self) -> Var:
        if False:
            print('Hello World!')
        'Get the index var for the tag (with curly braces).\n\n        This is used to reference the index var within the tag.\n\n        Returns:\n            The index var.\n        '
        return BaseVar(_var_name=self.index_var_name, _var_type=int)

    def get_index_var_arg(self) -> Var:
        if False:
            for i in range(10):
                print('nop')
        'Get the index var for the tag (without curly braces).\n\n        This is used to render the index var in the .map() function.\n\n        Returns:\n            The index var.\n        '
        return BaseVar(_var_name=self.index_var_name, _var_type=int, _var_is_local=True)

    def render_component(self, arg: Var) -> Component:
        if False:
            while True:
                i = 10
        'Render the component.\n\n        Args:\n            arg: The argument to pass to the render function.\n\n        Returns:\n            The rendered component.\n        '
        from reflex.components.layout.cond import Cond
        from reflex.components.layout.foreach import Foreach
        from reflex.components.layout.fragment import Fragment
        args = inspect.getfullargspec(self.render_fn).args
        index = self.get_index_var()
        if len(args) == 1:
            component = self.render_fn(arg)
        else:
            assert len(args) == 2
            component = self.render_fn(arg, index)
        if isinstance(component, (Foreach, Cond)):
            component = Fragment.create(component)
        if component.key is None:
            component.key = index
        return component