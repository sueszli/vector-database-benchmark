"""Handle dynamic routes in static exports via client-side routing.

Works with /utils/client_side_routing.js to handle the redirect and state.

When the user hits a 404 accessing a route, redirect them to the same page,
setting a reactive state var "routeNotFound" to true if the redirect fails.  The
`wait_for_client_redirect` function will render the component only after
routeNotFound becomes true.
"""
from __future__ import annotations
from reflex import constants
from ...vars import Var
from ..component import Component
from ..layout.cond import Cond
route_not_found: Var = Var.create_safe(constants.ROUTE_NOT_FOUND)

class ClientSideRouting(Component):
    """The client-side routing component."""
    library = '/utils/client_side_routing'
    tag = 'useClientSideRouting'

    def _get_hooks(self) -> str:
        if False:
            return 10
        'Get the hooks to render.\n\n        Returns:\n            The useClientSideRouting hook.\n        '
        return f'const {constants.ROUTE_NOT_FOUND} = {self.tag}()'

    def render(self) -> str:
        if False:
            print('Hello World!')
        'Render the component.\n\n        Returns:\n            Empty string, because this component is only used for its hooks.\n        '
        return ''

def wait_for_client_redirect(component) -> Component:
    if False:
        for i in range(10):
            print('nop')
    'Wait for a redirect to occur before rendering a component.\n\n    This prevents the 404 page from flashing while the redirect is happening.\n\n    Args:\n        component: The component to render after the redirect.\n\n    Returns:\n        The conditionally rendered component.\n    '
    return Cond.create(cond=route_not_found, comp1=component, comp2=ClientSideRouting.create())

class Default404Page(Component):
    """The NextJS default 404 page."""
    library = 'next/error'
    tag = 'Error'
    is_default = True
    status_code: Var[int] = 404