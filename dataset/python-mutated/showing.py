"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any
from ..models.ui import UIElement
from ..util.browser import NEW_PARAM, get_browser_controller
from .notebook import DEFAULT_JUPYTER_URL, ProxyUrlFunc, run_notebook_hook
from .saving import save
from .state import curstate
if TYPE_CHECKING:
    from typing_extensions import TypeGuard
    from ..application.application import Application
    from ..application.handlers.function import ModifyDoc
    from ..util.browser import BrowserLike, BrowserTarget
    from .notebook import CommsHandle
    from .state import State
__all__ = ('show',)

def show(obj: UIElement | Application | ModifyDoc, browser: str | None=None, new: BrowserTarget='tab', notebook_handle: bool=False, notebook_url: str | ProxyUrlFunc=DEFAULT_JUPYTER_URL, **kwargs: Any) -> CommsHandle | None:
    if False:
        print('Hello World!')
    'Immediately display a Bokeh object or application.\n\n    :func:`show` may be called multiple times in a single Jupyter notebook\n    cell to display multiple objects. The objects are displayed in order.\n\n    Args:\n        obj (UIElement or Application or callable) :\n            A Bokeh object to display.\n\n            Bokeh plots, widgets, layouts (i.e. rows and columns) may be\n            passed to ``show`` in order to display them. If |output_file|\n            has been called, the output will be saved to an HTML file, which is also\n            opened in a new browser window or tab. If |output_notebook|\n            has been called in a Jupyter notebook, the output will be inline\n            in the associated notebook output cell.\n\n            In a Jupyter notebook, a Bokeh application or callable may also\n            be passed. A callable will be turned into an Application using a\n            ``FunctionHandler``. The application will be run and displayed\n            inline in the associated notebook output cell.\n\n        browser (str, optional) :\n            Specify the browser to use to open output files(default: None)\n\n            For file output, the **browser** argument allows for specifying\n            which browser to display in, e.g. "safari", "firefox", "opera",\n            "windows-default". Not all platforms may support this option, see\n            the documentation for the standard library\n            :doc:`webbrowser <python:library/webbrowser>` module for\n            more information\n\n        new (str, optional) :\n            Specify the browser mode to use for output files (default: "tab")\n\n            For file output, opens or raises the browser window showing the\n            current output file.  If **new** is \'tab\', then opens a new tab.\n            If **new** is \'window\', then opens a new window.\n\n        notebook_handle (bool, optional) :\n            Whether to create a notebook interaction handle (default: False)\n\n            For notebook output, toggles whether a handle which can be used\n            with ``push_notebook`` is returned. Note that notebook handles\n            only apply to standalone plots, layouts, etc. They do not apply\n            when showing Applications in the notebook.\n\n        notebook_url (URL, optional) :\n            Location of the Jupyter notebook page (default: "localhost:8888")\n\n            When showing Bokeh applications, the Bokeh server must be\n            explicitly configured to allow connections originating from\n            different URLs. This parameter defaults to the standard notebook\n            host and port. If you are running on a different location, you\n            will need to supply this value for the application to display\n            properly. If no protocol is supplied in the URL, e.g. if it is\n            of the form "localhost:8888", then "http" will be used.\n\n            ``notebook_url`` can also be a function that takes one int for the\n            bound server port.  If the port is provided, the function needs\n            to generate the full public URL to the bokeh server.  If None\n            is passed, the function is to generate the origin URL.\n\n            If the environment variable JUPYTER_BOKEH_EXTERNAL_URL is set\n            to the external URL of a JupyterHub, notebook_url is overridden\n            with a callable which enables Bokeh to traverse the JupyterHub\n            proxy without specifying this parameter.\n\n    Some parameters are only useful when certain output modes are active:\n\n    * The ``browser`` and ``new`` parameters only apply when |output_file|\n      is active.\n\n    * The ``notebook_handle`` parameter only applies when |output_notebook|\n      is active, and non-Application objects are being shown. It is only\n      supported in Jupyter notebook and raises an exception for other notebook\n      types when it is True.\n\n    * The ``notebook_url`` parameter only applies when showing Bokeh\n      Applications in a Jupyter notebook.\n\n    * Any additional keyword arguments are passed to :class:`~bokeh.server.Server` when\n      showing a Bokeh app (added in version 1.1)\n\n    Returns:\n        When in a Jupyter notebook (with |output_notebook| enabled)\n        and ``notebook_handle=True``, returns a handle that can be used by\n        ``push_notebook``, None otherwise.\n\n    '
    state = curstate()
    if isinstance(obj, UIElement):
        return _show_with_state(obj, state, browser, new, notebook_handle=notebook_handle)

    def is_application(obj: Any) -> TypeGuard[Application]:
        if False:
            for i in range(10):
                print('nop')
        return getattr(obj, '_is_a_bokeh_application_class', False)
    if is_application(obj) or callable(obj):
        assert state.notebook_type is not None
        return run_notebook_hook(state.notebook_type, 'app', obj, state, notebook_url, **kwargs)
    raise ValueError(_BAD_SHOW_MSG)
_BAD_SHOW_MSG = 'Invalid object to show. The object to passed to show must be one of:\n\n* a UIElement (e.g. a plot, figure, widget or layout)\n* a Bokeh Application\n* a callable suitable to an application FunctionHandler\n'

def _show_file_with_state(obj: UIElement, state: State, new: BrowserTarget, controller: BrowserLike) -> None:
    if False:
        i = 10
        return i + 15
    '\n\n    '
    filename = save(obj, state=state)
    controller.open('file://' + filename, new=NEW_PARAM[new])

def _show_with_state(obj: UIElement, state: State, browser: str | None, new: BrowserTarget, notebook_handle: bool=False) -> CommsHandle | None:
    if False:
        print('Hello World!')
    '\n\n    '
    controller = get_browser_controller(browser=browser)
    comms_handle = None
    shown = False
    if state.notebook:
        assert state.notebook_type is not None
        comms_handle = run_notebook_hook(state.notebook_type, 'doc', obj, state, notebook_handle)
        shown = True
    if state.file or not shown:
        _show_file_with_state(obj, state, new, controller)
    return comms_handle