"""The ``AppStateWatcher`` enables a Frontend to:

- subscribe to App state changes
- to access and change the App state.

This is particularly useful for the ``PanelFrontend`` but can be used by other frontends too.

"""
from __future__ import annotations
import os
from lightning.app.frontend.panel.app_state_comm import _watch_app_state
from lightning.app.frontend.utils import _get_flow_state
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.imports import _is_param_available, requires
from lightning.app.utilities.state import AppState
_logger = Logger(__name__)
if _is_param_available():
    from param import ClassSelector, Parameterized, edit_constant
else:
    Parameterized = object
    ClassSelector = dict

class AppStateWatcher(Parameterized):
    """The `AppStateWatcher` enables a Frontend to:

    - Subscribe to any App state changes.
    - To access and change the App state from the UI.

    This is particularly useful for the `PanelFrontend , but can be used by
    other frontends too.

    Example
    -------

    .. code-block:: python

        import param

        app = AppStateWatcher()

        app.state.counter = 1


        @param.depends(app.param.state, watch=True)
        def update(state):
            print(f"The counter was updated to {state.counter}")


        app.state.counter += 1

    This would print ``The counter was updated to 2``.

    The ``AppStateWatcher`` is built on top of Param, which is a framework like dataclass, attrs and
    Pydantic which additionally provides powerful and unique features for building reactive apps.

    Please note the ``AppStateWatcher`` is a singleton, i.e., only one instance is instantiated

    """
    state: AppState = ClassSelector(class_=AppState, constant=True, doc='The AppState holds the state of the app reduced to the scope of the Flow')

    def __new__(cls):
        if False:
            return 10
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

    @requires('param')
    def __init__(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, '_initialized'):
            super().__init__(name='singleton')
            self._start_watching()
            self.param.state.allow_None = False
            self._initialized = True
        if not self.state:
            raise Exception('.state has not been set.')
        if not self.state._state:
            raise Exception('.state._state has not been set.')

    def _start_watching(self):
        if False:
            i = 10
            return i + 15
        _watch_app_state(self._update_flow_state)
        self._update_flow_state()

    def _get_flow_state(self) -> AppState:
        if False:
            while True:
                i = 10
        flow = os.environ['LIGHTNING_FLOW_NAME']
        return _get_flow_state(flow)

    def _update_flow_state(self):
        if False:
            print('Hello World!')
        with edit_constant(self):
            self.state = self._get_flow_state()
        _logger.debug('Requested App State.')