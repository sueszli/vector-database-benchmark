import logging
from functools import lru_cache, partial
from gi.repository import Gio, GLib
from ulauncher.api.result import Result
from ulauncher.utils.timer import timer
logger = logging.getLogger()

class DeferredResultRenderer:
    """
    Handles asynchronous render for extensions
    """
    LOADING_DELAY = 0.3

    @classmethod
    @lru_cache(maxsize=None)
    def get_instance(cls) -> 'DeferredResultRenderer':
        if False:
            i = 10
            return i + 15
        '\n        Returns singleton instance\n        '
        return cls()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.loading = None
        self.active_event = None
        self.active_controller = None
        self.app = Gio.Application.get_default()

    def get_active_controller(self):
        if False:
            print('Hello World!')
        return self.active_controller

    def handle_event(self, event, controller):
        if False:
            while True:
                i = 10
        '\n        Schedules "Loading..." message\n        '
        icon = controller.get_normalized_icon_path()
        loading_message = Result(name='Loading...', icon=icon)
        self._cancel_loading()
        self.loading = timer(self.LOADING_DELAY, partial(self.app.window.show_results, [loading_message]))
        self.active_event = event
        self.active_controller = controller
        return True

    def handle_response(self, response, controller):
        if False:
            while True:
                i = 10
        '\n        Calls :func:`response.action.run`\n        '
        if self.active_controller != controller or self.active_event != response.get('event'):
            return
        self._cancel_loading()
        if self.app and hasattr(self.app, 'window'):
            GLib.idle_add(self.app.window.handle_event, response.get('action'))

    def on_query_change(self):
        if False:
            while True:
                i = 10
        '\n        Cancel "Loading...", reset active_event and active_controller\n        '
        self._cancel_loading()
        self.active_event = None
        self.active_controller = None

    def _cancel_loading(self):
        if False:
            while True:
                i = 10
        if self.loading:
            self.loading.cancel()
            self.loading = None