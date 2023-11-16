""" Bokeh Application Handler to look for Bokeh server lifecycle callbacks
in a specified Python module.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import os
from types import ModuleType
from ...core.types import PathLike
from ...util.callback_manager import _check_callback
from .code_runner import CodeRunner
from .lifecycle import LifecycleHandler
__all__ = ('ServerLifecycleHandler',)

class ServerLifecycleHandler(LifecycleHandler):
    """ Load a script which contains server lifecycle callbacks.

    """

    def __init__(self, *, filename: PathLike, argv: list[str]=[], package: ModuleType | None=None) -> None:
        if False:
            return 10
        '\n\n        Keyword Args:\n            filename (str) : path to a module to load lifecycle callbacks from\n\n            argv (list[str], optional) : a list of string arguments to use as\n                ``sys.argv`` when the callback code is executed. (default: [])\n\n        '
        super().__init__()
        with open(filename, encoding='utf-8') as f:
            source = f.read()
        self._runner = CodeRunner(source, filename, argv, package=package)
        if not self._runner.failed:
            self._module = self._runner.new_module()

            def extract_callbacks() -> None:
                if False:
                    for i in range(10):
                        print('nop')
                contents = self._module.__dict__
                if 'on_server_loaded' in contents:
                    self._on_server_loaded = contents['on_server_loaded']
                if 'on_server_unloaded' in contents:
                    self._on_server_unloaded = contents['on_server_unloaded']
                if 'on_session_created' in contents:
                    self._on_session_created = contents['on_session_created']
                if 'on_session_destroyed' in contents:
                    self._on_session_destroyed = contents['on_session_destroyed']
                _check_callback(self._on_server_loaded, ('server_context',), what='on_server_loaded')
                _check_callback(self._on_server_unloaded, ('server_context',), what='on_server_unloaded')
                _check_callback(self._on_session_created, ('session_context',), what='on_session_created')
                _check_callback(self._on_session_destroyed, ('session_context',), what='on_session_destroyed')
            self._runner.run(self._module, extract_callbacks)

    @property
    def error(self) -> str | None:
        if False:
            print('Hello World!')
        ' If the handler fails, may contain a related error message.\n\n        '
        return self._runner.error

    @property
    def error_detail(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        ' If the handler fails, may contain a traceback or other details.\n\n        '
        return self._runner.error_detail

    @property
    def failed(self) -> bool:
        if False:
            return 10
        ' ``True`` if the lifecycle callbacks failed to execute\n\n        '
        return self._runner.failed

    def url_path(self) -> str | None:
        if False:
            print('Hello World!')
        ' The last path component for the basename of the path to the\n        callback module.\n\n        '
        if self.failed:
            return None
        else:
            return '/' + os.path.splitext(os.path.basename(self._runner.path))[0]