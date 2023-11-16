""" Bokeh Application Handler to look for Bokeh server request callbacks
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
from .request_handler import RequestHandler
__all__ = ('ServerRequestHandler',)

class ServerRequestHandler(RequestHandler):
    """ Load a script which contains server request handler callbacks.

    """
    _module: ModuleType

    def __init__(self, *, filename: PathLike, argv: list[str]=[], package: ModuleType | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Keyword Args:\n            filename (str) : path to a module to load request handler callbacks from\n\n            argv (list[str], optional) : a list of string arguments to use as\n                ``sys.argv`` when the callback code is executed. (default: [])\n\n        '
        super().__init__()
        with open(filename, encoding='utf-8') as f:
            source = f.read()
        self._runner = CodeRunner(source, filename, argv, package=package)
        if not self._runner.failed:
            self._module = self._runner.new_module()

            def extract_callbacks() -> None:
                if False:
                    return 10
                contents = self._module.__dict__
                if 'process_request' in contents:
                    self._process_request = contents['process_request']
                _check_callback(self._process_request, ('request',), what='process_request')
            self._runner.run(self._module, extract_callbacks)

    @property
    def error(self) -> str | None:
        if False:
            i = 10
            return i + 15
        ' If the handler fails, may contain a related error message.\n\n        '
        return self._runner.error

    @property
    def error_detail(self) -> str | None:
        if False:
            while True:
                i = 10
        ' If the handler fails, may contain a traceback or other details.\n\n        '
        return self._runner.error_detail

    @property
    def failed(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ' ``True`` if the request handler callbacks failed to execute\n\n        '
        return self._runner.failed

    def url_path(self) -> str | None:
        if False:
            i = 10
            return i + 15
        ' The last path component for the basename of the path to the\n        callback module.\n\n        '
        if self.failed:
            return None
        else:
            return '/' + os.path.splitext(os.path.basename(self._runner.path))[0]