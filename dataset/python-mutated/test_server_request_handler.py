from __future__ import annotations
import pytest
pytest
from bokeh.application.handlers.handler import Handler
from tests.support.util.filesystem import with_file_contents
import bokeh.application.handlers.server_request_handler as basrh
script_adds_handler = "\ndef process_request(request):\n    return {'Custom': 'Test'}\n"

class Test_ServerRequestHandler:

    def test_request_bad_syntax(self) -> None:
        if False:
            print('Hello World!')
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                return 10
            handler = basrh.ServerRequestHandler(filename=filename)
            result['handler'] = handler
        with_file_contents('This is a syntax error', load)
        handler = result['handler']
        assert handler.error is not None
        assert 'Invalid syntax' in handler.error

    def test_request_runtime_error(self) -> None:
        if False:
            return 10
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                for i in range(10):
                    print('nop')
            handler = basrh.ServerRequestHandler(filename=filename)
            result['handler'] = handler
        with_file_contents("raise RuntimeError('nope')", load)
        handler = result['handler']
        assert handler.error is not None
        assert 'nope' in handler.error

    def test_lifecycle_bad_process_request_signature(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                while True:
                    i = 10
            handler = basrh.ServerRequestHandler(filename=filename)
            result['handler'] = handler
        with_file_contents('\ndef process_request(a,b):\n    pass\n', load)
        handler = result['handler']
        assert handler.error is not None
        assert 'process_request must have signature func(request)' in handler.error
        assert 'func(a, b)' in handler.error

    def test_url_path(self) -> None:
        if False:
            print('Hello World!')
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                while True:
                    i = 10
            handler = basrh.ServerRequestHandler(filename=filename)
            result['handler'] = handler
        with_file_contents('def process_request(request): return {}', load)
        handler = result['handler']
        assert handler.error is None
        url_path = handler.url_path()
        assert url_path is not None and url_path.startswith('/')

    async def test_empty_request_handler(self) -> None:
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                return 10
            handler = basrh.ServerRequestHandler(filename=filename)
            result['handler'] = handler
        with_file_contents('# This script does nothing', load)
        handler = result['handler']
        payload = handler.process_request(None)
        if handler.failed:
            raise RuntimeError(handler.error)
        assert payload == {}

    async def test_calling_request_handler(self) -> None:
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                return 10
            handler = result['handler'] = basrh.ServerRequestHandler(filename=filename)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_file_contents(script_adds_handler, load)
        handler = result['handler']
        assert {'Custom': 'Test'} == handler.process_request(None)