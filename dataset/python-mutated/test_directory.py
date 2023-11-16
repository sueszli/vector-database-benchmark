from __future__ import annotations
import pytest
pytest
from typing import TYPE_CHECKING
import jinja2
from bokeh.core.templates import FILE
from bokeh.document import Document
from tests.support.util.filesystem import with_directory_contents
if TYPE_CHECKING:
    from bokeh.application.handlers.handler import Handler
import bokeh.application.handlers.directory as bahd
script_adds_two_roots_template = '\nfrom bokeh.io import curdoc\nfrom bokeh.model import Model\nfrom bokeh.core.properties import Int, Instance, Nullable\n\nclass %s(Model):\n    bar = Int(1)\n\nclass %s(Model):\n    foo = Int(2)\n    child = Nullable(Instance(Model))\n\ncurdoc().add_root(%s())\ncurdoc().add_root(%s())\n'
script_has_lifecycle_handlers = '\ndef on_server_loaded(server_context):\n    return "on_server_loaded"\ndef on_server_unloaded(server_context):\n    return "on_server_unloaded"\ndef on_session_created(session_context):\n    return "on_session_created"\ndef on_session_destroyed(session_context):\n    return "on_session_destroyed"\n'
script_has_request_handler = "\ndef process_request(request):\n    return request['headers']\n"
script_has_lifecycle_and_request_handlers = script_has_lifecycle_handlers + script_has_request_handler

def script_adds_two_roots(some_model_name: str, another_model_name: str) -> str:
    if False:
        while True:
            i = 10
    return script_adds_two_roots_template % (another_model_name, some_model_name, another_model_name, some_model_name)

class Test_DirectoryHandler:

    def test_directory_empty_mainpy(self) -> None:
        if False:
            return 10
        doc = Document()

        def load(filename: str):
            if False:
                for i in range(10):
                    print('nop')
            handler = bahd.DirectoryHandler(filename=filename)
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': '# This script does nothing'}, load)
        assert not doc.roots

    def test_directory_initpy(self) -> None:
        if False:
            i = 10
            return i + 15
        doc = Document()
        results = {}

        def load(filename: str):
            if False:
                print('Hello World!')
            handler = bahd.DirectoryHandler(filename=filename)
            handler.modify_document(doc)
            handler.on_server_loaded('server_context')
            results['package'] = handler._package is not None and handler._package_runner is not None and handler._package_runner.ran
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': 'from . import foo\n' + script_adds_two_roots('SomeModelInTestDirectory', 'AnotherModelInTestDirectory'), '__init__.py': '', 'foo.py': ' # this script does nothing'}, load)
        assert len(doc.roots) == 2
        assert results['package'] is True

    def test_directory_mainpy_adds_roots(self) -> None:
        if False:
            i = 10
            return i + 15
        doc = Document()

        def load(filename: str):
            if False:
                print('Hello World!')
            handler = bahd.DirectoryHandler(filename=filename)
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': script_adds_two_roots('SomeModelInTestDirectory', 'AnotherModelInTestDirectory')}, load)
        assert len(doc.roots) == 2

    def test_directory_empty_mainipynb(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        import nbformat
        doc = Document()
        source = nbformat.v4.new_notebook()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                i = 10
                return i + 15
            handler = bahd.DirectoryHandler(filename=filename)
            handler.modify_document(doc)
            result['handler'] = handler
            result['filename'] = filename
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.ipynb': nbformat.writes(source)}, load)
        assert not doc.roots

    def test_directory_mainipynb_adds_roots(self) -> None:
        if False:
            return 10
        import nbformat
        doc = Document()
        source = nbformat.v4.new_notebook()
        code = script_adds_two_roots('SomeModelInNbTestDirectory', 'AnotherModelInNbTestDirectory')
        source.cells.append(nbformat.v4.new_code_cell(code))
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                return 10
            handler = bahd.DirectoryHandler(filename=filename)
            handler.modify_document(doc)
            result['handler'] = handler
            result['filename'] = filename
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.ipynb': nbformat.writes(source)}, load)
        assert len(doc.roots) == 2

    def test_directory_both_mainipynb_and_mainpy(self) -> None:
        if False:
            print('Hello World!')
        doc = Document()

        def load(filename: str):
            if False:
                while True:
                    i = 10
            handler = bahd.DirectoryHandler(filename=filename)
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        import nbformat
        source = nbformat.v4.new_notebook()
        with_directory_contents({'main.py': script_adds_two_roots('SomeModelInTestDirectory', 'AnotherModelInTestDirectory'), 'main.ipynb': nbformat.writes(source)}, load)
        assert len(doc.roots) == 2

    def test_directory_missing_main(self) -> None:
        if False:
            return 10
        doc = Document()

        def load(filename: str):
            if False:
                while True:
                    i = 10
            handler = bahd.DirectoryHandler(filename=filename)
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with pytest.raises(ValueError):
            with_directory_contents({}, load)

    def test_directory_has_theme_file(self) -> None:
        if False:
            while True:
                i = 10
        doc = Document()

        def load(filename: str):
            if False:
                print('Hello World!')
            handler = bahd.DirectoryHandler(filename=filename)
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        custom_theme = '\nattrs:\n    AnotherModelInTestDirectoryTheme:\n        bar: 42\n    SomeModelInTestDirectoryTheme:\n        foo: 14\n'
        with_directory_contents({'main.py': script_adds_two_roots('SomeModelInTestDirectoryTheme', 'AnotherModelInTestDirectoryTheme') + "\n# we're testing that the script can override the theme\nsome = next(m for m in curdoc().roots if isinstance(m, SomeModelInTestDirectoryTheme))\nsome.foo = 57\n            ", 'theme.yaml': custom_theme}, load)
        assert len(doc.roots) == 2
        some_model = next((m for m in doc.roots if m.__class__.__name__ == 'SomeModelInTestDirectoryTheme'))
        another_model = next((m for m in doc.roots if m.__class__.__name__ == 'AnotherModelInTestDirectoryTheme'))
        assert another_model.bar == 42
        assert some_model.foo == 57
        del some_model.foo
        assert some_model.foo == 14
        doc.theme = None
        assert some_model.foo == 2
        assert another_model.bar == 1

    async def test_directory_with_server_lifecycle(self) -> None:
        doc = Document()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                for i in range(10):
                    print('nop')
            handler = bahd.DirectoryHandler(filename=filename)
            result['handler'] = handler
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': script_adds_two_roots('SomeModelInTestDirectoryWithLifecycle', 'AnotherModelInTestDirectoryWithLifecycle'), 'server_lifecycle.py': script_has_lifecycle_handlers}, load)
        assert len(doc.roots) == 2
        handler = result['handler']
        assert 'on_server_loaded' == handler.on_server_loaded(None)
        assert 'on_server_unloaded' == handler.on_server_unloaded(None)
        assert 'on_session_created' == await handler.on_session_created(None)
        assert 'on_session_destroyed' == await handler.on_session_destroyed(None)

    async def test_directory_with_app_hooks(self) -> None:
        doc = Document()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                for i in range(10):
                    print('nop')
            handler = bahd.DirectoryHandler(filename=filename)
            result['handler'] = handler
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': script_adds_two_roots('SomeModelInTestDirectoryWithLifecycle', 'AnotherModelInTestDirectoryWithLifecycle'), 'app_hooks.py': script_has_lifecycle_and_request_handlers}, load)
        assert len(doc.roots) == 2
        handler = result['handler']
        assert 'on_server_loaded' == handler.on_server_loaded(None)
        assert 'on_server_unloaded' == handler.on_server_unloaded(None)
        assert 'on_session_created' == await handler.on_session_created(None)
        assert 'on_session_destroyed' == await handler.on_session_destroyed(None)
        assert dict(foo=10) == handler.process_request(dict(headers=dict(foo=10)))

    async def test_directory_with_lifecycle_and_app_hooks_errors(self) -> None:

        def load(filename: str):
            if False:
                while True:
                    i = 10
            with pytest.raises(ValueError):
                bahd.DirectoryHandler(filename=filename)
        with_directory_contents({'main.py': script_adds_two_roots('SomeModelInTestDirectoryWithLifecycle', 'AnotherModelInTestDirectoryWithLifecycle'), 'app_hooks.py': script_has_lifecycle_handlers, 'server_lifecycle.py': script_has_request_handler}, load)

    async def test_directory_with_request_handler(self) -> None:
        doc = Document()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                while True:
                    i = 10
            handler = bahd.DirectoryHandler(filename=filename)
            result['handler'] = handler
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': script_adds_two_roots('SomeModelInTestDirectoryWithLifecycle', 'AnotherModelInTestDirectoryWithLifecycle'), 'app_hooks.py': script_has_request_handler}, load)
        assert len(doc.roots) == 2
        handler = result['handler']
        assert dict(foo=10) == handler.process_request(dict(headers=dict(foo=10)))

    def test_directory_with_static(self) -> None:
        if False:
            return 10
        doc = Document()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                i = 10
                return i + 15
            handler = bahd.DirectoryHandler(filename=filename)
            result['handler'] = handler
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': '# This script does nothing', 'static/js/foo.js': '# some JS'}, load)
        assert not doc.roots
        handler = result['handler']
        assert handler.static_path() is not None
        assert handler.static_path().endswith('static')

    def test_directory_without_static(self) -> None:
        if False:
            while True:
                i = 10
        doc = Document()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                while True:
                    i = 10
            handler = bahd.DirectoryHandler(filename=filename)
            result['handler'] = handler
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': '# This script does nothing'}, load)
        assert not doc.roots
        handler = result['handler']
        assert handler.static_path() is None

    def test_directory_with_template(self) -> None:
        if False:
            while True:
                i = 10
        doc = Document()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                print('Hello World!')
            handler = bahd.DirectoryHandler(filename=filename)
            result['handler'] = handler
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': '# This script does nothing', 'templates/index.html': '<div>some HTML</div>'}, load)
        assert not doc.roots
        assert isinstance(doc.template, jinja2.Template)

    def test_directory_without_template(self) -> None:
        if False:
            while True:
                i = 10
        doc = Document()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                return 10
            handler = bahd.DirectoryHandler(filename=filename)
            result['handler'] = handler
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
        with_directory_contents({'main.py': '# This script does nothing'}, load)
        assert not doc.roots
        assert doc.template is FILE

    def test_safe_to_fork(self) -> None:
        if False:
            return 10
        doc = Document()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                for i in range(10):
                    print('nop')
            handler = bahd.DirectoryHandler(filename=filename)
            assert handler.safe_to_fork
            result['handler'] = handler
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
            assert not handler.safe_to_fork
        with_directory_contents({'main.py': '# This script does nothing'}, load)

    def test_url_path(self) -> None:
        if False:
            print('Hello World!')
        doc = Document()
        result: dict[str, Handler] = {}

        def load(filename: str):
            if False:
                print('Hello World!')
            handler = bahd.DirectoryHandler(filename=filename)
            assert handler.safe_to_fork
            result['handler'] = handler
            handler.modify_document(doc)
            if handler.failed:
                raise RuntimeError(handler.error)
            assert not handler.safe_to_fork
        with_directory_contents({'main.py': '# This script does nothing'}, load)
        h = result['handler']
        assert h.url_path().startswith('/')
        h._main_handler._runner._failed = True
        assert h.url_path() is None