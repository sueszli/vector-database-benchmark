from __future__ import annotations
import pytest
pytest
from bokeh.document import Document
import bokeh.application.handlers.code as bahc
script_adds_two_roots = '\nfrom bokeh.io import curdoc\nfrom bokeh.model import Model\nfrom bokeh.core.properties import Int, Instance, Nullable\n\nclass AnotherModelInTestScript(Model):\n    bar = Int(1)\n\nclass SomeModelInTestScript(Model):\n    foo = Int(2)\n    child = Nullable(Instance(Model))\n\ncurdoc().add_root(AnotherModelInTestScript())\ncurdoc().add_root(SomeModelInTestScript())\n'

class TestCodeHandler:

    def test_empty_script(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        doc = Document()
        handler = bahc.CodeHandler(source='# This script does nothing', filename='path/to/test_filename')
        handler.modify_document(doc)
        if handler.failed:
            raise RuntimeError(handler.error)
        assert not doc.roots

    def test_script_adds_roots(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        doc = Document()
        handler = bahc.CodeHandler(source=script_adds_two_roots, filename='path/to/test_filename')
        handler.modify_document(doc)
        if handler.failed:
            raise RuntimeError(handler.error)
        assert len(doc.roots) == 2

    def test_script_bad_syntax(self) -> None:
        if False:
            return 10
        doc = Document()
        handler = bahc.CodeHandler(source='This is a syntax error', filename='path/to/test_filename')
        handler.modify_document(doc)
        assert handler.error is not None
        assert 'Invalid syntax' in handler.error

    def test_script_runtime_error(self) -> None:
        if False:
            return 10
        doc = Document()
        handler = bahc.CodeHandler(source="raise RuntimeError('nope')", filename='path/to/test_filename')
        handler.modify_document(doc)
        assert handler.error is not None
        assert 'nope' in handler.error

    def test_script_sys_path(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        doc = Document()
        handler = bahc.CodeHandler(source='import sys; raise RuntimeError("path: \'%s\'" % sys.path[0])', filename='path/to/test_filename')
        handler.modify_document(doc)
        assert handler.error is not None
        assert "path: 'path/to'" in handler.error

    def test_script_argv(self) -> None:
        if False:
            print('Hello World!')
        doc = Document()
        handler = bahc.CodeHandler(source='import sys; raise RuntimeError("argv: %r" % sys.argv)', filename='path/to/test_filename')
        handler.modify_document(doc)
        assert handler.error is not None
        assert "argv: ['test_filename']" in handler.error
        doc = Document()
        handler = bahc.CodeHandler(source='import sys; raise RuntimeError("argv: %r" % sys.argv)', filename='path/to/test_filename', argv=['10', '20', '30'])
        handler.modify_document(doc)
        assert handler.error is not None
        assert "argv: ['test_filename', '10', '20', '30']" in handler.error

    def test_safe_to_fork(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        doc = Document()
        handler = bahc.CodeHandler(source='# This script does nothing', filename='path/to/test_filename')
        assert handler.safe_to_fork
        handler.modify_document(doc)
        if handler.failed:
            raise RuntimeError(handler.error)
        assert not handler.safe_to_fork