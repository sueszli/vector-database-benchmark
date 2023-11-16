from __future__ import annotations
import pytest
pytest
import nbconvert
import nbformat
from packaging import version
from bokeh.document import Document
from tests.support.util.filesystem import with_temporary_file
import bokeh.application.handlers.notebook as bahn

def with_script_contents(contents, func):
    if False:
        for i in range(10):
            print('nop')

    def with_file_object(f):
        if False:
            return 10
        nbsource = nbformat.writes(contents)
        f.write(nbsource.encode('UTF-8'))
        f.flush()
        func(f.name)
    with_temporary_file(with_file_object)

class Test_NotebookHandler:

    def test_runner_strips_line_magics(self, ipython) -> None:
        if False:
            print('Hello World!')
        doc = Document()
        source = nbformat.v4.new_notebook()
        source.cells.append(nbformat.v4.new_code_cell('%time'))

        def load(filename):
            if False:
                while True:
                    i = 10
            handler = bahn.NotebookHandler(filename=filename)
            handler.modify_document(doc)
            assert handler._runner.failed is False
        with_script_contents(source, load)

    def test_runner_strips_cell_magics(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        doc = Document()
        source = nbformat.v4.new_notebook()
        code = '%%timeit\n1+1'
        source.cells.append(nbformat.v4.new_code_cell(code))

        def load(filename):
            if False:
                i = 10
                return i + 15
            handler = bahn.NotebookHandler(filename=filename)
            handler.modify_document(doc)
            assert handler._runner.failed is False
        with_script_contents(source, load)

    def test_runner_uses_source_from_filename(self) -> None:
        if False:
            return 10
        doc = Document()
        source = nbformat.v4.new_notebook()
        result = {}

        def load(filename):
            if False:
                while True:
                    i = 10
            handler = bahn.NotebookHandler(filename=filename)
            handler.modify_document(doc)
            result['handler'] = handler
            result['filename'] = filename
        with_script_contents(source, load)
        assert result['handler']._runner.path == result['filename']
        if version.parse(nbconvert.__version__) < version.parse('5.4'):
            expected_source = '\n# coding: utf-8\n'
        else:
            expected_source = '#!/usr/bin/env python\n# coding: utf-8\n'
        assert result['handler']._runner.source == expected_source
        assert not doc.roots