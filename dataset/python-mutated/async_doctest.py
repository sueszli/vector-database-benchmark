from doctest import DocTest, DocTestRunner
from textwrap import indent
from typing import Any
from sphinx.application import Sphinx
from sphinx.ext.doctest import DocTestBuilder
from sphinx.ext.doctest import setup as doctest_setup
test_template = '\nimport asyncio as __test_template_asyncio\n\nasync def __test_template__main():\n\n    {test}\n\n    globals().update(locals())\n\n__test_template_asyncio.run(__test_template__main())\n'

class TestRunnerWrapper:

    def __init__(self, runner: DocTestRunner):
        if False:
            for i in range(10):
                print('nop')
        self._runner = runner

    def __getattr__(self, name: str) -> Any:
        if False:
            while True:
                i = 10
        return getattr(self._runner, name)

    def run(self, test: DocTest, *args: Any, **kwargs: Any) -> Any:
        if False:
            print('Hello World!')
        for ex in test.examples:
            ex.source = test_template.format(test=indent(ex.source, '    ').strip())
        return self._runner.run(test, *args, **kwargs)

class AsyncDoctestBuilder(DocTestBuilder):

    @property
    def test_runner(self) -> DocTestRunner:
        if False:
            return 10
        return self._test_runner

    @test_runner.setter
    def test_runner(self, value: DocTestRunner) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._test_runner = TestRunnerWrapper(value)

def setup(app: Sphinx) -> None:
    if False:
        return 10
    doctest_setup(app)
    app.add_builder(AsyncDoctestBuilder, override=True)