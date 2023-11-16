import contextlib
import doctest
import inspect
import io
import itertools
import os
import numpy as np
import pytest
import cudf
pytestmark = pytest.mark.filterwarnings('ignore::FutureWarning')
tests = [cudf, cudf.core.groupby]

def _name_in_all(parent, name):
    if False:
        for i in range(10):
            print('nop')
    return name in getattr(parent, '__all__', [])

def _is_public_name(parent, name):
    if False:
        for i in range(10):
            print('nop')
    return not name.startswith('_')

def _find_doctests_in_obj(obj, finder=None, criteria=None):
    if False:
        i = 10
        return i + 15
    "Find all doctests in an object.\n\n    Parameters\n    ----------\n    obj : module or class\n        The object to search for docstring examples.\n    finder : doctest.DocTestFinder, optional\n        The DocTestFinder object to use. If not provided, a DocTestFinder is\n        constructed.\n    criteria : callable, optional\n        Callable indicating whether to recurse over members of the provided\n        object. If not provided, names not defined in the object's ``__all__``\n        property are ignored.\n\n    Yields\n    ------\n    doctest.DocTest\n        The next doctest found in the object.\n    "
    if finder is None:
        finder = doctest.DocTestFinder()
    if criteria is None:
        criteria = _name_in_all
    for docstring in finder.find(obj):
        if docstring.examples:
            yield docstring
    for (name, member) in inspect.getmembers(obj):
        if not criteria(obj, name):
            continue
        if inspect.ismodule(member):
            yield from _find_doctests_in_obj(member, finder, criteria=_name_in_all)
        if inspect.isclass(member):
            yield from _find_doctests_in_obj(member, finder, criteria=_is_public_name)

class TestDoctests:

    @pytest.fixture(autouse=True)
    def chdir_to_tmp_path(cls, tmp_path):
        if False:
            return 10
        original_directory = os.getcwd()
        os.chdir(tmp_path)
        yield
        os.chdir(original_directory)

    @pytest.mark.parametrize('docstring', itertools.chain(*[_find_doctests_in_obj(mod) for mod in tests]), ids=lambda docstring: docstring.name)
    def test_docstring(self, docstring):
        if False:
            return 10
        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)
        globals = dict(cudf=cudf, np=np)
        docstring.globs = globals
        doctest_stdout = io.StringIO()
        with contextlib.redirect_stdout(doctest_stdout):
            runner.run(docstring)
            results = runner.summarize()
        assert not results.failed, f'{results.failed} of {results.attempted} doctests failed for {docstring.name}:\n{doctest_stdout.getvalue()}'