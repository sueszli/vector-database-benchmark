"""
Run all doctest examples of the `polars` module using Python's built-in doctest module.

How to check examples: run this script, if exits with code 0, all is good. Otherwise,
the errors will be reported.

How to modify behaviour for doctests:
1. if you would like code to be run and output checked: add the output below the code
   block
2. if you would like code to be run (and thus checked whether it actually not fails),
   but output not be checked: add `# doctest: +IGNORE_RESULT` to the code block. You may
   still add example output.
3. if you would not like code to run: add `#doctest: +SKIP`. You may still add example
   output.

Notes
-----
* Doctest does not have a built-in IGNORE_RESULT directive. We have a number of tests
  where we want to ensure that the code runs, but the output may be random by design, or
  not interesting for us to check. To allow for this behaviour, a custom output checker
  has been created, see below.
* The doctests depend on the exact string representation staying the same. This may not
  be true in the future. For instance, in the past, the printout of DataFrames has
  changed from rounded corners to less rounded corners. To facilitate such a change,
  whilst not immediately having to add IGNORE_RESULT directives everywhere or changing
  all outputs, set `IGNORE_RESULT_ALL=True` below. Do note that this does mean no output
  is being checked anymore.

"""
from __future__ import annotations
import doctest
import importlib
import sys
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Iterator
import polars
if TYPE_CHECKING:
    from types import ModuleType

def doctest_teardown(d: doctest.DocTest) -> None:
    if False:
        print('Hello World!')
    polars.Config.restore_defaults()
    polars.disable_string_cache()

def modules_in_path(p: Path) -> Iterator[ModuleType]:
    if False:
        while True:
            i = 10
    for file in p.rglob('*.py'):
        file_name_import = '.'.join(file.relative_to(p).parts)[:-3]
        temp_module = importlib.import_module(p.name + '.' + file_name_import)
        yield temp_module
if __name__ == '__main__':
    IGNORE_RESULT_ALL = False
    IGNORE_RESULT = doctest.register_optionflag('IGNORE_RESULT')
    warnings.simplefilter('error', DeprecationWarning)
    OutputChecker = doctest.OutputChecker

    class IgnoreResultOutputChecker(OutputChecker):
        """Python doctest output checker with support for IGNORE_RESULT."""

        def check_output(self, want: str, got: str, optionflags: Any) -> bool:
            if False:
                return 10
            'Return True iff the actual output from an example matches the output.'
            if IGNORE_RESULT_ALL:
                return True
            if IGNORE_RESULT & optionflags:
                return True
            else:
                return OutputChecker.check_output(self, want, got, optionflags)
    doctest.OutputChecker = IgnoreResultOutputChecker
    doctest.NORMALIZE_WHITESPACE = True
    doctest.DONT_ACCEPT_TRUE_FOR_1 = True
    src_dir = Path(polars.__file__).parent
    with TemporaryDirectory() as tmpdir:
        tests = [doctest.DocTestSuite(m, extraglobs={'pl': polars, 'dirpath': Path(tmpdir)}, optionflags=1, tearDown=doctest_teardown) for m in modules_in_path(src_dir)]
        test_suite = unittest.TestSuite(tests)
        result = unittest.TextTestRunner().run(test_suite)
        success_flag = (result.testsRun > 0) & (len(result.failures) == 0)
        sys.exit(int(not success_flag))