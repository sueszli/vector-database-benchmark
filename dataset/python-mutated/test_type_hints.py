import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, set_cwd
import tempfile
import torch
import doctest
import os
import inspect
from pathlib import Path
try:
    import mypy.api
    HAVE_MYPY = True
except ImportError:
    HAVE_MYPY = False

def get_examples_from_docstring(docstr):
    if False:
        print('Hello World!')
    '\n    Extracts all runnable python code from the examples\n    in docstrings; returns a list of lines.\n    '
    examples = doctest.DocTestParser().get_examples(docstr)
    return [f'    {l}' for e in examples for l in e.source.splitlines()]

def get_all_examples():
    if False:
        i = 10
        return i + 15
    'get_all_examples() -> str\n\n    This function grabs (hopefully all) examples from the torch documentation\n    strings and puts them in one nonsensical module returned as a string.\n    '
    blocklist = {'_np'}
    allexamples = ''
    example_file_lines = ['import torch', 'import torch.nn.functional as F', 'import math', 'import numpy', 'import io', 'import itertools', '', 'def preprocess(inp):', '    # type: (torch.Tensor) -> torch.Tensor', '    return inp']
    for fname in dir(torch):
        fn = getattr(torch, fname)
        docstr = inspect.getdoc(fn)
        if docstr and fname not in blocklist:
            e = get_examples_from_docstring(docstr)
            if e:
                example_file_lines.append(f'\n\ndef example_torch_{fname}():')
                example_file_lines += e
    for fname in dir(torch.Tensor):
        fn = getattr(torch.Tensor, fname)
        docstr = inspect.getdoc(fn)
        if docstr and fname not in blocklist:
            e = get_examples_from_docstring(docstr)
            if e:
                example_file_lines.append(f'\n\ndef example_torch_tensor_{fname}():')
                example_file_lines += e
    return '\n'.join(example_file_lines)

class TestTypeHints(TestCase):

    @unittest.skipIf(not HAVE_MYPY, 'need mypy')
    def test_doc_examples(self):
        if False:
            i = 10
            return i + 15
        '\n        Run documentation examples through mypy.\n        '
        fn = Path(__file__).resolve().parent / 'generated_type_hints_smoketest.py'
        with open(fn, 'w') as f:
            print(get_all_examples(), file=f)
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                os.symlink(os.path.dirname(torch.__file__), os.path.join(tmp_dir, 'torch'), target_is_directory=True)
            except OSError:
                raise unittest.SkipTest('cannot symlink') from None
            repo_rootdir = Path(__file__).resolve().parent.parent
            with set_cwd(str(repo_rootdir)):
                (stdout, stderr, result) = mypy.api.run(['--cache-dir=.mypy_cache/doc', '--no-strict-optional', str(fn)])
            if result != 0:
                self.fail(f'mypy failed:\n{stderr}\n{stdout}')
if __name__ == '__main__':
    run_tests()