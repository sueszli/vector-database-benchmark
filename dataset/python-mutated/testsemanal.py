"""Semantic analyzer test cases"""
from __future__ import annotations
import sys
from typing import Dict
from mypy import build
from mypy.defaults import PYTHON3_VERSION
from mypy.errors import CompileError
from mypy.modulefinder import BuildSource
from mypy.nodes import TypeInfo
from mypy.options import Options
from mypy.test.config import test_temp_dir
from mypy.test.data import DataDrivenTestCase, DataSuite
from mypy.test.helpers import assert_string_arrays_equal, find_test_files, normalize_error_messages, parse_options, testfile_pyversion
semanal_files = find_test_files(pattern='semanal-*.test', exclude=['semanal-errors-python310.test', 'semanal-errors.test', 'semanal-typeinfo.test', 'semanal-symtable.test'])
if sys.version_info < (3, 10):
    semanal_files.remove('semanal-python310.test')

def get_semanal_options(program_text: str, testcase: DataDrivenTestCase) -> Options:
    if False:
        i = 10
        return i + 15
    options = parse_options(program_text, testcase, 1)
    options.use_builtins_fixtures = True
    options.semantic_analysis_only = True
    options.show_traceback = True
    options.python_version = PYTHON3_VERSION
    options.force_uppercase_builtins = True
    return options

class SemAnalSuite(DataSuite):
    files = semanal_files
    native_sep = True

    def run_case(self, testcase: DataDrivenTestCase) -> None:
        if False:
            i = 10
            return i + 15
        test_semanal(testcase)

def test_semanal(testcase: DataDrivenTestCase) -> None:
    if False:
        print('Hello World!')
    'Perform a semantic analysis test case.\n\n    The testcase argument contains a description of the test case\n    (inputs and output).\n    '
    try:
        src = '\n'.join(testcase.input)
        options = get_semanal_options(src, testcase)
        options.python_version = testfile_pyversion(testcase.file)
        result = build.build(sources=[BuildSource('main', None, src)], options=options, alt_lib_path=test_temp_dir)
        a = result.errors
        if a:
            raise CompileError(a)
        for module in sorted(result.files.keys()):
            if module in testcase.test_modules:
                a += result.files[module].str_with_options(options).split('\n')
    except CompileError as e:
        a = e.messages
    if testcase.normalize_output:
        a = normalize_error_messages(a)
    assert_string_arrays_equal(testcase.output, a, f'Invalid semantic analyzer output ({testcase.file}, line {testcase.line})')

class SemAnalErrorSuite(DataSuite):
    files = ['semanal-errors.test']
    if sys.version_info >= (3, 10):
        semanal_files.append('semanal-errors-python310.test')

    def run_case(self, testcase: DataDrivenTestCase) -> None:
        if False:
            while True:
                i = 10
        test_semanal_error(testcase)

def test_semanal_error(testcase: DataDrivenTestCase) -> None:
    if False:
        return 10
    'Perform a test case.'
    try:
        src = '\n'.join(testcase.input)
        res = build.build(sources=[BuildSource('main', None, src)], options=get_semanal_options(src, testcase), alt_lib_path=test_temp_dir)
        a = res.errors
    except CompileError as e:
        a = e.messages
    if testcase.normalize_output:
        a = normalize_error_messages(a)
    assert_string_arrays_equal(testcase.output, a, f'Invalid compiler output ({testcase.file}, line {testcase.line})')

class SemAnalSymtableSuite(DataSuite):
    required_out_section = True
    files = ['semanal-symtable.test']

    def run_case(self, testcase: DataDrivenTestCase) -> None:
        if False:
            print('Hello World!')
        'Perform a test case.'
        try:
            src = '\n'.join(testcase.input)
            result = build.build(sources=[BuildSource('main', None, src)], options=get_semanal_options(src, testcase), alt_lib_path=test_temp_dir)
            a = result.errors
            if a:
                raise CompileError(a)
            for module in sorted(result.files.keys()):
                if module in testcase.test_modules:
                    a.append(f'{module}:')
                    for s in str(result.files[module].names).split('\n'):
                        a.append('  ' + s)
        except CompileError as e:
            a = e.messages
        assert_string_arrays_equal(testcase.output, a, f'Invalid semantic analyzer output ({testcase.file}, line {testcase.line})')

class SemAnalTypeInfoSuite(DataSuite):
    required_out_section = True
    files = ['semanal-typeinfo.test']

    def run_case(self, testcase: DataDrivenTestCase) -> None:
        if False:
            print('Hello World!')
        'Perform a test case.'
        try:
            src = '\n'.join(testcase.input)
            result = build.build(sources=[BuildSource('main', None, src)], options=get_semanal_options(src, testcase), alt_lib_path=test_temp_dir)
            a = result.errors
            if a:
                raise CompileError(a)
            typeinfos = TypeInfoMap()
            for (module, file) in result.files.items():
                if module in testcase.test_modules:
                    for n in file.names.values():
                        if isinstance(n.node, TypeInfo):
                            assert n.fullname
                            if any((n.fullname.startswith(m + '.') for m in testcase.test_modules)):
                                typeinfos[n.fullname] = n.node
            a = str(typeinfos).split('\n')
        except CompileError as e:
            a = e.messages
        assert_string_arrays_equal(testcase.output, a, f'Invalid semantic analyzer output ({testcase.file}, line {testcase.line})')

class TypeInfoMap(Dict[str, TypeInfo]):

    def __str__(self) -> str:
        if False:
            return 10
        a: list[str] = ['TypeInfoMap(']
        for (x, y) in sorted(self.items()):
            ti = ('\n' + '  ').join(str(y).split('\n'))
            a.append(f'  {x} : {ti}')
        a[-1] += ')'
        return '\n'.join(a)