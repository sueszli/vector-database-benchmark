"""Type checker test cases"""
from __future__ import annotations
import os
import re
import sys
from mypy import build
from mypy.build import Graph
from mypy.errors import CompileError
from mypy.modulefinder import BuildSource, FindModuleCache, SearchPaths
from mypy.test.config import test_data_prefix, test_temp_dir
from mypy.test.data import DataDrivenTestCase, DataSuite, FileOperation, module_from_path
from mypy.test.helpers import assert_module_equivalence, assert_string_arrays_equal, assert_target_equivalence, check_test_output_files, find_test_files, normalize_error_messages, parse_options, perform_file_operations
from mypy.test.update_data import update_testcase_output
try:
    import lxml
except ImportError:
    lxml = None
import pytest
typecheck_files = find_test_files(pattern='check-*.test')
if sys.version_info < (3, 9):
    typecheck_files.remove('check-python39.test')
if sys.version_info < (3, 10):
    typecheck_files.remove('check-python310.test')
if sys.version_info < (3, 11):
    typecheck_files.remove('check-python311.test')
if sys.version_info < (3, 12):
    typecheck_files.remove('check-python312.test')
if sys.platform not in ('darwin', 'win32'):
    typecheck_files.remove('check-modules-case.test')

class TypeCheckSuite(DataSuite):
    files = typecheck_files

    def run_case(self, testcase: DataDrivenTestCase) -> None:
        if False:
            for i in range(10):
                print('nop')
        if lxml is None and os.path.basename(testcase.file) == 'check-reports.test':
            pytest.skip('Cannot import lxml. Is it installed?')
        incremental = 'incremental' in testcase.name.lower() or 'incremental' in testcase.file or 'serialize' in testcase.file
        if incremental:
            num_steps = max([2] + list(testcase.output2.keys()))
            for (dn, dirs, files) in os.walk(os.curdir):
                for file in files:
                    m = re.search('\\.([2-9])$', file)
                    if m and int(m.group(1)) > num_steps:
                        raise ValueError('Output file {} exists though test case only has {} runs'.format(file, num_steps))
            steps = testcase.find_steps()
            for step in range(1, num_steps + 1):
                idx = step - 2
                ops = steps[idx] if idx < len(steps) and idx >= 0 else []
                self.run_case_once(testcase, ops, step)
        else:
            self.run_case_once(testcase)

    def _sort_output_if_needed(self, testcase: DataDrivenTestCase, a: list[str]) -> None:
        if False:
            while True:
                i = 10
        idx = testcase.output_inline_start
        if not testcase.files or idx == len(testcase.output):
            return

        def _filename(_msg: str) -> str:
            if False:
                while True:
                    i = 10
            return _msg.partition(':')[0]
        file_weights = {file: idx for (idx, file) in enumerate((_filename(msg) for msg in a))}
        testcase.output[idx:] = sorted(testcase.output[idx:], key=lambda msg: file_weights.get(_filename(msg), -1))

    def run_case_once(self, testcase: DataDrivenTestCase, operations: list[FileOperation]=[], incremental_step: int=0) -> None:
        if False:
            print('Hello World!')
        original_program_text = '\n'.join(testcase.input)
        module_data = self.parse_module(original_program_text, incremental_step)
        for (file, _) in testcase.files:
            module = module_from_path(file)
            if module.endswith('_plugin') and module in sys.modules:
                del sys.modules[module]
        if incremental_step == 0 or incremental_step == 1:
            for (module_name, program_path, program_text) in module_data:
                if module_name == '__main__':
                    with open(program_path, 'w', encoding='utf8') as f:
                        f.write(program_text)
                    break
        elif incremental_step > 1:
            perform_file_operations(operations)
        options = parse_options(original_program_text, testcase, incremental_step)
        options.use_builtins_fixtures = True
        options.show_traceback = True
        if 'columns' in testcase.file:
            options.show_column_numbers = True
        if 'errorcodes' in testcase.file:
            options.hide_error_codes = False
        if 'abstract' not in testcase.file:
            options.allow_empty_bodies = not testcase.name.endswith('_no_empty')
        if 'lowercase' not in testcase.file:
            options.force_uppercase_builtins = True
        if 'union-error' not in testcase.file:
            options.force_union_syntax = True
        if incremental_step and options.incremental:
            options.incremental = True
        else:
            options.incremental = False
            if not testcase.writescache:
                options.cache_dir = os.devnull
        sources = []
        for (module_name, program_path, program_text) in module_data:
            sources.append(BuildSource(program_path, module_name, None if incremental_step else program_text))
        plugin_dir = os.path.join(test_data_prefix, 'plugins')
        sys.path.insert(0, plugin_dir)
        res = None
        try:
            res = build.build(sources=sources, options=options, alt_lib_path=test_temp_dir)
            a = res.errors
        except CompileError as e:
            a = e.messages
        finally:
            assert sys.path[0] == plugin_dir
            del sys.path[0]
        if testcase.normalize_output:
            a = normalize_error_messages(a)
        if incremental_step < 2:
            if incremental_step == 1:
                msg = 'Unexpected type checker output in incremental, run 1 ({}, line {})'
            else:
                assert incremental_step == 0
                msg = 'Unexpected type checker output ({}, line {})'
            self._sort_output_if_needed(testcase, a)
            output = testcase.output
        else:
            msg = f'Unexpected type checker output in incremental, run {incremental_step}' + ' ({}, line {})'
            output = testcase.output2.get(incremental_step, [])
        if output != a and testcase.config.getoption('--update-data', False):
            update_testcase_output(testcase, a, incremental_step=incremental_step)
        assert_string_arrays_equal(output, a, msg.format(testcase.file, testcase.line))
        if res:
            if options.cache_dir != os.devnull:
                self.verify_cache(module_data, res.errors, res.manager, res.graph)
            name = 'targets'
            if incremental_step:
                name += str(incremental_step + 1)
            expected = testcase.expected_fine_grained_targets.get(incremental_step + 1)
            actual = [target for (module, target) in res.manager.processed_targets if module in testcase.test_modules]
            if expected is not None:
                assert_target_equivalence(name, expected, actual)
            if incremental_step > 1:
                suffix = '' if incremental_step == 2 else str(incremental_step - 1)
                expected_rechecked = testcase.expected_rechecked_modules.get(incremental_step - 1)
                if expected_rechecked is not None:
                    assert_module_equivalence('rechecked' + suffix, expected_rechecked, res.manager.rechecked_modules)
                expected_stale = testcase.expected_stale_modules.get(incremental_step - 1)
                if expected_stale is not None:
                    assert_module_equivalence('stale' + suffix, expected_stale, res.manager.stale_modules)
        if testcase.output_files:
            check_test_output_files(testcase, incremental_step, strip_prefix='tmp/')

    def verify_cache(self, module_data: list[tuple[str, str, str]], a: list[str], manager: build.BuildManager, graph: Graph) -> None:
        if False:
            print('Hello World!')
        error_paths = self.find_error_message_paths(a)
        busted_paths = {m.path for (id, m) in manager.modules.items() if graph[id].transitive_error}
        modules = self.find_module_files(manager)
        modules.update({module_name: path for (module_name, path, text) in module_data})
        missing_paths = self.find_missing_cache_files(modules, manager)
        assert error_paths or not busted_paths, 'Some modules reported error despite no errors'
        if not missing_paths == busted_paths:
            raise AssertionError(f'cache data discrepancy {missing_paths} != {busted_paths}')
        assert os.path.isfile(os.path.join(manager.options.cache_dir, '.gitignore'))
        cachedir_tag = os.path.join(manager.options.cache_dir, 'CACHEDIR.TAG')
        assert os.path.isfile(cachedir_tag)
        with open(cachedir_tag) as f:
            assert f.read().startswith('Signature: 8a477f597d28d172789f06886806bc55')

    def find_error_message_paths(self, a: list[str]) -> set[str]:
        if False:
            while True:
                i = 10
        hits = set()
        for line in a:
            m = re.match('([^\\s:]+):(\\d+:)?(\\d+:)? (error|warning|note):', line)
            if m:
                p = m.group(1)
                hits.add(p)
        return hits

    def find_module_files(self, manager: build.BuildManager) -> dict[str, str]:
        if False:
            print('Hello World!')
        return {id: module.path for (id, module) in manager.modules.items()}

    def find_missing_cache_files(self, modules: dict[str, str], manager: build.BuildManager) -> set[str]:
        if False:
            return 10
        ignore_errors = True
        missing = {}
        for (id, path) in modules.items():
            meta = build.find_cache_meta(id, path, manager)
            if not build.validate_meta(meta, id, path, ignore_errors, manager):
                missing[id] = path
        return set(missing.values())

    def parse_module(self, program_text: str, incremental_step: int=0) -> list[tuple[str, str, str]]:
        if False:
            return 10
        "Return the module and program names for a test case.\n\n        Normally, the unit tests will parse the default ('__main__')\n        module and follow all the imports listed there. You can override\n        this behavior and instruct the tests to check multiple modules\n        by using a comment like this in the test case input:\n\n          # cmd: mypy -m foo.bar foo.baz\n\n        You can also use `# cmdN:` to have a different cmd for incremental\n        step N (2, 3, ...).\n\n        Return a list of tuples (module name, file name, program text).\n        "
        m = re.search('# cmd: mypy -m ([a-zA-Z0-9_. ]+)$', program_text, flags=re.MULTILINE)
        if incremental_step > 1:
            alt_regex = f'# cmd{incremental_step}: mypy -m ([a-zA-Z0-9_. ]+)$'
            alt_m = re.search(alt_regex, program_text, flags=re.MULTILINE)
            if alt_m is not None:
                m = alt_m
        if m:
            module_names = m.group(1)
            out = []
            search_paths = SearchPaths((test_temp_dir,), (), (), ())
            cache = FindModuleCache(search_paths, fscache=None, options=None)
            for module_name in module_names.split(' '):
                path = cache.find_module(module_name)
                assert isinstance(path, str), f"Can't find ad hoc case file: {module_name}"
                with open(path, encoding='utf8') as f:
                    program_text = f.read()
                out.append((module_name, path, program_text))
            return out
        else:
            return [('__main__', 'main', program_text)]