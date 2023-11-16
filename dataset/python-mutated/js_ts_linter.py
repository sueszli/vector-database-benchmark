"""Lint checks for Js and Ts files."""
from __future__ import annotations
import collections
import os
import re
import shutil
import subprocess
import esprima
from typing import Dict, Final, List, Tuple, Union
from . import linter_utils
from .. import common
from .. import concurrent_task_utils
MYPY = False
if MYPY:
    from scripts.linters import pre_commit_linter
ParsedExpressionsType = Dict[str, Dict[str, List[esprima.nodes.Node]]]
COMPILED_TYPESCRIPT_TMP_PATH: Final = 'tmpcompiledjs/'
INJECTABLES_TO_IGNORE: Final = ['MockIgnoredService', 'UpgradedServices', 'CanAccessSplashPageGuard']

def _parse_js_or_ts_file(filepath: str, file_content: str, comment: bool=False) -> Union[esprima.nodes.Module, esprima.nodes.Script]:
    if False:
        while True:
            i = 10
    "Runs the correct function to parse the given file's source code.\n\n    With ES2015 and later, a JavaScript program can be either a script or a\n    module. It is a very important distinction, since a parser such as Esprima\n    needs to know the type of the source to be able to analyze its syntax\n    correctly. This is achieved by choosing the parseScript function to parse a\n    script and the parseModule function to parse a module.\n\n    https://esprima.readthedocs.io/en/latest/syntactic-analysis.html#distinguishing-a-script-and-a-module\n\n    Args:\n        filepath: str. Path of the source file.\n        file_content: str. Code to compile.\n        comment: bool. Whether to collect comments while parsing the js or ts\n            files.\n\n    Returns:\n        Union[Script, Module]. Parsed contents produced by esprima.\n    "
    parse_function = esprima.parseScript if filepath.endswith('.js') else esprima.parseModule
    return parse_function(file_content, comment=comment)

def _get_expression_from_node_if_one_exists(parsed_node: esprima.nodes.Node, possible_component_names: List[str]) -> esprima.nodes.Node:
    if False:
        return 10
    "This function first checks whether the parsed node represents\n    the required angular component that needs to be derived by checking if\n    it's in the 'possible_component_names' list. If yes, then it will return\n    the expression part of the node from which the component can be derived.\n    If no, it will return None. It is done by filtering out\n    'AssignmentExpression' (as it represents an assignment) and 'Identifier'\n    (as it represents a static expression).\n\n    Args:\n        parsed_node: Node. Parsed node of the body of a JS file.\n        possible_component_names: list(str). List of angular components to check\n            in a JS file. These include directives, factories, controllers,\n            etc.\n\n    Returns:\n        expression: dict or None. Expression part of the node if the node\n        represents a component else None.\n    "
    if parsed_node.type != 'ExpressionStatement':
        return
    expression = parsed_node.expression
    if expression.type != 'CallExpression':
        return
    if expression.callee.type != 'MemberExpression':
        return
    component = expression.callee.property.name
    if component not in possible_component_names:
        return
    return expression

def compile_all_ts_files() -> None:
    if False:
        i = 10
        return i + 15
    'Compiles all project typescript files into\n    COMPILED_TYPESCRIPT_TMP_PATH. Previously, we only compiled\n    the TS files that were needed, but when a relative import was used, the\n    linter would crash with a FileNotFound exception before being able to\n    run. For more details, please see issue #9458.\n    '
    cmd = './node_modules/typescript/bin/tsc -p %s -outDir %s' % ('./tsconfig-lint.json', COMPILED_TYPESCRIPT_TMP_PATH)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (_, encoded_stderr) = proc.communicate()
    stderr = encoded_stderr.decode('utf-8')
    if stderr:
        raise Exception(stderr)

class JsTsLintChecksManager(linter_utils.BaseLinter):
    """Manages all the Js and Ts linting functions."""

    def __init__(self, js_files: List[str], ts_files: List[str], file_cache: pre_commit_linter.FileCache) -> None:
        if False:
            print('Hello World!')
        'Constructs a JsTsLintChecksManager object.\n\n        Args:\n            js_files: list(str). The list of js filepaths to be linted.\n            ts_files: list(str). The list of ts filepaths to be linted.\n            file_cache: object(FileCache). Provides thread-safe access to cached\n                file content.\n        '
        os.environ['PATH'] = '%s/bin:' % common.NODE_PATH + os.environ['PATH']
        self.js_files = js_files
        self.ts_files = ts_files
        self.file_cache = file_cache
        self.parsed_js_and_ts_files: Dict[str, esprima.nodes.Module] = {}
        self.parsed_expressions_in_files: ParsedExpressionsType = {}

    @property
    def js_filepaths(self) -> List[str]:
        if False:
            return 10
        'Return all js filepaths.'
        return self.js_files

    @property
    def ts_filepaths(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Return all ts filepaths.'
        return self.ts_files

    @property
    def all_filepaths(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Return all filepaths.'
        return self.js_filepaths + self.ts_filepaths

    def _validate_and_parse_js_and_ts_files(self) -> Dict[str, Union[esprima.nodes.Module, esprima.nodes.Script]]:
        if False:
            while True:
                i = 10
        "This function validates JavaScript and Typescript files and\n        returns the parsed contents as a Python dictionary.\n\n        Returns:\n            dict. A dict which has key as filepath and value as contents of js\n            and ts files after validating and parsing the files.\n\n        Raises:\n            Exception. The filepath ends with '.js'.\n        "
        files_to_check = self.all_filepaths
        parsed_js_and_ts_files = {}
        concurrent_task_utils.log('Validating and parsing JS and TS files ...')
        for filepath in files_to_check:
            file_content = self.file_cache.read(filepath)
            try:
                parsed_js_and_ts_files[filepath] = _parse_js_or_ts_file(filepath, file_content, comment=True)
            except Exception:
                if filepath.endswith('.js'):
                    raise
                compiled_js_filepath = self._get_compiled_ts_filepath(filepath)
                file_content = self.file_cache.read(compiled_js_filepath)
                parsed_js_and_ts_files[filepath] = _parse_js_or_ts_file(filepath, file_content)
        return parsed_js_and_ts_files

    def _get_expressions_from_parsed_script(self) -> ParsedExpressionsType:
        if False:
            for i in range(10):
                print('nop')
        'This function returns the expressions in the script parsed using\n        js and ts files.\n\n        Returns:\n            dict. A dict which has key as filepath and value as the expressions\n            in the script parsed using js and ts files.\n        '
        parsed_expressions_in_files: ParsedExpressionsType = collections.defaultdict(dict)
        components_to_check = ['controller', 'directive', 'factory', 'filter']
        for (filepath, parsed_script) in self.parsed_js_and_ts_files.items():
            parsed_expressions_in_files[filepath] = collections.defaultdict(list)
            parsed_nodes = parsed_script.body
            for parsed_node in parsed_nodes:
                for component in components_to_check:
                    expression = _get_expression_from_node_if_one_exists(parsed_node, [component])
                    parsed_expressions_in_files[filepath][component].append(expression)
        return parsed_expressions_in_files

    def _get_compiled_ts_filepath(self, filepath: str) -> str:
        if False:
            print('Hello World!')
        'Returns the path for compiled ts file.\n\n        Args:\n            filepath: str. Filepath of ts file.\n\n        Returns:\n            str. Filepath of compiled ts file.\n        '
        compiled_js_filepath = os.path.join(os.getcwd(), COMPILED_TYPESCRIPT_TMP_PATH, os.path.relpath(filepath).replace('.ts', '.js'))
        return compiled_js_filepath

    def _check_constants_declaration(self) -> concurrent_task_utils.TaskResult:
        if False:
            for i in range(10):
                print('nop')
        'Checks the declaration of constants in the TS files to ensure that\n        the constants are not declared in files other than *.constants.ajs.ts\n        and that the constants are declared only single time. This also checks\n        that the constants are declared in both *.constants.ajs.ts (for\n        AngularJS) and in *.constants.ts (for Angular 8).\n\n        Returns:\n            TaskResult. A TaskResult object representing the result of the lint\n            check.\n        '
        name = 'Constants declaration'
        error_messages = []
        failed = False
        ts_files_to_check = self.ts_filepaths
        constants_to_source_filepaths_dict: Dict[str, str] = {}
        for filepath in ts_files_to_check:
            is_corresponding_angularjs_filepath = False
            if filepath.endswith('.constants.ts'):
                filename_without_extension = filepath[:-3]
                corresponding_angularjs_filepath = filename_without_extension + '.ajs.ts'
                is_corresponding_angularjs_filepath = os.path.isfile(corresponding_angularjs_filepath)
                if is_corresponding_angularjs_filepath:
                    compiled_js_filepath = self._get_compiled_ts_filepath(corresponding_angularjs_filepath)
                    file_content = self.file_cache.read(compiled_js_filepath)
                    parsed_script = _parse_js_or_ts_file(filepath, file_content)
                    parsed_nodes = parsed_script.body
                    angularjs_constants_list = []
                    components_to_check = ['constant']
                    for parsed_node in parsed_nodes:
                        expression = _get_expression_from_node_if_one_exists(parsed_node, components_to_check)
                        if not expression:
                            continue
                        angularjs_constants_name = expression.arguments[0].value
                        angularjs_constants_value = expression.arguments[1]
                        if angularjs_constants_value.property:
                            angularjs_constants_value = angularjs_constants_value.property.name
                        if angularjs_constants_value != angularjs_constants_name:
                            failed = True
                            error_messages.append('%s --> Please ensure that the constant %s is initialized from the value from the corresponding Angular constants file (the *.constants.ts file). Please create one in the Angular constants file if it does not exist there.' % (filepath, angularjs_constants_name))
                        angularjs_constants_list.append(angularjs_constants_name)
            parsed_script = self.parsed_js_and_ts_files[filepath]
            parsed_nodes = parsed_script.body
            components_to_check = ['constant']
            for parsed_node in parsed_nodes:
                expression = _get_expression_from_node_if_one_exists(parsed_node, components_to_check)
                if not expression:
                    continue
                constant_name = expression.arguments[0].raw
                if constant_name in constants_to_source_filepaths_dict:
                    failed = True
                    error_message = '%s --> The constant %s is already declared in %s. Please import the file where the constant is declared or rename the constant.' % (filepath, constant_name, constants_to_source_filepaths_dict[constant_name])
                    error_messages.append(error_message)
                else:
                    constants_to_source_filepaths_dict[constant_name] = filepath
        return concurrent_task_utils.TaskResult(name, failed, error_messages, error_messages)

    def _check_angular_services_index(self) -> concurrent_task_utils.TaskResult:
        if False:
            print('Hello World!')
        'Finds all @Injectable classes and makes sure that they are added to\n            Oppia root and Angular Services Index.\n\n        Returns:\n            TaskResult. TaskResult having all the messages returned by the\n            lint checks.\n        '
        name = 'Angular Services Index file'
        error_messages: List[str] = []
        injectable_pattern = '%s%s' % ("Injectable\\({\\n*\\s*providedIn: 'root'\\n*}\\)\\n", 'export class ([A-Za-z0-9]*)')
        angular_services_index_path = './core/templates/services/angular-services.index.ts'
        angular_services_index = self.file_cache.read(angular_services_index_path)
        error_messages = []
        failed = False
        for file_path in self.ts_files:
            file_content = self.file_cache.read(file_path)
            class_names = re.findall(injectable_pattern, file_content)
            for class_name in class_names:
                if class_name in INJECTABLES_TO_IGNORE:
                    continue
                import_statement_regex = 'import {[\\s*\\w+,]*%s' % class_name
                if not re.search(import_statement_regex, angular_services_index):
                    error_message = 'Please import %s to Angular Services Index file in %sfrom %s' % (class_name, angular_services_index_path, file_path)
                    error_messages.append(error_message)
                    failed = True
                service_name_type_pair_regex = "\\['%s',\\n*\\s*%s\\]" % (class_name, class_name)
                service_name_type_pair = "['%s', %s]" % (class_name, class_name)
                if not re.search(service_name_type_pair_regex, angular_services_index):
                    error_message = 'Please add the pair %s to the angularServices in %s' % (service_name_type_pair, angular_services_index_path)
                    error_messages.append(error_message)
                    failed = True
        return concurrent_task_utils.TaskResult(name, failed, error_messages, error_messages)

    def perform_all_lint_checks(self) -> List[concurrent_task_utils.TaskResult]:
        if False:
            print('Hello World!')
        'Perform all the lint checks and returns the messages returned by all\n        the checks.\n\n        Returns:\n            list(TaskResult). A list of TaskResult objects representing the\n            results of the lint checks.\n        '
        if not self.all_filepaths:
            return [concurrent_task_utils.TaskResult('JS TS lint', False, [], ['There are no JavaScript or Typescript files to lint.'])]
        shutil.rmtree(COMPILED_TYPESCRIPT_TMP_PATH, ignore_errors=True)
        compile_all_ts_files()
        self.parsed_js_and_ts_files = self._validate_and_parse_js_and_ts_files()
        self.parsed_expressions_in_files = self._get_expressions_from_parsed_script()
        linter_stdout = []
        linter_stdout.append(self._check_constants_declaration())
        linter_stdout.append(self._check_angular_services_index())
        shutil.rmtree(COMPILED_TYPESCRIPT_TMP_PATH, ignore_errors=True)
        return linter_stdout

class ThirdPartyJsTsLintChecksManager(linter_utils.BaseLinter):
    """Manages all the third party Python linting functions."""

    def __init__(self, files_to_lint: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a ThirdPartyJsTsLintChecksManager object.\n\n        Args:\n            files_to_lint: list(str). A list of filepaths to lint.\n        '
        super().__init__()
        self.files_to_lint = files_to_lint

    @property
    def all_filepaths(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Return all filepaths.'
        return self.files_to_lint

    @staticmethod
    def _get_trimmed_error_output(eslint_output: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Remove extra bits from eslint messages.\n\n        Args:\n            eslint_output: str. Output returned by the eslint linter.\n\n        Returns:\n            str. A string with the trimmed messages.\n        '
        trimmed_error_messages = []
        eslint_output_lines = eslint_output.split('\n')
        newlines_present = eslint_output_lines[-1] == '' and eslint_output_lines[-2] == ''
        fix_option_present = eslint_output_lines[-3].endswith('`--fix` option.')
        unicode_x_present = eslint_output_lines[-4].startswith('✖')
        if newlines_present and fix_option_present and unicode_x_present:
            eslint_output_lines = eslint_output_lines[:-4]
        for line in eslint_output_lines:
            if re.search('^\\d+:\\d+', line.lstrip()):
                searched_error_string = re.search('error', line)
                assert searched_error_string is not None
                error_string = searched_error_string.group(0)
                error_message = line.replace(error_string, '', 1)
            else:
                error_message = line
            trimmed_error_messages.append(error_message)
        return '\n'.join(trimmed_error_messages) + '\n'

    def _lint_js_and_ts_files(self) -> concurrent_task_utils.TaskResult:
        if False:
            for i in range(10):
                print('nop')
        'Prints a list of lint errors in the given list of JavaScript files.\n\n        Returns:\n            TaskResult. A TaskResult object representing the result of the lint\n            check.\n\n        Raises:\n            Exception. The start.py file not executed.\n        '
        node_path = os.path.join(common.NODE_PATH, 'bin', 'node')
        eslint_path = os.path.join('node_modules', 'eslint', 'bin', 'eslint.js')
        if not os.path.exists(eslint_path):
            raise Exception('ERROR    Please run start.py first to install node-eslint and its dependencies.')
        files_to_lint = self.all_filepaths
        error_messages = []
        full_error_messages = []
        failed = False
        name = 'ESLint'
        eslint_cmd_args = [node_path, eslint_path, '--quiet']
        proc_args = eslint_cmd_args + files_to_lint
        proc = subprocess.Popen(proc_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (encoded_linter_stdout, encoded_linter_stderr) = proc.communicate()
        linter_stdout = encoded_linter_stdout.decode('utf-8')
        linter_stderr = encoded_linter_stderr.decode('utf-8')
        if linter_stderr:
            raise Exception(linter_stderr)
        if linter_stdout:
            failed = True
            full_error_messages.append(linter_stdout)
            error_messages.append(self._get_trimmed_error_output(linter_stdout))
        return concurrent_task_utils.TaskResult(name, failed, error_messages, full_error_messages)

    def perform_all_lint_checks(self) -> List[concurrent_task_utils.TaskResult]:
        if False:
            return 10
        'Perform all the lint checks and returns the messages returned by all\n        the checks.\n\n        Returns:\n            list(TaskResult). A list of TaskResult objects representing the\n            results of the lint checks.\n        '
        if not self.all_filepaths:
            return [concurrent_task_utils.TaskResult('JS TS lint', False, [], ['There are no JavaScript or Typescript files to lint.'])]
        return [self._lint_js_and_ts_files()]

def get_linters(js_filepaths: List[str], ts_filepaths: List[str], file_cache: pre_commit_linter.FileCache) -> Tuple[JsTsLintChecksManager, ThirdPartyJsTsLintChecksManager]:
    if False:
        return 10
    'Creates JsTsLintChecksManager and ThirdPartyJsTsLintChecksManager\n        objects and return them.\n\n    Args:\n        js_filepaths: list(str). A list of js filepaths to lint.\n        ts_filepaths: list(str). A list of ts filepaths to lint.\n        file_cache: object(FileCache). Provides thread-safe access to cached\n            file content.\n\n    Returns:\n        tuple(JsTsLintChecksManager, ThirdPartyJsTsLintChecksManager. A 2-tuple\n        of custom and third_party linter objects.\n    '
    js_ts_file_paths = js_filepaths + ts_filepaths
    custom_linter = JsTsLintChecksManager(js_filepaths, ts_filepaths, file_cache)
    third_party_linter = ThirdPartyJsTsLintChecksManager(js_ts_file_paths)
    return (custom_linter, third_party_linter)