import re
import ast
import uuid
from collections import defaultdict
import astor
import pandas as pd
from pandasai.helpers.path import find_project_root
from pandasai.helpers.skills_manager import SkillsManager
from .node_visitors import AssignmentVisitor, CallVisitor
from .save_chart import add_save_chart
from .optional import import_dependency
from ..exceptions import BadImportError
from ..middlewares.base import Middleware
from ..constants import WHITELISTED_BUILTINS, WHITELISTED_LIBRARIES
from ..middlewares.charts import ChartsMiddleware
from typing import Union, List, Optional, Generator, Any
from ..helpers.logger import Logger
from ..schemas.df_config import Config
import logging
import traceback

class CodeExecutionContext:
    _prompt_id: uuid.UUID = None
    _skills_manager: SkillsManager = None

    def __init__(self, prompt_id: uuid.UUID, skills_manager: SkillsManager):
        if False:
            return 10
        '\n        Additional Context for code execution\n        Args:\n            prompt_id (uuid.UUID): prompt unique id\n            skill (List): list[functions] of  skills added\n        '
        self._skills_manager = skills_manager
        self._prompt_id = prompt_id

    @property
    def prompt_id(self):
        if False:
            while True:
                i = 10
        return self._prompt_id

    @property
    def skills_manager(self):
        if False:
            for i in range(10):
                print('nop')
        return self._skills_manager

class CodeManager:
    _dfs: List
    _middlewares: List[Middleware] = [ChartsMiddleware()]
    _config: Union[Config, dict]
    _logger: Logger = None
    _additional_dependencies: List[dict] = []
    _ast_comparatos_map: dict = {ast.Eq: '=', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Is: 'is', ast.IsNot: 'is not', ast.In: 'in', ast.NotIn: 'not in'}
    _current_code_executed: str = None
    _last_code_executed: str = None

    def __init__(self, dfs: List, config: Union[Config, dict], logger: Logger):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            config (Union[Config, dict], optional): Config to be used. Defaults to None.\n            logger (Logger, optional): Logger to be used. Defaults to None.\n        '
        self._dfs = dfs
        self._config = config
        self._logger = logger
        if self._config.middlewares is not None:
            self.add_middlewares(*self._config.middlewares)

    def add_middlewares(self, *middlewares: Optional[Middleware]):
        if False:
            return 10
        '\n        Add middlewares to PandasAI instance.\n\n        Args:\n            *middlewares: Middlewares to be added\n\n        '
        self._middlewares.extend(middlewares)

    def _execute_catching_errors(self, code: str, environment: dict) -> Optional[Exception]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform execution of the code directly.\n        Call `exec()` for the given `code`, catch any non-base exceptions.\n\n        Args:\n            code (str): Python code.\n            environment (dict): Context for the `exec()`.\n\n        Returns:\n            (Optional[Exception]): Any exception raised during execution.\n                `None` if executed without exceptions.\n        '
        try:
            if ' = analyze_data(' not in code:
                code += '\n\nresult = analyze_data(dfs)'
            exec(code, environment)
        except Exception as exc:
            self._logger.log('Error of executing code', level=logging.WARNING)
            self._logger.log(f'{traceback.format_exc()}', level=logging.DEBUG)
            return exc

    def _handle_error(self, exc: Exception, code: str, environment: dict, use_error_correction_framework: bool=True):
        if False:
            while True:
                i = 10
        '\n        Handle error occurred during first executing of code.\n        If `exc` is instance of `NameError`, try to import the name, extend\n        the context and then call `_execute_catching_errors()` again.\n        If OK, returns the code string; if failed, continuing handling.\n        Args:\n            exc (Exception): The caught exception.\n            code (str): Python code.\n            environment (dict): Context for the `exec()`\n        Raises:\n            Exception: Any exception which has been caught during\n                the very first execution of the code.\n        Returns:\n            str: Python code. Either original or new one, given by\n                error correction framework.\n        '
        if not isinstance(exc, NameError):
            return
        name_to_be_imported = None
        if hasattr(exc, 'name'):
            name_to_be_imported = exc.name
        elif exc.args and isinstance(exc.args[0], str):
            name_ptrn = "'([0-9a-zA-Z_]+)'"
            if (search_name_res := re.search(name_ptrn, exc.args[0])):
                name_to_be_imported = search_name_res[1]
        if name_to_be_imported and name_to_be_imported in WHITELISTED_LIBRARIES:
            try:
                package = import_dependency(name_to_be_imported)
                environment[name_to_be_imported] = package
                caught_error = self._execute_catching_errors(code, environment)
                if caught_error is None:
                    return code
            except ModuleNotFoundError:
                self._logger.log(f"Unable to fix `NameError`: package '{name_to_be_imported}' could not be imported.", level=logging.DEBUG)
            except Exception as new_exc:
                exc = new_exc
                self._logger.log(f'Unable to fix `NameError`: an exception was raised: {traceback.format_exc()}', level=logging.DEBUG)
        if not use_error_correction_framework:
            raise exc

    def _required_dfs(self, code: str) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        List the index of the DataFrames that are needed to execute the code. The goal\n        is to avoid to run the connectors if the code does not need them.\n\n        Args:\n            code (str): Python code to execute\n\n        Returns:\n            List[int]: A list of the index of the DataFrames that are needed to execute\n            the code.\n        '
        required_dfs = []
        for (i, df) in enumerate(self._dfs):
            if f'dfs[{i}]' in code:
                required_dfs.append(df)
            else:
                required_dfs.append(None)
        return required_dfs

    def execute_code(self, code: str, context: CodeExecutionContext) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Execute the python code generated by LLMs to answer the question\n        about the input dataframe. Run the code in the current context and return the\n        result.\n\n        Args:\n            code (str): Python code to execute.\n            context (CodeExecutionContext): Code Execution Context\n                    with prompt id and skills.\n\n        Returns:\n            Any: The result of the code execution. The type of the result depends\n                on the generated code.\n\n        '
        self._current_code_executed = code
        for middleware in self._middlewares:
            code = middleware(code)
        if self._config.save_charts:
            code = add_save_chart(code, logger=self._logger, file_name=str(context.prompt_id), save_charts_path_str=self._config.save_charts_path)
        else:
            code = add_save_chart(code, logger=self._logger, file_name='temp_chart', save_charts_path_str=find_project_root())
        context.skills_manager.used_skills = []
        code_to_run = self._clean_code(code, context)
        self.last_code_executed = code_to_run
        self._logger.log(f'\nCode running:\n```\n{code_to_run}\n        ```')
        dfs = self._required_dfs(code_to_run)
        environment: dict = self._get_environment()
        if context.skills_manager.used_skills:
            for skill_func_name in context.skills_manager.used_skills:
                skill = context.skills_manager.get_skill_by_func_name(skill_func_name)
                environment[skill_func_name] = skill
        environment['dfs'] = self._get_samples(dfs)
        caught_error = self._execute_catching_errors(code_to_run, environment)
        if caught_error is not None:
            self._handle_error(caught_error, code_to_run, environment, use_error_correction_framework=self._config.use_error_correction_framework)
        analyze_data = environment.get('analyze_data')
        return analyze_data(self._get_originals(dfs))

    def _get_samples(self, dfs):
        if False:
            i = 10
            return i + 15
        '\n        Get samples from the dfs\n\n        Args:\n            dfs (list): List of dfs\n\n        Returns:\n            list: List of samples\n        '
        samples = []
        for df in dfs:
            if df is not None:
                samples.append(df.head_df)
            else:
                samples.append(None)
        return samples

    def _get_originals(self, dfs):
        if False:
            print('Hello World!')
        '\n        Get original dfs\n\n        Args:\n            dfs (list): List of dfs\n\n        Returns:\n            list: List of dfs\n        '
        original_dfs = []
        for (index, df) in enumerate(dfs):
            if df is None:
                original_dfs.append(None)
                continue
            if df.has_connector:
                extracted_filters = self._extract_filters(self._current_code_executed)
                filters = extracted_filters.get(f'dfs[{index}]', [])
                df.connector.set_additional_filters(filters)
                df.load_connector(temporary=len(filters) > 0)
            original_dfs.append(df.dataframe)
        return original_dfs

    def _get_environment(self) -> dict:
        if False:
            while True:
                i = 10
        '\n        Returns the environment for the code to be executed.\n\n        Returns (dict): A dictionary of environment variables\n        '
        return {'pd': pd, **{lib['alias']: getattr(import_dependency(lib['module']), lib['name']) if hasattr(import_dependency(lib['module']), lib['name']) else import_dependency(lib['module']) for lib in self._additional_dependencies}, '__builtins__': {**{builtin: __builtins__[builtin] for builtin in WHITELISTED_BUILTINS}, '__build_class__': __build_class__, '__name__': '__main__'}}

    def _is_jailbreak(self, node: ast.stmt) -> bool:
        if False:
            while True:
                i = 10
        '\n        Remove jailbreaks from the code to prevent malicious code execution.\n        Args:\n            node (ast.stmt): A code node to be checked.\n        Returns (bool):\n        '
        DANGEROUS_BUILTINS = ['__subclasses__', '__builtins__', '__import__']
        node_str = ast.dump(node)
        return any((builtin in node_str for builtin in DANGEROUS_BUILTINS))

    def _is_unsafe(self, node: ast.stmt) -> bool:
        if False:
            print('Hello World!')
        '\n        Remove unsafe code from the code to prevent malicious code execution.\n\n        Args:\n            node (ast.stmt): A code node to be checked.\n\n        Returns (bool):\n        '
        code = astor.to_source(node)
        return any((method in code for method in ['.to_csv', '.to_excel', '.to_json', '.to_sql', '.to_feather', '.to_hdf', '.to_parquet', '.to_pickle', '.to_gbq', '.to_stata', '.to_records', '.to_latex', '.to_html', '.to_markdown', '.to_clipboard']))

    def _sanitize_analyze_data(self, analyze_data_node: ast.stmt) -> ast.stmt:
        if False:
            i = 10
            return i + 15
        sanitized_analyze_data = [node for node in analyze_data_node.body if not self._is_df_overwrite(node) and (not self._is_jailbreak(node)) and (not self._is_unsafe(node))]
        analyze_data_node.body = sanitized_analyze_data
        return analyze_data_node

    def _clean_code(self, code: str, context: CodeExecutionContext) -> str:
        if False:
            i = 10
            return i + 15
        '\n        A method to clean the code to prevent malicious code execution.\n\n        Args:\n            code(str): A python code.\n\n        Returns:\n            str: A clean code string.\n\n        '
        self._additional_dependencies = []
        tree = ast.parse(code)
        new_body = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._check_imports(node)
                continue
            if isinstance(node, ast.FunctionDef) and node.name == 'analyze_data':
                analyze_data_node = node
                sanitized_analyze_data = self._sanitize_analyze_data(analyze_data_node)
                if len(context.skills_manager.skills) > 0:
                    for node in ast.walk(analyze_data_node):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                            function_name = node.func.id
                            context.skills_manager.add_used_skill(function_name)
                new_body.append(sanitized_analyze_data)
                continue
            new_body.append(node)
        new_tree = ast.Module(body=new_body)
        return astor.to_source(new_tree, pretty_source=lambda x: ''.join(x)).strip()

    def _is_df_overwrite(self, node: ast.stmt) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove df declarations from the code to prevent malicious code execution.\n\n        Args:\n            node (ast.stmt): A code node to be checked.\n\n        Returns (bool):\n\n        '
        return isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and (node.targets[0].id == 'dfs')

    def _check_imports(self, node: Union[ast.Import, ast.ImportFrom]):
        if False:
            print('Hello World!')
        '\n        Add whitelisted imports to _additional_dependencies.\n\n        Args:\n            node (object): ast.Import or ast.ImportFrom\n\n        Raises:\n            BadImportError: If the import is not whitelisted\n\n        '
        module = node.names[0].name if isinstance(node, ast.Import) else node.module
        library = module.split('.')[0]
        if library == 'pandas':
            return
        if library in WHITELISTED_LIBRARIES + self._config.custom_whitelisted_dependencies:
            for alias in node.names:
                self._additional_dependencies.append({'module': module, 'name': alias.name, 'alias': alias.asname or alias.name})
            return
        if library not in WHITELISTED_BUILTINS:
            raise BadImportError(library)

    @staticmethod
    def _get_nearest_func_call(current_lineno: int, calls: list[ast.Call], func_name: str) -> ast.Call:
        if False:
            i = 10
            return i + 15
        "\n        Utility function to get the nearest previous call node.\n\n        Sort call nodes list (copy of the list) by line number.\n        Iterate over the call nodes list. If the call node's function name\n        equals to `func_name`, set `nearest_call` to the node object.\n\n        Args:\n            current_lineno (int): Number of the current processed line.\n            calls (list[ast.Assign]): List of call nodes.\n            func_name (str): Name of the target function.\n\n        Returns:\n            ast.Call: The node of the nearest previous call `<func_name>()`.\n        "
        calls = sorted(calls, key=lambda node: node.lineno)
        nearest_call = None
        for call_node in calls:
            if call_node.lineno > current_lineno:
                return nearest_call
            try:
                if call_node.func.attr == func_name:
                    nearest_call = call_node
            except AttributeError:
                continue

    @staticmethod
    def _tokenize_operand(operand_node: ast.expr) -> Generator[str, None, None]:
        if False:
            i = 10
            return i + 15
        "\n        Utility generator function to get subscript slice contants.\n\n        Args:\n            operand_node (ast.expr):\n                The node to be tokenized.\n        Yields:\n            str: Token string.\n\n        Examples:\n            >>> code = '''\n            ... foo = [1, [2, 3], [[4, 5], [6, 7]]]\n            ... print(foo[2][1][0])\n            ... '''\n            >>> tree = ast.parse(code)\n            >>> res = CodeManager._tokenize_operand(tree.body[1].value.args[0])\n            >>> print(list(res))\n            ['foo', 2, 1, 0]\n        "
        if isinstance(operand_node, ast.Subscript):
            slice_ = operand_node.slice.value
            yield from CodeManager._tokenize_operand(operand_node.value)
            yield slice_
        if isinstance(operand_node, ast.Name):
            yield operand_node.id
        if isinstance(operand_node, ast.Constant):
            yield operand_node.value

    @staticmethod
    def _get_df_id_by_nearest_assignment(current_lineno: int, assignments: list[ast.Assign], target_name: str):
        if False:
            return 10
        '\n        Utility function to get df label by finding the nearest assigment.\n\n        Sort assignment nodes list (copy of the list) by line number.\n        Iterate over the assignment nodes list. If the assignment node\'s value\n        looks like `dfs[<index>]` and target label equals to `target_name`,\n        set `nearest_assignment` to "dfs[<index>]".\n\n        Args:\n            current_lineno (int): Number of the current processed line.\n            assignments (list[ast.Assign]): List of assignment nodes.\n            target_name (str): Name of the target variable. The assignment\n                node is supposed to assign to this name.\n\n        Returns:\n            str: The string representing df label, looks like "dfs[<index>]".\n        '
        nearest_assignment = None
        assignments = sorted(assignments, key=lambda node: node.lineno)
        for assignment in assignments:
            if assignment.lineno > current_lineno:
                return nearest_assignment
            try:
                is_subscript = isinstance(assignment.value, ast.Subscript)
                dfs_on_the_right = assignment.value.value.id == 'dfs'
                assign_to_target = assignment.targets[0].id == target_name
                if is_subscript and dfs_on_the_right and assign_to_target:
                    nearest_assignment = f'dfs[{assignment.value.slice.value}]'
            except AttributeError:
                continue

    def _extract_comparisons(self, tree: ast.Module) -> dict[str, list]:
        if False:
            print('Hello World!')
        '\n        Process nodes from passed tree to extract filters.\n\n        Collects all assignments in the tree.\n        Collects all function calls in the tree.\n        Walk over the tree and handle each comparison node.\n        For each comparison node, defined what `df` is this node related to.\n        Parse constants values from the comparison node.\n        Add to the result dict.\n\n        Args:\n            tree (str): A snippet of code to be parsed.\n\n        Returns:\n            dict: The `defaultdict(list)` instance containing all filters\n                parsed from the passed instructions tree. The dictionary has\n                the following structure:\n                {\n                    "<df_number>": [\n                        ("<left_operand>", "<operator>", "<right_operand>")\n                    ]\n                }\n        '
        comparisons = defaultdict(list)
        current_df = 'dfs[0]'
        visitor = AssignmentVisitor()
        visitor.visit(tree)
        assignments = visitor.assignment_nodes
        call_visitor = CallVisitor()
        call_visitor.visit(tree)
        calls = call_visitor.call_nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                is_call_on_left = isinstance(node.left, ast.Call)
                is_polars = False
                is_calling_col = False
                try:
                    is_polars = node.left.func.value.id in ('pl', 'polars')
                    is_calling_col = node.left.func.attr == 'col'
                except AttributeError:
                    pass
                if is_call_on_left and is_polars and is_calling_col:
                    df_name = self._get_nearest_func_call(node.lineno, calls, 'filter').func.value.id
                    current_df = self._get_df_id_by_nearest_assignment(node.lineno, assignments, df_name)
                    left_str = node.left.args[0].value
                    for (op, right) in zip(node.ops, node.comparators):
                        op_str = self._ast_comparatos_map.get(type(op), 'Unknown')
                        right_str = right.value
                        comparisons[current_df].append((left_str, op_str, right_str))
                elif isinstance(node.left, ast.Subscript):
                    (name, *slices) = self._tokenize_operand(node.left)
                    current_df = self._get_df_id_by_nearest_assignment(node.lineno, assignments, name) or current_df
                    left_str = slices[-1] if slices else name
                    for (op, right) in zip(node.ops, node.comparators):
                        op_str = self._ast_comparatos_map.get(type(op), 'Unknown')
                        (name, *slices) = self._tokenize_operand(right)
                        right_str = slices[-1] if slices else name
                        comparisons[current_df].append((left_str, op_str, right_str))
        return comparisons

    def _extract_filters(self, code) -> dict[str, list]:
        if False:
            while True:
                i = 10
        '\n        Extract filters to be applied to the dataframe from passed code.\n\n        Args:\n            code (str): A snippet of code to be parsed.\n\n        Returns:\n            dict: The dictionary containing all filters parsed from\n                the passed code. The dictionary has the following structure:\n                {\n                    "<df_number>": [\n                        ("<left_operand>", "<operator>", "<right_operand>")\n                    ]\n                }\n\n        Raises:\n            SyntaxError: If the code is unable to be parsed by `ast.parse()`.\n            Exception: If any exception is raised during working with nodes\n                of the code tree.\n        '
        try:
            parsed_tree = ast.parse(code)
        except SyntaxError:
            self._logger.log('Invalid code passed for extracting filters', level=logging.ERROR)
            self._logger.log(f'{traceback.format_exc()}', level=logging.DEBUG)
            raise
        try:
            filters = self._extract_comparisons(parsed_tree)
        except Exception:
            self._logger.log('Unable to extract filters for passed code', level=logging.ERROR)
            self._logger.log(f'{traceback.format_exc()}', level=logging.DEBUG)
            raise
        return filters

    @property
    def middlewares(self):
        if False:
            while True:
                i = 10
        return self._middlewares

    @property
    def last_code_executed(self):
        if False:
            i = 10
            return i + 15
        return self._last_code_executed

    @last_code_executed.setter
    def last_code_executed(self, code: str):
        if False:
            while True:
                i = 10
        self._last_code_executed = code