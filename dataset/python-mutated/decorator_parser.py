import ast
import logging
from dataclasses import dataclass
from typing import cast, List, Optional, Set, Tuple, Union
from typing_extensions import Final
FunctionDefinition = Union[ast.FunctionDef, ast.AsyncFunctionDef]
LOG: logging.Logger = logging.getLogger(__name__)

@dataclass
class Decorator:
    name: str
    arguments: Final[Optional[Set[str]]] = None
    keywords: Final[Optional[Set[Tuple[Optional[str], str]]]] = None

class DecoratorParser:

    def __init__(self, unparsed_target_decorators: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._unparsed_target_decorators: str = unparsed_target_decorators
        self._target_decorators: Optional[List[Decorator]] = None

    @property
    def target_decorators(self) -> List[Decorator]:
        if False:
            for i in range(10):
                print('nop')
        target_decorators = self._target_decorators
        if target_decorators is None:
            target_decorators = self._parse_target_decorators(self._unparsed_target_decorators)
            self._target_decorators = target_decorators
        return target_decorators

    def function_matches_target_decorators(self, node: FunctionDefinition) -> bool:
        if False:
            for i in range(10):
                print('nop')
        target_decorator: Decorator = self.target_decorators[0]
        for decorator in node.decorator_list:
            node_decorator = self._parse_decorator(cast(Union[ast.Name, ast.Call, ast.Attribute], decorator))
            if target_decorator.name == node_decorator.name and (not target_decorator.arguments or (node_decorator.arguments and target_decorator.arguments.issubset(node_decorator.arguments))) and (not target_decorator.keywords or (node_decorator.keywords and target_decorator.keywords.issubset(node_decorator.keywords))):
                return True
        return False

    def _resolve_decorator_func_name(self, func: ast.expr) -> str:
        if False:
            return 10
        if isinstance(func, ast.Name):
            return func.id
        func = cast(ast.Attribute, func)
        return self._resolve_decorator_func_name(func.value) + '.' + func.attr

    def _parse_decorator(self, decorator: Union[ast.Name, ast.Call, ast.Attribute]) -> Decorator:
        if False:
            i = 10
            return i + 15
        if isinstance(decorator, ast.Name) or isinstance(decorator, ast.Attribute):
            return Decorator(self._resolve_decorator_func_name(decorator), set(), set())
        decorator_name = self._resolve_decorator_func_name(decorator.func)
        decorator_arguments = {argument.s for argument in decorator.args if isinstance(argument, ast.Str)}
        decorator_keywords = {(keyword.arg, cast(ast.Str, keyword.value).s) for keyword in decorator.keywords if isinstance(keyword.value, ast.Str)}
        return Decorator(decorator_name, decorator_arguments, decorator_keywords)

    def _parse_target_decorators(self, target_decorator: str) -> List[Decorator]:
        if False:
            print('Hello World!')
        '\n        Responsible for parsing the target decorator to extract the\n        decorator name, named and unnamed attributes.\n        '
        well_formed_decorator = target_decorator + '\ndef foo(): ...'
        try:
            parsed_ast = ast.parse(well_formed_decorator)
        except SyntaxError as error:
            LOG.error(f"Can't parse `{well_formed_decorator}`.")
            raise error
        function_definition = parsed_ast.body[0]
        if not isinstance(function_definition, ast.FunctionDef):
            return []
        decorator_list = function_definition.decorator_list
        if len(decorator_list) < 1:
            LOG.error('No target decorators were specified.')
            raise Exception('No target decorators were specified.')
        return [self._parse_decorator(cast(Union[ast.Name, ast.Call, ast.Attribute], decorator)) for decorator in decorator_list]