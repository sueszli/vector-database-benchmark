from typing import Dict, Optional, List
from posthog.hogql import ast
from posthog.hogql.errors import HogQLException
from posthog.hogql.visitor import CloningVisitor, TraversingVisitor

def replace_placeholders(node: ast.Expr, placeholders: Optional[Dict[str, ast.Expr]]) -> ast.Expr:
    if False:
        return 10
    return ReplacePlaceholders(placeholders).visit(node)

def find_placeholders(node: ast.Expr) -> List[str]:
    if False:
        print('Hello World!')
    finder = FindPlaceholders()
    finder.visit(node)
    return list(finder.found)

class FindPlaceholders(TraversingVisitor):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.found: set[str] = set()

    def visit_placeholder(self, node: ast.Placeholder):
        if False:
            i = 10
            return i + 15
        self.found.add(node.field)

class ReplacePlaceholders(CloningVisitor):

    def __init__(self, placeholders: Optional[Dict[str, ast.Expr]]):
        if False:
            return 10
        super().__init__()
        self.placeholders = placeholders

    def visit_placeholder(self, node):
        if False:
            print('Hello World!')
        if not self.placeholders:
            raise HogQLException(f'Placeholders, such as {{{node.field}}}, are not supported in this context')
        if node.field in self.placeholders and self.placeholders[node.field] is not None:
            new_node = self.placeholders[node.field]
            new_node.start = node.start
            new_node.end = node.end
            return new_node
        raise HogQLException(f'Placeholder {{{node.field}}} is not available in this context. You can use the following: ' + ', '.join((f'{placeholder}' for placeholder in self.placeholders)))