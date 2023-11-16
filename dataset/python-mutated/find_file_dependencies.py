import ast
from typing import List, Optional, Tuple
from ._importlib import _resolve_name

class _ExtractModuleReferences(ast.NodeVisitor):
    """
    Extract the list of global variables a block of code will read and write
    """

    @classmethod
    def run(cls, src: str, package: str) -> List[Tuple[str, Optional[str]]]:
        if False:
            while True:
                i = 10
        visitor = cls(package)
        tree = ast.parse(src)
        visitor.visit(tree)
        return list(visitor.references.keys())

    def __init__(self, package):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.package = package
        self.references = {}

    def _absmodule(self, module_name: str, level: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        if level > 0:
            return _resolve_name(module_name, self.package, level)
        return module_name

    def visit_Import(self, node):
        if False:
            for i in range(10):
                print('nop')
        for alias in node.names:
            self.references[alias.name, None] = True

    def visit_ImportFrom(self, node):
        if False:
            return 10
        name = self._absmodule(node.module, 0 if node.level is None else node.level)
        for alias in node.names:
            if alias.name != '*':
                self.references[name, alias.name] = True
            else:
                self.references[name, None] = True

    def _grab_node_int(self, node):
        if False:
            print('Hello World!')
        return node.value

    def _grab_node_str(self, node):
        if False:
            print('Hello World!')
        return node.value

    def visit_Call(self, node):
        if False:
            return 10
        if hasattr(node.func, 'id') and node.func.id == '__import__':
            try:
                name = self._grab_node_str(node.args[0])
                fromlist = []
                level = 0
                if len(node.args) > 3:
                    for v in node.args[3].elts:
                        fromlist.append(self._grab_node_str(v))
                elif hasattr(node, 'keywords'):
                    for keyword in node.keywords:
                        if keyword.arg == 'fromlist':
                            for v in keyword.value.elts:
                                fromlist.append(self._grab_node_str(v))
                if len(node.args) > 4:
                    level = self._grab_node_int(node.args[4])
                elif hasattr(node, 'keywords'):
                    for keyword in node.keywords:
                        if keyword.arg == 'level':
                            level = self._grab_node_int(keyword.value)
                if fromlist == []:
                    self.references[name, None] = True
                    top_name = name.rsplit('.', maxsplit=1)[0]
                    if top_name != name:
                        top_name = self._absmodule(top_name, level)
                        self.references[top_name, None] = True
                else:
                    name = self._absmodule(name, level)
                    for alias in fromlist:
                        if alias != '*':
                            self.references[name, alias] = True
                        else:
                            self.references[name, None] = True
            except Exception as e:
                return
find_files_source_depends_on = _ExtractModuleReferences.run