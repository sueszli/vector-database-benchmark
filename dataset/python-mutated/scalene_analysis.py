import ast
import importlib
import os
import sys
from typing import cast, Dict, List, Tuple
if sys.version_info < (3, 9):
    import astunparse
    ast.unparse = astunparse.unparse

class ScaleneAnalysis:

    @staticmethod
    def is_native(package_name: str) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns whether a package is native or not.\n        '
        result = False
        try:
            package = importlib.import_module(package_name)
            package_dir = os.path.dirname(package.__file__)
            for (root, dirs, files) in os.walk(package_dir):
                for filename in files:
                    if filename.endswith('.so') or filename.endswith('.pyd'):
                        return True
            result = False
        except ImportError:
            result = False
        except AttributeError:
            result = True
        except TypeError:
            result = True
        except ModuleNotFoundError:
            result = False
        return result

    @staticmethod
    def get_imported_modules(source: str) -> List[str]:
        if False:
            return 10
        '\n        Extracts a list of imported modules from the given source code.\n\n        Parameters:\n        - source (str): The source code to be analyzed.\n\n        Returns:\n        - imported_modules (list[str]): A list of import statements.\n        '
        source = ScaleneAnalysis.strip_magic_line(source)
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported_modules.append(ast.unparse(node))
        return imported_modules

    @staticmethod
    def get_native_imported_modules(source: str) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Extracts a list of **native** imported modules from the given source code.\n\n        Parameters:\n        - source (str): The source code to be analyzed.\n\n        Returns:\n        - imported_modules (list[str]): A list of import statements.\n        '
        source = ScaleneAnalysis.strip_magic_line(source)
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if ScaleneAnalysis.is_native(alias.name):
                        imported_modules.append(ast.unparse(node))
            elif isinstance(node, ast.ImportFrom):
                node.module = cast(str, node.module)
                if ScaleneAnalysis.is_native(node.module):
                    imported_modules.append(ast.unparse(node))
        return imported_modules

    @staticmethod
    def find_regions(src: str) -> Dict[int, Tuple[int, int]]:
        if False:
            i = 10
            return i + 15
        'This function collects the start and end lines of all loops and functions in the AST, and then uses these to determine the narrowest region containing each line in the source code (that is, loops take precedence over functions.'
        src = ScaleneAnalysis.strip_magic_line(src)
        srclines = src.split('\n')
        tree = ast.parse(src)
        regions = {}
        loops = {}
        functions = {}
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for line in range(node.lineno, node.end_lineno + 1):
                    classes[line] = (node.lineno, node.end_lineno)
            if isinstance(node, (ast.For, ast.While)):
                for line in range(node.lineno, node.end_lineno + 1):
                    loops[line] = (node.lineno, node.end_lineno)
            if isinstance(node, ast.FunctionDef):
                for line in range(node.lineno, node.end_lineno + 1):
                    functions[line] = (node.lineno, node.end_lineno)
        for (lineno, line) in enumerate(srclines, 1):
            if lineno in loops:
                regions[lineno] = loops[lineno]
            elif lineno in functions:
                regions[lineno] = functions[lineno]
            elif lineno in classes:
                regions[lineno] = classes[lineno]
            else:
                regions[lineno] = (lineno, lineno)
        return regions

    @staticmethod
    def find_outermost_loop(src: str) -> Dict[int, Tuple[int, int]]:
        if False:
            return 10
        src = ScaleneAnalysis.strip_magic_line(src)
        srclines = src.split('\n')
        tree = ast.parse(src)
        regions = {}

        def walk(node, current_outermost_region, outer_class):
            if False:
                while True:
                    i = 10
            nonlocal regions
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or (isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.If)) and (outer_class is ast.FunctionDef or outer_class is ast.AsyncFunctionDef or outer_class is ast.ClassDef or (outer_class is None))):
                current_outermost_region = (node.lineno, node.end_lineno)
                outer_class = node.__class__
            for child_node in ast.iter_child_nodes(node):
                walk(child_node, current_outermost_region, outer_class)
            if isinstance(node, ast.stmt):
                outermost_is_loop = outer_class in [ast.For, ast.AsyncFor, ast.While]
                curr_is_block_not_loop = node.__class__ in [ast.With, ast.If, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]
                for line in range(node.lineno, node.end_lineno + 1):
                    if line not in regions:
                        if current_outermost_region and outermost_is_loop:
                            regions[line] = current_outermost_region
                        elif curr_is_block_not_loop and len(srclines[line - 1].strip()) > 0:
                            regions[line] = (node.lineno, node.end_lineno)
                        else:
                            regions[line] = (line, line)
        walk(tree, None, None)
        for (lineno, line) in enumerate(srclines, 1):
            regions[lineno] = regions.get(lineno, (lineno, lineno))
        return regions

    @staticmethod
    def strip_magic_line(source: str) -> str:
        if False:
            return 10
        import re
        srclines = map(lambda x: re.sub('^\\%.*', '', x), source.split('\n'))
        source = '\n'.join(srclines)
        return source