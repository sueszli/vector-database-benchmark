"""This script generates topological sort of all the services based on how
services are dependent on each other.
"""
from __future__ import annotations
import collections
import os
from core import utils
import esprima
from typing import Dict, List, Tuple
DIRECTORY_NAMES = ['core/templates', 'extensions']
SERVICE_FILES_SUFFICES = ('.service.ts', 'Service.ts', 'Factory.ts')

def dfs(node: str, topo_sort_stack: List[str], adj_list: Dict[str, List[str]], visit_stack: List[str]) -> None:
    if False:
        while True:
            i = 10
    'Depth First Search starting with node.\n\n    Args:\n        node: str. The service name from which dfs will begin.\n        topo_sort_stack: list(str). Stores topological sort of services\n            in reveresed way.\n        adj_list: dict. Adjacency list of the graph formed with services\n            as nodes and dependencies as edges.\n        visit_stack: list(str). Keeps track of visited and unvisited nodes.\n    '
    visit_stack.append(node)
    for pt in adj_list[node]:
        if pt not in visit_stack:
            dfs(pt, topo_sort_stack, adj_list, visit_stack)
    topo_sort_stack.append(node)

def make_graph() -> Tuple[Dict[str, List[str]], List[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Creates an adjaceny list considering services as node and dependencies\n    as edges.\n\n    Returns:\n        tuple(dict, list(str)). Adjancency list of the graph formed with\n        services as nodes and dependencies as edges, list of all the services.\n    '
    adj_list = collections.defaultdict(list)
    nodes_list = []
    for dirname in DIRECTORY_NAMES:
        for (root, _, filenames) in os.walk(dirname):
            for filename in filenames:
                if filename.endswith(SERVICE_FILES_SUFFICES):
                    nodes_list.append(filename)
                    filepath = os.path.join(root, filename)
                    with utils.open_file(filepath, 'r') as f:
                        file_lines = f.readlines()
                    dep_lines = ''
                    index = 0
                    while index < len(file_lines):
                        line = file_lines[index]
                        if line.startswith('require'):
                            while not line.endswith(';\n'):
                                dep_lines = dep_lines + line
                                index += 1
                                line = file_lines[index]
                            dep_lines = dep_lines + line
                            index += 1
                        elif line.startswith('import'):
                            while not line.endswith(';\n'):
                                index += 1
                                line = file_lines[index]
                                if "'" in line:
                                    break
                            dep_lines = dep_lines + ('require (' + line[line.find("'"):line.rfind("'") + 1] + ');\n')
                            index += 1
                        else:
                            index += 1
                    parsed_script = esprima.parseScript(dep_lines, comment=True)
                    parsed_nodes = parsed_script.body
                    for parsed_node in parsed_nodes:
                        assert parsed_node.type == 'ExpressionStatement'
                        assert parsed_node.expression.callee.name == 'require'
                        arguments = parsed_node.expression.arguments
                        for argument in arguments:
                            dep_path = argument.value
                            if argument.operator == '+':
                                dep_path = argument.left.value + argument.right.value
                            if not dep_path.endswith('.ts'):
                                dep_path = dep_path + '.ts'
                            if dep_path.endswith(SERVICE_FILES_SUFFICES):
                                dep_name = os.path.basename(dep_path)
                                adj_list[dep_name].append(filename)
    return (adj_list, nodes_list)

def main() -> None:
    if False:
        print('Hello World!')
    'Prints the topological order of the services based on the\n    dependencies.\n    '
    (adj_list, nodes_list) = make_graph()
    visit_stack: List[str] = []
    topo_sort_stack: List[str] = []
    nodes_list.sort()
    for unchecked_node in nodes_list:
        if unchecked_node not in visit_stack:
            dfs(unchecked_node, topo_sort_stack, adj_list, visit_stack)
    topo_sort_stack.reverse()
    for service in topo_sort_stack:
        print(service)
if __name__ == '__main__':
    main()