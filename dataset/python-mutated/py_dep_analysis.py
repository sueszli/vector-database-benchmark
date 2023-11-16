import argparse
import ast
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple

class DepGraph(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.edges: Dict[str, Dict[str, bool]] = {}
        self.ids: Dict[str, int] = {}
        self.inv_ids: Dict[int, str] = {}

def _run_shell(args: List[str]) -> str:
    if False:
        i = 10
        return i + 15
    return subprocess.check_output(args).decode(sys.stdout.encoding)

def list_rllib_tests(n: int=-1, test: str=None) -> Tuple[str, List[str]]:
    if False:
        return 10
    'List RLlib tests.\n\n    Args:\n        n: return at most n tests. all tests if n = -1.\n        test: only return information about a specific test.\n    '
    tests_res = _run_shell(['bazel', 'query', 'tests(//python/ray/rllib:*)', '--output', 'label'])
    all_tests = []
    tests = [t.strip() for t in tests_res.splitlines() if t.strip()]
    for t in tests:
        if test and t != test:
            continue
        src_out = _run_shell(['bazel', 'query', 'kind("source file", deps({}))'.format(t), '--output', 'label'])
        srcs = [f.strip() for f in src_out.splitlines()]
        srcs = [f for f in srcs if f.startswith('//python') and f.endswith('.py')]
        if srcs:
            all_tests.append((t, srcs))
        if n > 0 and len(all_tests) >= n:
            break
    return all_tests

def _new_dep(graph: DepGraph, src_module: str, dep: str):
    if False:
        while True:
            i = 10
    'Create a new dependency between src_module and dep.'
    if dep not in graph.ids:
        graph.ids[dep] = len(graph.ids)
    src_id = graph.ids[src_module]
    dep_id = graph.ids[dep]
    if src_id not in graph.edges:
        graph.edges[src_id] = {}
    graph.edges[src_id][dep_id] = True

def _new_import(graph: DepGraph, src_module: str, dep_module: str):
    if False:
        while True:
            i = 10
    'Process a new import statement in src_module.'
    if not dep_module.startswith('ray'):
        return
    _new_dep(graph, src_module, dep_module)

def _is_path_module(module: str, name: str, _base_dir: str) -> bool:
    if False:
        return 10
    'Figure out if base.sub is a python module or not.'
    if module == 'ray._raylet':
        return False
    bps = ['python'] + module.split('.')
    path = os.path.join(_base_dir, os.path.join(*bps), name + '.py')
    if os.path.isfile(path):
        return True
    return False

def _new_from_import(graph: DepGraph, src_module: str, dep_module: str, dep_name: str, _base_dir: str):
    if False:
        return 10
    'Process a new "from ... import ..." statement in src_module.'
    if not dep_module or not dep_module.startswith('ray'):
        return
    if _is_path_module(dep_module, dep_name, _base_dir):
        _new_dep(graph, src_module, _full_module_path(dep_module, dep_name))
    else:
        _new_dep(graph, src_module, dep_module)

def _process_file(graph: DepGraph, src_path: str, src_module: str, _base_dir=''):
    if False:
        for i in range(10):
            print('nop')
    'Create dependencies from src_module to all the valid imports in src_path.\n\n    Args:\n        graph: the DepGraph to be added to.\n        src_path: .py file to be processed.\n        src_module: full module path of the source file.\n        _base_dir: use a different base dir than current dir. For unit testing.\n    '
    with open(os.path.join(_base_dir, src_path), 'r') as in_f:
        tree = ast.parse(in_f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _new_import(graph, src_module, alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    _new_from_import(graph, src_module, node.module, alias.name, _base_dir)

def build_dep_graph() -> DepGraph:
    if False:
        for i in range(10):
            print('nop')
    'Build index from py files to their immediate dependees.'
    graph = DepGraph()
    for (root, sub_dirs, files) in os.walk('python', followlinks=True):
        if _should_skip(root):
            continue
        module = _bazel_path_to_module_path(root)
        for f in files:
            if not f.endswith('.py'):
                continue
            full = _full_module_path(module, f)
            if full.startswith('ray.serve.tests.test_config_files.'):
                continue
            if full not in graph.ids:
                graph.ids[full] = len(graph.ids)
            _process_file(graph, os.path.join(root, f), full)
    graph.inv_ids = {v: k for (k, v) in graph.ids.items()}
    return graph

def _full_module_path(module, f) -> str:
    if False:
        while True:
            i = 10
    if f == '__init__.py':
        return module
    fn = re.sub('\\.py$', '', f)
    if not module:
        return fn
    return module + '.' + fn

def _should_skip(d: str) -> bool:
    if False:
        print('Hello World!')
    'Skip directories that should not contain py sources.'
    if d.startswith('python/.eggs/'):
        return True
    if d.startswith('python/.'):
        return True
    if d.startswith('python/build'):
        return True
    if d.startswith('python/ray/cpp'):
        return True
    return False

def _bazel_path_to_module_path(d: str) -> str:
    if False:
        print('Hello World!')
    'Convert a Bazel file path to python module path.\n\n    Example: //python/ray/rllib:xxx/yyy/dd -> ray.rllib.xxx.yyy.dd\n    '
    d = re.sub('^\\/\\/', '', d)
    d = re.sub('^python', '', d)
    d = re.sub('^[\\/:]', '', d)
    return d.replace('/', '.').replace(':', '.')

def _file_path_to_module_path(f: str) -> str:
    if False:
        i = 10
        return i + 15
    'Return the corresponding module path for a .py file.'
    (dir, fn) = os.path.split(f)
    return _full_module_path(_bazel_path_to_module_path(dir), fn)

def _depends(graph: DepGraph, visited: Dict[int, bool], tid: int, qid: int) -> List[int]:
    if False:
        i = 10
        return i + 15
    'Whether there is a dependency path from module tid to module qid.\n\n    Given graph, and without going through visited.\n    '
    if tid not in graph.edges or qid not in graph.edges:
        return []
    if qid in graph.edges[tid]:
        return [tid, qid]
    for c in graph.edges[tid]:
        if c in visited:
            continue
        visited[c] = True
        ds = _depends(graph, visited, c, qid)
        if ds:
            return [tid] + ds
    return []

def test_depends_on_file(graph: DepGraph, test: Tuple[str, Tuple[str]], path: str) -> List[int]:
    if False:
        for i in range(10):
            print('nop')
    'Give dependency graph, check if a test depends on a specific .py file.\n\n    Args:\n        graph: the dependency graph.\n        test: information about a test, in the format of:\n            [test_name, (src files for the test)]\n    '
    query = _file_path_to_module_path(path)
    if query not in graph.ids:
        return []
    (t, srcs) = test
    if t.startswith('//python/ray/rllib:examples/'):
        return []
    for src in srcs:
        if src == 'ray.rllib.tests.run_regression_tests':
            return []
        tid = _file_path_to_module_path(src)
        if tid not in graph.ids:
            continue
        branch = _depends(graph, {}, graph.ids[tid], graph.ids[query])
        if branch:
            return branch
    return []

def _find_circular_dep_impl(graph: DepGraph, id: str, branch: str) -> bool:
    if False:
        print('Hello World!')
    if id not in graph.edges:
        return False
    for c in graph.edges[id]:
        if c in branch:
            branch.append(c)
            return True
        branch.append(c)
        if _find_circular_dep_impl(graph, c, branch):
            return True
        branch.pop()
    return False

def find_circular_dep(graph: DepGraph) -> Dict[str, List[int]]:
    if False:
        while True:
            i = 10
    'Find circular dependencies among a dependency graph.'
    known = {}
    circles = {}
    for (m, id) in graph.ids.items():
        branch = []
        if _find_circular_dep_impl(graph, id, branch):
            if branch[-1] in known:
                continue
            for n in branch:
                known[n] = True
            circles[m] = branch
    return circles
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test-dep', help='test-dep: find dependencies for a specified test. circular-dep: find circular dependencies in the specific codebase.')
    parser.add_argument('--file', type=str, help='Path of a .py source file relative to --base_dir.')
    parser.add_argument('--test', type=str, help='Specific test to check.')
    parser.add_argument('--smoke-test', action='store_true', help='Load only a few tests for testing.')
    args = parser.parse_args()
    print('building dep graph ...')
    graph = build_dep_graph()
    print('done. total {} files, {} of which have dependencies.'.format(len(graph.ids), len(graph.edges)))
    if args.mode == 'circular-dep':
        circles = find_circular_dep(graph)
        print('Found following circular dependencies: \n')
        for (m, b) in circles.items():
            print(m)
            for n in b:
                print('    ', graph.inv_ids[n])
            print()
    if args.mode == 'test-dep':
        assert args.file, 'Must specify --file for the query.'
        tests = list_rllib_tests(5 if args.smoke_test else -1, args.test)
        print('Total # of tests: ', len(tests))
        for t in tests:
            branch = test_depends_on_file(graph, t, args.file)
            if branch:
                print('{} depends on {}'.format(t[0], args.file))
                for n in branch:
                    print('    ', graph.inv_ids[n])
            else:
                print('{} does not depend on {}'.format(t[0], args.file))