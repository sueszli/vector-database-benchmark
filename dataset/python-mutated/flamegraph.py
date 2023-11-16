import queue
from typing import Any, Dict, List, Optional, Tuple
from .functree import FuncTree, FuncTreeNode

class _FlameNode:

    def __init__(self, parent: Optional['_FlameNode'], name: str) -> None:
        if False:
            return 10
        self.name: str = name
        self.value: float = 0
        self.count: int = 0
        self.parent: Optional['_FlameNode'] = parent
        self.children: Dict[str, '_FlameNode'] = {}

    def get_child(self, child: FuncTreeNode) -> None:
        if False:
            return 10
        if child.fullname not in self.children:
            self.children[child.fullname] = _FlameNode(self, child.fullname)
        self.children[child.fullname].value += child.end - child.start
        self.children[child.fullname].count += 1
        for grandchild in child.children:
            self.children[child.fullname].get_child(grandchild)

class _FlameTree:

    def __init__(self, func_tree: FuncTree) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.root: _FlameNode = _FlameNode(None, '__root__')
        self.parse(func_tree)

    def parse(self, func_tree: FuncTree) -> None:
        if False:
            i = 10
            return i + 15
        self.root = _FlameNode(None, '__root__')
        for child in func_tree.root.children:
            self.root.get_child(child)

class FlameGraph:

    def __init__(self, trace_data: Optional[Dict[str, Any]]=None) -> None:
        if False:
            return 10
        self.trees: Dict[str, _FlameTree] = {}
        if trace_data:
            self.parse(trace_data)

    def parse(self, trace_data: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        func_trees: Dict[str, FuncTree] = {}
        for data in trace_data['traceEvents']:
            key = f"p{data['pid']}_t{data['tid']}"
            if key in func_trees:
                tree = func_trees[key]
            else:
                tree = FuncTree(data['pid'], data['tid'])
                func_trees[key] = tree
            if data['ph'] == 'X':
                tree.add_event(data)
        for (key, tree) in func_trees.items():
            self.trees[key] = _FlameTree(tree)

    def dump_to_perfetto(self) -> List[Dict[str, Any]]:
        if False:
            return 10
        '\n        Reformat data to what perfetto likes\n        private _functionProfileDetails?: FunctionProfileDetails[]\n        export interface FunctionProfileDetails {\n          name?: string;\n          flamegraph?: CallsiteInfo[];\n          expandedCallsite?: CallsiteInfo;\n          expandedId?: number;\n        }\n        export interface CallsiteInfo {\n          id: number;\n          parentId: number;\n          depth: number;\n          name?: string;\n          totalSize: number;\n          selfSize: number;\n          mapping: string;\n          merged: boolean;\n          highlighted: boolean;\n        }\n        '
        ret = []
        for (name, tree) in self.trees.items():
            q: queue.Queue[Tuple[_FlameNode, int, int]] = queue.Queue()
            for child in tree.root.children.values():
                q.put((child, -1, 0))
            if q.empty():
                continue
            flamegraph = []
            idx = 0
            while not q.empty():
                (node, parent, depth) = q.get()
                flamegraph.append({'id': idx, 'parentId': parent, 'depth': depth, 'name': node.name, 'totalSize': node.value, 'selfSize': node.value - sum((n.value for n in node.children.values())), 'mapping': f'{node.count}', 'merged': False, 'highlighted': False})
                for n in node.children.values():
                    q.put((n, idx, depth + 1))
                idx += 1
            detail = {'name': name, 'flamegraph': flamegraph}
            ret.append(detail)
        return ret