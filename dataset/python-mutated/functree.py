import bisect
import copy
import re
from typing import Any, Dict, Generator, List, Optional

class FuncTreeNode:
    name_regex = '(.*) \\((.*?):([0-9]+)\\)'

    def __init__(self, event: Optional[Dict[str, Any]]=None) -> None:
        if False:
            return 10
        self.filename: Optional[str] = None
        self.lineno: Optional[int] = None
        self.is_python: Optional[bool] = False
        self.funcname: Optional[str] = None
        self.parent: Optional[FuncTreeNode] = None
        self.children: List[FuncTreeNode] = []
        self.start: float = -2 ** 64
        self.end: float = 2 ** 64
        self.event: Dict[str, Any] = {}
        if event is None:
            self.event = {'name': '__ROOT__'}
            self.fullname = '__ROOT__'
        else:
            self.event = copy.copy(event)
            self.start = self.event['ts']
            self.end = self.event['ts'] + self.event['dur']
            self.fullname = self.event['name']
            m = re.match(self.name_regex, self.fullname)
            if m:
                self.is_python = True
                self.funcname = m.group(1)
                self.filename = m.group(2)
                self.lineno = int(m.group(3))

    def is_ancestor(self, other: 'FuncTreeNode') -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.start < other.start and self.end > other.end

    def is_same(self, other: 'FuncTreeNode') -> bool:
        if False:
            return 10
        return self.fullname == other.fullname and len(self.children) == len(other.children) and all((t[0].is_same(t[1]) for t in zip(self.children, other.children)))

    def adopt(self, other: 'FuncTreeNode') -> None:
        if False:
            print('Hello World!')
        new_children = []
        if self.is_ancestor(other):
            if not self.children:
                start_idx = end_idx = 0
            else:
                if other.start > self.children[-1].start:
                    start_idx = len(self.children)
                elif other.start < self.children[0].start:
                    start_idx = 0
                else:
                    start_array = [n.start for n in self.children]
                    start_idx = bisect.bisect(start_array, other.start)
                if other.end > self.children[-1].end:
                    end_idx = len(self.children)
                else:
                    end_array = [n.end for n in self.children]
                    end_idx = bisect.bisect(end_array, other.end)
            if start_idx == end_idx + 1:
                self.children[end_idx].adopt(other)
            elif start_idx == end_idx:
                other.parent = self
                self.children.insert(start_idx, other)
            elif start_idx < end_idx:

                def change_parent(node):
                    if False:
                        for i in range(10):
                            print('nop')
                    node.parent = other
                new_children = self.children[start_idx:end_idx]
                list(map(change_parent, new_children))
                other.children = new_children
                other.parent = self
                self.children = self.children[:start_idx] + [other] + self.children[end_idx:]
            else:
                raise Exception('This should not be possible')
        elif self.parent is not None:
            self.parent.adopt(other)
        else:
            raise Exception('This should not be possible')

class FuncTree:

    def __init__(self, pid: int=0, tid: int=0) -> None:
        if False:
            return 10
        self.root: FuncTreeNode = FuncTreeNode()
        self.curr: FuncTreeNode = self.root
        self.pid: int = pid
        self.tid: int = tid

    def is_same(self, other: 'FuncTree') -> bool:
        if False:
            while True:
                i = 10
        return self.root.is_same(other.root)

    def add_event(self, event: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        node = FuncTreeNode(event)
        self.curr.adopt(node)
        self.curr = node

    def first_ts(self) -> float:
        if False:
            i = 10
            return i + 15
        return self.root.children[0].event['ts']

    def first_node(self) -> FuncTreeNode:
        if False:
            print('Hello World!')
        return self.root.children[0]

    def node_by_timestamp(self, ts: float) -> FuncTreeNode:
        if False:
            while True:
                i = 10
        starts = [node.start for node in self.root.children]
        idx = bisect.bisect(starts, ts)
        if idx == 0:
            return self.root.children[0]
        else:
            return self.root.children[idx - 1]

    def normalize(self, first_ts: float) -> None:
        if False:
            print('Hello World!')
        for node in self.inorder_traverse():
            node.start -= first_ts
            node.end -= first_ts

    def inorder_traverse(self) -> Generator[FuncTreeNode, None, None]:
        if False:
            for i in range(10):
                print('nop')
        lst = [self.root]
        while lst:
            ret = lst.pop()
            lst.extend(ret.children[::-1])
            yield ret
        return