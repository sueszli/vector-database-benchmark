"""
Define ``RegionVisitor`` and ``RegionTransformer`` classes to process SCFG
"""
import abc
from collections.abc import Mapping
from typing import TypeVar, Generic
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import BasicBlock, RegionBlock

def _compute_incoming_labels(graph: Mapping[str, BasicBlock]) -> dict[str, set[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a backward mapping from destination blocks to their\n    incoming blocks.\n    '
    jump_table: dict[str, set[str]] = {}
    blk: BasicBlock
    for k in graph:
        jump_table[k] = set()
    for blk in graph.values():
        for dst in blk.jump_targets:
            if dst in jump_table:
                jump_table[dst].add(blk.name)
    return jump_table

def toposort_graph(graph: Mapping[str, BasicBlock]) -> list[list[str]]:
    if False:
        print('Hello World!')
    'Topologically sort the graph returning a list.\n\n    The first element of the list is the source and the last element is the\n    sink, according to the direction of the dataflow.\n    Each element of the list is a list of nodes at the same topological level.\n    '
    incoming_labels = _compute_incoming_labels(graph)
    visited: set[str] = set()
    toposorted: list[list[str]] = []
    while incoming_labels:
        level = []
        for (k, vs) in incoming_labels.items():
            if not vs - visited:
                level.append(k)
        for k in level:
            del incoming_labels[k]
        visited |= set(level)
        toposorted.append(level)
    return toposorted
Tdata = TypeVar('Tdata')

class RegionVisitor(abc.ABC, Generic[Tdata]):
    """A non-mutating pass on a SCFG.

    When each block is visited, their parent must have be visited.
    The abstract ``visit_*`` methods will receive and will return any custom
    data of type Tdata.
    """
    direction = 'forward'
    'The direction in which the graph is processed. Default is set to\n    "forward". Set to "backward" for reverse dataflow direction.\n    '

    @abc.abstractmethod
    def visit_block(self, block: BasicBlock, data: Tdata) -> Tdata:
        if False:
            return 10
        'This is called when a BasicBlock is visited.'
        pass

    @abc.abstractmethod
    def visit_loop(self, region: RegionBlock, data: Tdata) -> Tdata:
        if False:
            for i in range(10):
                print('nop')
        'This is called when a loop region is visited.\n\n        When overriding this method, remember to handle the merging path of\n        ``data`` for the backedge back to the head of the loop.\n        '
        pass

    @abc.abstractmethod
    def visit_switch(self, region: RegionBlock, data: Tdata) -> Tdata:
        if False:
            print('Hello World!')
        'This is called when a switch region is visited.\n\n        When overriding this method, remember to handle the merging path of\n        ``data`` for all the branches in the switch when joining into the tail.\n        '
        pass

    def visit_linear(self, region: RegionBlock, data: Tdata) -> Tdata:
        if False:
            while True:
                i = 10
        'This is called when a linear region is visited.'
        return self.visit_graph(region.subregion, data)

    def visit_graph(self, scfg: SCFG, data: Tdata) -> Tdata:
        if False:
            for i in range(10):
                print('nop')
        'Process a SCFG in topological order.'
        toposorted = self._toposort_graph(scfg)
        label: str
        for lvl in toposorted:
            for label in lvl:
                data = self.visit(scfg[label], data)
        return data

    def _toposort_graph(self, scfg: SCFG):
        if False:
            print('Hello World!')
        toposorted = toposort_graph(scfg.graph)
        if self.direction == 'forward':
            return toposorted
        elif self.direction == 'backward':
            return reversed(toposorted)
        else:
            assert False, f'invalid direction {self.direction!r}'

    def visit(self, block: BasicBlock, data: Tdata) -> Tdata:
        if False:
            return 10
        'A generic visit method that will dispatch to the correct'
        if isinstance(block, RegionBlock):
            if block.kind == 'loop':
                fn = self.visit_loop
            elif block.kind == 'switch':
                fn = self.visit_switch
            else:
                raise NotImplementedError('unreachable')
            data = fn(block, data)
        else:
            data = self.visit_block(block, data)
        return data

class RegionTransformer(abc.ABC, Generic[Tdata]):
    """A mutating pass over a SCFG.

    This class is similar to ``RegionVisitor`` but only a "forward" direction
    is supported.
    """

    @abc.abstractmethod
    def visit_block(self, parent: SCFG, block: BasicBlock, data: Tdata) -> Tdata:
        if False:
            return 10
        pass

    @abc.abstractmethod
    def visit_loop(self, parent: SCFG, region: RegionBlock, data: Tdata) -> Tdata:
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def visit_switch(self, parent: SCFG, region: RegionBlock, data: Tdata) -> Tdata:
        if False:
            i = 10
            return i + 15
        pass

    def visit_linear(self, parent: SCFG, region: RegionBlock, data: Tdata) -> Tdata:
        if False:
            return 10
        return self.visit_graph(region.subregion, data)

    def visit_graph(self, scfg: SCFG, data: Tdata) -> Tdata:
        if False:
            for i in range(10):
                print('nop')
        toposorted = toposort_graph(scfg.graph)
        label: str
        for lvl in toposorted:
            for label in lvl:
                data = self.visit(scfg, scfg[label], data)
        return data

    def visit(self, parent: SCFG, block: BasicBlock, data: Tdata) -> Tdata:
        if False:
            print('Hello World!')
        if isinstance(block, RegionBlock):
            if block.kind == 'loop':
                fn = self.visit_loop
            elif block.kind == 'switch':
                fn = self.visit_switch
            elif block.kind in {'head', 'tail', 'branch'}:
                fn = self.visit_linear
            else:
                raise NotImplementedError('unreachable', block.name, block.kind)
            data = fn(parent, block, data)
        else:
            data = self.visit_block(parent, block, data)
        return data