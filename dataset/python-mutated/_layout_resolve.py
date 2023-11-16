from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence, cast
from typing_extensions import Protocol

class EdgeProtocol(Protocol):
    """Any object that defines an edge (such as Layout)."""
    size: int | None
    fraction: int
    min_size: int

@dataclass
class Edge:
    size: int | None = None
    fraction: int | None = 1
    min_size: int = 1

def layout_resolve(total: int, edges: Sequence[EdgeProtocol]) -> list[int]:
    if False:
        i = 10
        return i + 15
    'Divide total space to satisfy size, fraction, and min_size, constraints.\n\n    The returned list of integers should add up to total in most cases, unless it is\n    impossible to satisfy all the constraints. For instance, if there are two edges\n    with a minimum size of 20 each and `total` is 30 then the returned list will be\n    greater than total. In practice, this would mean that a Layout object would\n    clip the rows that would overflow the screen height.\n\n    Args:\n        total: Total number of characters.\n        edges: Edges within total space.\n\n    Returns:\n        Number of characters for each edge.\n    '
    sizes = [edge.size or None for edge in edges]
    if None not in sizes:
        return cast('list[int]', sizes)
    flexible_edges = [(index, edge) for (index, (size, edge)) in enumerate(zip(sizes, edges)) if size is None]
    remaining = total - sum([size or 0 for size in sizes])
    if remaining <= 0:
        return [edge.min_size or 1 if size is None else size for (size, edge) in zip(sizes, edges)]
    total_flexible = sum([edge.fraction or 1 for (_, edge) in flexible_edges])
    while flexible_edges:
        portion = Fraction(remaining, total_flexible)
        for (flexible_index, (index, edge)) in enumerate(flexible_edges):
            if portion * edge.fraction < edge.min_size:
                sizes[index] = edge.min_size
                remaining -= edge.min_size
                total_flexible -= edge.fraction or 1
                del flexible_edges[flexible_index]
                break
        else:
            remainder = Fraction(0)
            for (index, edge) in flexible_edges:
                (sizes[index], remainder) = divmod(portion * edge.fraction + remainder, 1)
            break
    return cast('list[int]', sizes)