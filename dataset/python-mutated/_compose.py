from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .app import App
    from .widget import Widget

def compose(node: App | Widget) -> list[Widget]:
    if False:
        while True:
            i = 10
    'Compose child widgets.\n\n    Args:\n        node: The parent node.\n\n    Returns:\n        A list of widgets.\n    '
    app = node.app
    nodes: list[Widget] = []
    compose_stack: list[Widget] = []
    composed: list[Widget] = []
    app._compose_stacks.append(compose_stack)
    app._composed.append(composed)
    try:
        for child in node.compose():
            if composed:
                nodes.extend(composed)
                composed.clear()
            if compose_stack:
                compose_stack[-1].compose_add_child(child)
            else:
                nodes.append(child)
        if composed:
            nodes.extend(composed)
            composed.clear()
    finally:
        app._compose_stacks.pop()
        app._composed.pop()
    return nodes