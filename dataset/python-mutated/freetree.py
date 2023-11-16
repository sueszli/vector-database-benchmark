"""Generic node traverser visitor"""
from __future__ import annotations
from mypy.nodes import Block, MypyFile
from mypy.traverser import TraverserVisitor

class TreeFreer(TraverserVisitor):

    def visit_block(self, block: Block) -> None:
        if False:
            while True:
                i = 10
        super().visit_block(block)
        block.body.clear()

def free_tree(tree: MypyFile) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Free all the ASTs associated with a module.\n\n    This needs to be done recursively, since symbol tables contain\n    references to definitions, so those won't be freed but we want their\n    contents to be.\n    "
    tree.accept(TreeFreer())
    tree.defs.clear()