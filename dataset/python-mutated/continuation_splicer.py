"""Insert "continuation" nodes into lib2to3 tree.

The "backslash-newline" continuation marker is shoved into the node's prefix.
Pull them out and make it into nodes of their own.

  SpliceContinuations(): the main function exported by this module.
"""
from yapf_third_party._ylib2to3 import pytree
from yapf.yapflib import format_token

def SpliceContinuations(tree):
    if False:
        while True:
            i = 10
    'Given a pytree, splice the continuation marker into nodes.\n\n  Arguments:\n    tree: (pytree.Node) The tree to work on. The tree is modified by this\n      function.\n  '

    def RecSplicer(node):
        if False:
            print('Hello World!')
        'Inserts a continuation marker into the node.'
        if isinstance(node, pytree.Leaf):
            if node.prefix.lstrip().startswith('\\\n'):
                new_lineno = node.lineno - node.prefix.count('\n')
                return pytree.Leaf(type=format_token.CONTINUATION, value=node.prefix, context=('', (new_lineno, 0)))
            return None
        num_inserted = 0
        for (index, child) in enumerate(node.children[:]):
            continuation_node = RecSplicer(child)
            if continuation_node:
                node.children.insert(index + num_inserted, continuation_node)
                num_inserted += 1
    RecSplicer(tree)