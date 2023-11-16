"""Remove Transform Mark for Sphinx"""
from __future__ import annotations
import re
from docutils import nodes
from pygments.lexers import Python3Lexer, PythonLexer, guess_lexer
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms.code import TrimDoctestFlagsTransform
docmark_re = re.compile('\\s*#\\s*\\[(START|END)\\s*[a-z_A-Z]+].*$', re.MULTILINE)

class TrimDocMarkerFlagsTransform(SphinxTransform):
    """
    Trim doc marker like ``# [START howto_concept]` from python code-blocks.

    Based on:
    https://github.com/sphinx-doc/sphinx/blob/master/sphinx/transforms/post_transforms/code.py
    class TrimDoctestFlagsTransform
    """
    default_priority = TrimDoctestFlagsTransform.default_priority + 1

    def apply(self, **kwargs):
        if False:
            while True:
                i = 10
        for node in self.document.traverse(nodes.literal_block):
            if self.is_pycode(node):
                source = node.rawsource
                source = docmark_re.sub('', source)
                node.rawsource = source
                node[:] = [nodes.Text(source)]

    @staticmethod
    def is_pycode(node: nodes.literal_block) -> bool:
        if False:
            while True:
                i = 10
        'Checks if the node is literal block of python'
        if node.rawsource != node.astext():
            return False
        language = node.get('language')
        if language in ('py', 'py3', 'python', 'python3', 'default'):
            return True
        elif language == 'guess':
            try:
                lexer = guess_lexer(node.rawsource)
                return isinstance(lexer, (PythonLexer, Python3Lexer))
            except Exception:
                pass
        return False

def setup(app):
    if False:
        return 10
    'Sets the transform up'
    app.add_post_transform(TrimDocMarkerFlagsTransform)
    return {'version': 'builtin', 'parallel_read_safe': True, 'parallel_write_safe': True}