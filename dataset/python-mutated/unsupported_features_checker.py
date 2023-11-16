"""Checkers for detecting unsupported Python features."""
import gast
from nvidia.dali._autograph.pyct import errors

class UnsupportedFeaturesChecker(gast.NodeVisitor):
    """Quick check for Python features we know we don't support.

  Any features detected will cause AutoGraph to not compile a function.
  """

    def visit_Attribute(self, node):
        if False:
            return 10
        if node.attr is not None and node.attr.startswith('__') and (not node.attr.endswith('__')):
            raise errors.UnsupportedLanguageElementError('mangled names are not yet supported')
        self.generic_visit(node)

    def visit_For(self, node):
        if False:
            i = 10
            return i + 15
        if node.orelse:
            raise errors.UnsupportedLanguageElementError('for/else statement not yet supported')
        self.generic_visit(node)

    def visit_While(self, node):
        if False:
            print('Hello World!')
        if node.orelse:
            raise errors.UnsupportedLanguageElementError('while/else statement not yet supported')
        self.generic_visit(node)

    def visit_Yield(self, node):
        if False:
            print('Hello World!')
        raise errors.UnsupportedLanguageElementError('generators are not supported')

    def visit_YieldFrom(self, node):
        if False:
            return 10
        raise errors.UnsupportedLanguageElementError('generators are not supported')

def verify(node):
    if False:
        return 10
    UnsupportedFeaturesChecker().visit(node)