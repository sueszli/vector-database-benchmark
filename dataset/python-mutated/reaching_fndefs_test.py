"""Tests for reaching_fndefs module."""
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.platform import test

class ReachingFndefsAnalyzerTest(test.TestCase):

    def _parse_and_analyze(self, test_fn):
        if False:
            i = 10
            return i + 15
        (node, source) = parser.parse_entity(test_fn, future_features=())
        entity_info = transformer.EntityInfo(name=test_fn.__name__, source_code=source, source_file=None, future_features=(), namespace={})
        node = qual_names.resolve(node)
        namer = naming.Namer({})
        ctx = transformer.Context(entity_info, namer, None)
        node = activity.resolve(node, ctx)
        graphs = cfg.build(node)
        node = reaching_definitions.resolve(node, ctx, graphs)
        node = reaching_fndefs.resolve(node, ctx, graphs)
        return node

    def assertHasFnDefs(self, node):
        if False:
            print('Hello World!')
        anno.getanno(node, anno.Static.DEFINED_FNS_IN)
if __name__ == '__main__':
    test.main()