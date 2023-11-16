"""Tests for reaching_fndefs module."""
import unittest
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import cfg
from nvidia.dali._autograph.pyct import naming
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import qual_names
from nvidia.dali._autograph.pyct import transformer
from nvidia.dali._autograph.pyct.static_analysis import activity
from nvidia.dali._autograph.pyct.static_analysis import reaching_definitions
from nvidia.dali._autograph.pyct.static_analysis import reaching_fndefs

class ReachingFndefsAnalyzerTest(unittest.TestCase):

    def _parse_and_analyze(self, test_fn):
        if False:
            return 10
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
            for i in range(10):
                print('nop')
        anno.getanno(node, anno.Static.DEFINED_FNS_IN)