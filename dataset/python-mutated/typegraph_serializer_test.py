"""Tests for typegraph_serializer.py."""
from pytype.tests import test_base
from pytype.tests import test_utils
from pytype.typegraph import typegraph_serializer

class TypegraphSerializerTest(test_base.BaseTest):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        ctx = test_utils.make_context(self.options)
        (loc, defs) = ctx.vm.run_program(src='', filename='', maximum_depth=3)
        ctx.vm.analyze(loc, defs, maximum_depth=3)
        prog = ctx.program
        enc = typegraph_serializer.TypegraphEncoder()
        ser = enc.default(prog)
        self.assertEqual(len(prog.cfg_nodes), len(ser['cfg_nodes']))
        self.assertEqual(len(prog.variables), len(ser['variables']))
        self.assertEqual(prog.next_binding_id, len(ser['bindings']))
        self.assertEqual(prog.entrypoint.id, ser['entrypoint'])

    def test_deserialize(self):
        if False:
            for i in range(10):
                print('nop')
        ctx = test_utils.make_context(self.options)
        (loc, defs) = ctx.vm.run_program(src='', filename='', maximum_depth=3)
        ctx.vm.analyze(loc, defs, maximum_depth=3)
        prog = ctx.program
        ser = typegraph_serializer.encode_program(prog)
        dec = typegraph_serializer.decode_program(ser)
        self.assertIsInstance(dec, typegraph_serializer.SerializedProgram)
if __name__ == '__main__':
    test_base.main()