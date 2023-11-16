"""Backend-independent tests for the Python XLA client."""
import unittest
from absl.testing import absltest
import numpy as np
from xla.python import xla_client
try:
    import portpicker
except ImportError:
    portpicker = None
ops = xla_client.ops

class ShapeTest(absltest.TestCase):

    def testInvalidShapes(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(xla_client.XlaRuntimeError, 'invalid shape'):
            xla_client.Shape.array_shape(xla_client.PrimitiveType.F32, [-2, 4])
        with self.assertRaisesRegex(RuntimeError, 'layout minor_to_major field contains 1 element.*'):
            xla_client.Shape.array_shape(xla_client.PrimitiveType.F32, [2, 4], [3])
        with self.assertRaisesRegex(RuntimeError, 'layout minor_to_major field has out-of-bounds value.*'):
            xla_client.Shape.array_shape(xla_client.PrimitiveType.F32, [2, 4], [1, -1])

class ComputationPrinting(absltest.TestCase):

    def ExampleComputation(self):
        if False:
            for i in range(10):
                print('nop')
        builder = xla_client.XlaBuilder('acomputation')
        p0 = ops.Parameter(builder, 0, xla_client.shape_from_pyval(np.float32(0)))
        p1 = ops.Parameter(builder, 1, xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
        x = ops.Mul(p0, p1)
        ops.Add(x, x)
        return builder.build()

    def testComputationToHloText(self):
        if False:
            return 10
        computation = self.ExampleComputation()
        hlo_text = computation.as_hlo_text()
        self.assertTrue(hlo_text.startswith('HloModule acomputation'))

    def testComputationToHloGraph(self):
        if False:
            return 10
        computation = self.ExampleComputation()
        hlo_dot_graph = computation.as_hlo_dot_graph()
        self.assertTrue(hlo_dot_graph.startswith('digraph '))

    def testHloModuleToHloText(self):
        if False:
            for i in range(10):
                print('nop')
        computation = self.ExampleComputation()
        hlo_text = computation.as_hlo_module().to_string()
        self.assertTrue(hlo_text.startswith('HloModule acomputation'))

    def testHloModuleFromText(self):
        if False:
            i = 10
            return i + 15
        hlo_module_text = 'HloModule test\n        add {\n          x = f32[] parameter(0)\n          y = f32[] parameter(1)\n          ROOT add = f32[] add(x, y)\n        }\n        ENTRY entry {\n          p0 = f32[2,3] parameter(0)\n          start = f32[2,3] all-reduce-start(p0), to_apply=add\n          ROOT done = f32[2,3] all-reduce-done(start)\n        }'
        hlo_module = xla_client._xla.hlo_module_from_text(hlo_module_text)
        hlo_text = hlo_module.to_string()
        self.assertTrue(hlo_text.startswith('HloModule test'))

    def testHloModuleToHloGraph(self):
        if False:
            print('Hello World!')
        computation = self.ExampleComputation()
        hlo_dot_graph = xla_client._xla.hlo_module_to_dot_graph(computation.as_hlo_module())
        self.assertTrue(hlo_dot_graph.startswith('digraph '))

class ComputationHashTest(absltest.TestCase):

    def testHash(self):
        if False:
            return 10
        builder0 = xla_client.XlaBuilder('computation0')
        p0 = ops.Parameter(builder0, 0, xla_client.shape_from_pyval(np.float32(0)))
        p1 = ops.Parameter(builder0, 1, xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
        ops.Mul(p0, p1)
        computation0 = builder0.build()
        builder1 = xla_client.XlaBuilder('computation1')
        p0 = ops.Parameter(builder1, 0, xla_client.shape_from_pyval(np.float32(0)))
        p1 = ops.Parameter(builder1, 1, xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
        ops.Mul(p0, p1)
        computation1 = builder1.build()
        self.assertEqual(computation0.hash(), computation1.hash())

class AliasTest(absltest.TestCase):

    def testSetUpAlias(self):
        if False:
            print('Hello World!')
        c = xla_client.XlaBuilder(self.id())
        p1 = ops.Parameter(c, 0, xla_client.shape_from_pyval(np.array(1.0, np.float32)).with_major_to_minor_layout_if_absent())
        p2 = ops.Parameter(c, 1, xla_client.shape_from_pyval(np.array(1.0, np.float32)).with_major_to_minor_layout_if_absent())
        out = ops.Add(p1, p2)
        c.setup_alias([], 0, [])
        c.build(out)

class ProfilerTest(absltest.TestCase):

    def testTraceMe(self):
        if False:
            return 10
        with xla_client.profiler.TraceMe('test1'):
            pass
        with xla_client.profiler.TraceMe('test2', foo=123):
            pass
        with self.assertRaises(ValueError):
            with xla_client.profiler.TraceMe('test3'):
                raise ValueError('test')

    @unittest.skipIf(portpicker is None, 'Test requires portpicker')
    def testStartServer(self):
        if False:
            while True:
                i = 10
        port = portpicker.pick_unused_port()
        server = xla_client.profiler.start_server(port)
        del server

class HloModuleGroupTest(absltest.TestCase):

    def testHloModuleGroup(self):
        if False:
            while True:
                i = 10
        builder0 = xla_client.XlaBuilder('computation0')
        p0 = ops.Parameter(builder0, 0, xla_client.shape_from_pyval(np.float32(0)))
        p1 = ops.Parameter(builder0, 1, xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
        root = ops.Mul(p0, p1)
        computation0 = builder0.build(root)
        m = computation0.get_hlo_module()
        mg_name = 'test_module_group'
        mg = xla_client._xla.HloModuleGroup(mg_name, [m])
        self.assertEqual(mg.name, mg_name)
        modules = mg.to_modules()
        self.assertLen(modules, 1)
        self.assertEqual(m.to_string(), modules[0].to_string())

class RunHloPassTest(absltest.TestCase):

    def testHloDCE(self):
        if False:
            i = 10
            return i + 15
        b = xla_client.XlaBuilder('acomputation')
        p0 = ops.Parameter(b, 0, xla_client.shape_from_pyval(np.float32(0)))
        p1 = ops.Parameter(b, 1, xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
        root = ops.Mul(p0, p1)
        p2 = ops.Parameter(b, 2, xla_client.shape_from_pyval(np.float32(0)))
        ops.Add(p2, p2)
        hlo_module = b.build(root).get_hlo_module()
        self.assertTrue(xla_client._xla.HloDCE().run(hlo_module))
if __name__ == '__main__':
    absltest.main()