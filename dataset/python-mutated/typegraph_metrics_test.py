"""Basic tests for accessing typegraph metrics from Python."""
import textwrap
from pytype import context
from pytype import typegraph
from pytype.tests import test_base

class MetricsTest(test_base.BaseTest):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.ctx = context.Context(options=self.options, loader=self.loader)

    def run_program(self, src):
        if False:
            for i in range(10):
                print('nop')
        return self.ctx.vm.run_program(textwrap.dedent(src), '', maximum_depth=10)

    def assertNotEmpty(self, container, msg=None):
        if False:
            print('Hello World!')
        if not container:
            msg = msg or f'{container!r} has length of 0.'
            self.fail(msg=msg)

    def test_basics(self):
        if False:
            while True:
                i = 10
        self.run_program('\n        def foo(x: str) -> int:\n          return x + 1\n        a = foo(1)\n    ')
        metrics = self.ctx.program.calculate_metrics()
        self.assertIsInstance(metrics, typegraph.cfg.Metrics)
        self.assertGreater(metrics.binding_count, 0)
        self.assertNotEmpty(metrics.cfg_node_metrics)
        self.assertNotEmpty(metrics.variable_metrics)
        self.assertNotEmpty(metrics.solver_metrics)
        self.assertNotEmpty(metrics.solver_metrics[0].query_metrics)
if __name__ == '__main__':
    test_base.main()