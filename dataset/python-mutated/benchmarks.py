import conbench.runner
import _criterion

@conbench.runner.register_benchmark
class TestBenchmark(conbench.runner.Benchmark):
    name = 'test'

    def run(self, **kwargs):
        if False:
            return 10
        yield self.conbench.benchmark(self._f(), self.name, options=kwargs)

    def _f(self):
        if False:
            for i in range(10):
                print('nop')
        return lambda : 1 + 1

@conbench.runner.register_benchmark
class CargoBenchmarks(_criterion.CriterionBenchmark):
    name = 'datafusion'
    description = 'Run Arrow DataFusion micro benchmarks.'