"""
Unit tests for the global optimization benchmark functions
"""
import numpy as np
from .. import go_benchmark_functions as gbf
import inspect

class TestGoBenchmarkFunctions:

    def setup_method(self):
        if False:
            print('Hello World!')
        bench_members = inspect.getmembers(gbf, inspect.isclass)
        self.benchmark_functions = {it[0]: it[1] for it in bench_members if issubclass(it[1], gbf.Benchmark)}

    def teardown_method(self):
        if False:
            print('Hello World!')
        pass

    def test_optimum_solution(self):
        if False:
            print('Hello World!')
        for (name, klass) in self.benchmark_functions.items():
            if name in ['Benchmark', 'LennardJones'] or name.startswith('Problem'):
                continue
            f = klass()
            if name in ['Damavandi', 'Csendes']:
                with np.errstate(divide='ignore', invalid='ignore'):
                    print(name, f.fun(np.asarray(f.global_optimum[0])), f.fglob)
                    assert np.isnan(f.fun(np.asarray(f.global_optimum[0])))
                    continue
            print(name, f.fun(np.asarray(f.global_optimum[0])), f.fglob)
            assert f.success(f.global_optimum[0])

    def test_solution_exists(self):
        if False:
            print('Hello World!')
        for (name, klass) in self.benchmark_functions.items():
            if name == 'Benchmark':
                continue
            f = klass()
            _ = f.fglob

    def test_bounds_access_subscriptable(self):
        if False:
            return 10
        for (name, klass) in self.benchmark_functions.items():
            if name == 'Benchmark' or name.startswith('Problem'):
                continue
            f = klass()
            _ = f.bounds[0]

    def test_redimension(self):
        if False:
            for i in range(10):
                print('nop')
        LJ = self.benchmark_functions['LennardJones']
        L = LJ()
        L.change_dimensions(10)
        x0 = L.initial_vector()
        assert len(x0) == 10
        bounds = L.bounds
        assert len(bounds) == 10
        assert L.N == 10