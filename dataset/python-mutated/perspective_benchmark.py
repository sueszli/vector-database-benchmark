import os
from functools import partial
from bench import Benchmark, Suite, Runner
from perspective import Table
import perspective
import logging
SUPERSTORE_ARROW = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'node_modules', 'superstore-arrow', 'superstore.arrow')
with open(SUPERSTORE_ARROW, 'rb') as f:
    SUPERSTORE_ARROW_DATA = f.read()
VERSIONS = ['master', '0.4.1', '0.4.0rc6']

def make_meta(group, name):
    if False:
        for i in range(10):
            print('nop')
    return {'group': group, 'name': name}

def empty_callback(port_id):
    if False:
        for i in range(10):
            print('nop')
    pass

class PerspectiveBenchmark(Suite):
    AGG_OPTIONS = [[{'column': 'Sales', 'op': 'sum'}], [{'column': 'State', 'op': 'dominant'}], [{'column': 'Order Date', 'op': 'dominant'}]]
    split_by_OPTIONS = [[], ['Sub-Category'], ['Category', 'Sub-Category']]
    group_by_OPTIONS = [[], ['State'], ['State', 'City']]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Create a benchmark suite for the `perspective-python` runtime.'
        tbl = Table(SUPERSTORE_ARROW_DATA)
        for x in range(19):
            tbl.update(SUPERSTORE_ARROW_DATA)
        self._schema = tbl.schema()
        self._df_schema = tbl.schema()
        self._df_schema['Sales'] = int
        self._df_schema['Profit'] = int
        self._df_schema['Quantity'] = int
        self._view = tbl.view()
        self._table = tbl

    def _get_update_data(self, n=30):
        if False:
            return 10
        'Retrieve n rows from self.records to be used as update data.'
        return SUPERSTORE_ARROW_DATA

    def register_benchmarks(self):
        if False:
            while True:
                i = 10
        'Register all the benchmark methods - each method creates a number of\n        lambdas, and then calls `setattr` on the Suite itself so that the\n        `Runner` can find the tests at runtime.'
        self.benchmark_table_arrow()
        self.benchmark_table_csv()
        self.benchmark_view_zero()
        self.benchmark_view_one()
        self.benchmark_view_two()
        self.benchmark_view_zero_updates()
        self.benchmark_view_one_updates()
        self.benchmark_view_two_updates()
        self.benchmark_to_format_zero()
        self.benchmark_to_format_one()
        self.benchmark_to_format_two()

    def benchmark_table_arrow(self):
        if False:
            while True:
                i = 10
        'Benchmark table from arrow separately as it requires opening the\n        Arrow file from the filesystem.'
        test_meta = make_meta('table', 'arrow')
        func = Benchmark(lambda : Table(SUPERSTORE_ARROW_DATA), meta=test_meta)
        setattr(self, 'table_arrow', func)

    def benchmark_table_csv(self):
        if False:
            while True:
                i = 10
        'Benchmark table from csv separately as it requires opening the\n        Arrow file from the filesystem.'
        csv = self._view.to_csv()
        test_meta = make_meta('table', 'csv')
        func = Benchmark(lambda : Table(csv), meta=test_meta)
        setattr(self, 'table_csv', func)

    def benchmark_view_zero(self):
        if False:
            while True:
                i = 10
        'Benchmark view creation with zero pivots.'
        func = Benchmark(lambda : self._table.view(), meta=make_meta('view', 'zero'))
        setattr(self, 'view_zero', func)

    def benchmark_view_zero_updates(self):
        if False:
            i = 10
            return i + 15
        'Benchmark how long it takes for each update to resolve fully, using\n        the on update callback that forces resolution of updates across\n        10 views.'
        table = Table(self._schema)
        views = [table.view() for i in range(25)]
        for v in views:
            v.on_update(empty_callback)
        update_data = self._get_update_data(1000)

        def resolve_update():
            if False:
                print('Hello World!')
            table.update(update_data)
            table.size()
        func = Benchmark(resolve_update, meta=make_meta('update', 'zero'))
        setattr(self, 'update_zero', func)

    def benchmark_view_one(self):
        if False:
            while True:
                i = 10
        'Benchmark view creation with different pivots.'
        for pivot in PerspectiveBenchmark.group_by_OPTIONS:
            if len(pivot) == 0:
                continue
            test_meta = make_meta('view', 'one_{0}_pivot'.format(len(pivot)))
            view_constructor = partial(self._table.view, group_by=pivot)
            func = Benchmark(lambda : view_constructor(), meta=test_meta)
            setattr(self, 'view_{0}'.format(test_meta['name']), func)

    def benchmark_view_one_updates(self):
        if False:
            for i in range(10):
                print('nop')
        'Benchmark how long it takes for each update to resolve fully, using\n        the on update callback that forces resolution of updates across\n        25 views.'
        table = Table(self._schema)
        views = [table.view(group_by=['State', 'City']) for i in range(25)]
        for v in views:
            v.on_update(empty_callback)
        update_data = self._get_update_data(1000)

        def resolve_update():
            if False:
                while True:
                    i = 10
            table.update(update_data)
            table.size()
        func = Benchmark(resolve_update, meta=make_meta('update', 'one'))
        setattr(self, 'update_one', func)

    def benchmark_view_two(self):
        if False:
            return 10
        'Benchmark view creation with row and Split By.'
        for i in range(len(PerspectiveBenchmark.group_by_OPTIONS)):
            RP = PerspectiveBenchmark.group_by_OPTIONS[i]
            CP = PerspectiveBenchmark.split_by_OPTIONS[i]
            if len(RP) == 0 and len(CP) == 0:
                continue
            test_meta = make_meta('view', 'two_{0}x{1}_pivot'.format(len(RP), len(CP)))
            view_constructor = partial(self._table.view, group_by=RP, split_by=CP)
            func = Benchmark(lambda : view_constructor(), meta=test_meta)
            setattr(self, 'view_{0}'.format(test_meta['name']), func)

    def benchmark_view_two_updates(self):
        if False:
            while True:
                i = 10
        'Benchmark how long it takes for each update to resolve fully, using\n        the on update callback that forces resolution of updates across\n        25 views.'
        table = Table(self._schema)
        views = [table.view(group_by=['State', 'City'], split_by=['Category', 'Sub-Category']) for i in range(25)]
        for v in views:
            v.on_update(empty_callback)
        update_data = self._get_update_data(1000)

        def resolve_update():
            if False:
                i = 10
                return i + 15
            table.update(update_data)
            table.size()
        func = Benchmark(resolve_update, meta=make_meta('update', 'two'))
        setattr(self, 'update_two', func)

    def benchmark_to_format_zero(self):
        if False:
            return 10
        'Benchmark each `to_format` method.'
        for name in ('arrow', 'csv', 'columns', 'records'):
            method = 'to_{0}'.format(name)
            test_meta = make_meta('to_format', method)
            func = Benchmark(getattr(self._view, method), meta=test_meta)
            setattr(self, 'to_format_{0}'.format(name), func)

    def benchmark_to_format_one(self):
        if False:
            i = 10
            return i + 15
        'Benchmark each `to_format` method for one-sided contexts.'
        for name in ('arrow', 'csv', 'columns', 'records'):
            for pivot in PerspectiveBenchmark.group_by_OPTIONS:
                if len(pivot) == 0:
                    continue
                test_meta = make_meta('to_format', 'to_{0}_r{1}'.format(name, len(pivot)))
                view = self._table.view(group_by=pivot)
                method = 'to_{0}'.format(name)
                func = Benchmark(getattr(view, method), meta=test_meta)
                setattr(self, 'to_format_{0}'.format(test_meta['name']), func)

    def benchmark_to_format_two(self):
        if False:
            while True:
                i = 10
        'Benchmark each `to_format` method for two-sided contexts.'
        for name in ('arrow', 'csv', 'columns', 'records'):
            for i in range(len(PerspectiveBenchmark.group_by_OPTIONS)):
                RP = PerspectiveBenchmark.group_by_OPTIONS[i]
                CP = PerspectiveBenchmark.split_by_OPTIONS[i]
                if len(RP) == 0 and len(CP) == 0:
                    continue
                test_meta = make_meta('to_format', 'to_{0}_r{1}_c{2}'.format(name, len(RP), len(CP)))
                view = self._table.view(group_by=RP, split_by=CP)
                method = 'to_{0}'.format(name)
                func = Benchmark(getattr(view, method), meta=test_meta)
                setattr(self, 'to_format_{0}'.format(test_meta['name']), func)
if __name__ == '__main__':
    VERSION = os.environ.get('PSP_VERSION', 'Unknown Version')
    suite = PerspectiveBenchmark()
    runner = Runner(suite)
    logging.info('Benchmarking perspective-python=={}'.format(VERSION))
    logging.info('Detected {} {}'.format(perspective.__version__, perspective.__file__))
    runner.run(VERSION)
    runner.write_results()