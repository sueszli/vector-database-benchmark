"""These benchmarks are supposed to be run only for modin, since they do not make sense for pandas."""
import modin.pandas as pd
from modin.pandas.utils import from_pandas
try:
    from modin.utils import to_numpy, to_pandas
except ImportError:
    from modin.pandas.utils import to_pandas
import pandas
from ..utils import RAND_HIGH, RAND_LOW, execute, gen_data, generate_dataframe, get_benchmark_shapes

class TimeFromPandas:
    param_names = ['shape', 'cpus']
    params = [get_benchmark_shapes('TimeFromPandas'), [4, 16, 32]]

    def setup(self, shape, cpus):
        if False:
            return 10
        self.data = pandas.DataFrame(gen_data('int', *shape, RAND_LOW, RAND_HIGH))
        from modin.config import NPartitions
        NPartitions.get = lambda : cpus
        pd.DataFrame([])

    def time_from_pandas(self, shape, cpus):
        if False:
            for i in range(10):
                print('nop')
        execute(from_pandas(self.data))

class TimeToPandas:
    param_names = ['shape', 'cpus']
    params = [get_benchmark_shapes('TimeToPandas'), [4, 16, 32]]

    def setup(self, shape, cpus):
        if False:
            for i in range(10):
                print('nop')
        from modin.config import NPartitions
        NPartitions.get = lambda : cpus
        self.data = generate_dataframe('int', *shape, RAND_LOW, RAND_HIGH, impl='modin')

    def time_to_pandas(self, shape, cpus):
        if False:
            for i in range(10):
                print('nop')
        to_pandas(self.data)

class TimeToNumPy:
    param_names = ['shape', 'cpus']
    params = [get_benchmark_shapes('TimeToNumPy'), [4, 16, 32]]

    def setup(self, shape, cpus):
        if False:
            for i in range(10):
                print('nop')
        from modin.config import NPartitions
        NPartitions.get = lambda : cpus
        self.data = generate_dataframe('int', *shape, RAND_LOW, RAND_HIGH, impl='modin')

    def time_to_numpy(self, shape, cpus):
        if False:
            print('Hello World!')
        to_numpy(self.data)
from ..utils import setup