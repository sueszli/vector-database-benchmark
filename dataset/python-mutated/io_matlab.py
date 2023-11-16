from .common import set_mem_rlimit, run_monitored, get_mem_info
import os
import tempfile
from io import BytesIO
import numpy as np
from .common import Benchmark, safe_import
with safe_import():
    from scipy.io import savemat, loadmat

class MemUsage(Benchmark):
    param_names = ['size', 'compressed']
    timeout = 4 * 60
    unit = 'actual/optimal memory usage ratio'

    @property
    def params(self):
        if False:
            while True:
                i = 10
        return [list(self._get_sizes().keys()), [True, False]]

    def _get_sizes(self):
        if False:
            print('Hello World!')
        sizes = {'1M': 1000000.0, '10M': 10000000.0, '100M': 100000000.0, '300M': 300000000.0}
        return sizes

    def setup(self, size, compressed):
        if False:
            while True:
                i = 10
        set_mem_rlimit()
        self.sizes = self._get_sizes()
        size = int(self.sizes[size])
        mem_info = get_mem_info()
        try:
            mem_available = mem_info['memavailable']
        except KeyError:
            mem_available = mem_info['memtotal']
        max_size = int(mem_available * 0.7) // 4
        if size > max_size:
            raise NotImplementedError()
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.mat')
        f.close()
        self.filename = f.name

    def teardown(self, size, compressed):
        if False:
            return 10
        os.unlink(self.filename)

    def track_loadmat(self, size, compressed):
        if False:
            for i in range(10):
                print('nop')
        size = int(self.sizes[size])
        x = np.random.rand(size // 8).view(dtype=np.uint8)
        savemat(self.filename, dict(x=x), do_compression=compressed, oned_as='row')
        del x
        code = "\n        from scipy.io import loadmat\n        loadmat('%s')\n        " % (self.filename,)
        (time, peak_mem) = run_monitored(code)
        return peak_mem / size

    def track_savemat(self, size, compressed):
        if False:
            return 10
        size = int(self.sizes[size])
        code = "\n        import numpy as np\n        from scipy.io import savemat\n        x = np.random.rand(%d//8).view(dtype=np.uint8)\n        savemat('%s', dict(x=x), do_compression=%r, oned_as='row')\n        " % (size, self.filename, compressed)
        (time, peak_mem) = run_monitored(code)
        return peak_mem / size

class StructArr(Benchmark):
    params = [[(10, 10, 20), (20, 20, 40), (30, 30, 50)], [False, True]]
    param_names = ['(vars, fields, structs)', 'compression']

    @staticmethod
    def make_structarr(n_vars, n_fields, n_structs):
        if False:
            for i in range(10):
                print('nop')
        var_dict = {}
        for vno in range(n_vars):
            vname = 'var%00d' % vno
            end_dtype = [('f%d' % d, 'i4', 10) for d in range(n_fields)]
            s_arrs = np.zeros((n_structs,), dtype=end_dtype)
            var_dict[vname] = s_arrs
        return var_dict

    def setup(self, nvfs, compression):
        if False:
            return 10
        (n_vars, n_fields, n_structs) = nvfs
        self.var_dict = StructArr.make_structarr(n_vars, n_fields, n_structs)
        self.str_io = BytesIO()
        savemat(self.str_io, self.var_dict, do_compression=compression)

    def time_savemat(self, nvfs, compression):
        if False:
            return 10
        savemat(self.str_io, self.var_dict, do_compression=compression)

    def time_loadmat(self, nvfs, compression):
        if False:
            while True:
                i = 10
        loadmat(self.str_io)