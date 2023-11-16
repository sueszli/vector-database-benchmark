"""Tests for profiler_wrapper.cc pybind methods."""
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.profiler.internal import _pywrap_profiler as profiler_wrapper

class ProfilerSessionTest(test_util.TensorFlowTestCase):

    def test_xspace_to_tools_data_default_options(self):
        if False:
            while True:
                i = 10
        profiler_wrapper.xspace_to_tools_data([], 'trace_viewer')

    def _test_xspace_to_tools_data_options(self, options):
        if False:
            while True:
                i = 10
        profiler_wrapper.xspace_to_tools_data([], 'trace_viewer', options)

    def test_xspace_to_tools_data_empty_options(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_xspace_to_tools_data_options({})

    def test_xspace_to_tools_data_int_options(self):
        if False:
            return 10
        self._test_xspace_to_tools_data_options({'example_option': 0})

    def test_xspace_to_tools_data_str_options(self):
        if False:
            while True:
                i = 10
        self._test_xspace_to_tools_data_options({'example_option': 'example'})
if __name__ == '__main__':
    test.main()