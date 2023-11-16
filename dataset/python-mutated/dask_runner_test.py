import inspect
import unittest
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
try:
    from apache_beam.runners.dask.dask_runner import DaskOptions
    from apache_beam.runners.dask.dask_runner import DaskRunner
    import dask
    import dask.distributed as ddist
except (ImportError, ModuleNotFoundError):
    raise unittest.SkipTest('Dask must be installed to run tests.')

class DaskOptionsTest(unittest.TestCase):

    def test_parses_connection_timeout__defaults_to_none(self):
        if False:
            i = 10
            return i + 15
        default_options = PipelineOptions([])
        default_dask_options = default_options.view_as(DaskOptions)
        self.assertEqual(None, default_dask_options.timeout)

    def test_parses_connection_timeout__parses_int(self):
        if False:
            for i in range(10):
                print('nop')
        conn_options = PipelineOptions('--dask_connection_timeout 12'.split())
        dask_conn_options = conn_options.view_as(DaskOptions)
        self.assertEqual(12, dask_conn_options.timeout)

    def test_parses_connection_timeout__handles_bad_input(self):
        if False:
            print('Hello World!')
        err_options = PipelineOptions('--dask_connection_timeout foo'.split())
        dask_err_options = err_options.view_as(DaskOptions)
        self.assertEqual(dask.config.no_default, dask_err_options.timeout)

    def test_parser_destinations__agree_with_dask_client(self):
        if False:
            return 10
        options = PipelineOptions('--dask_client_address localhost:8080 --dask_connection_timeout 600 --dask_scheduler_file foobar.cfg --dask_client_name charlie --dask_connection_limit 1024'.split())
        dask_options = options.view_as(DaskOptions)
        client_args = list(inspect.signature(ddist.Client).parameters)
        for opt_name in dask_options.get_all_options(drop_default=True).keys():
            with self.subTest(f'{opt_name} in dask.distributed.Client constructor'):
                self.assertIn(opt_name, client_args)

class DaskRunnerRunPipelineTest(unittest.TestCase):
    """Test class used to introspect the dask runner via a debugger."""

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.pipeline = test_pipeline.TestPipeline(runner=DaskRunner())

    def test_create(self):
        if False:
            i = 10
            return i + 15
        with self.pipeline as p:
            pcoll = p | beam.Create([1])
            assert_that(pcoll, equal_to([1]))

    def test_create_and_map(self):
        if False:
            return 10

        def double(x):
            if False:
                print('Hello World!')
            return x * 2
        with self.pipeline as p:
            pcoll = p | beam.Create([1]) | beam.Map(double)
            assert_that(pcoll, equal_to([2]))

    def test_create_map_and_groupby(self):
        if False:
            i = 10
            return i + 15

        def double(x):
            if False:
                i = 10
                return i + 15
            return (x * 2, x)
        with self.pipeline as p:
            pcoll = p | beam.Create([1]) | beam.Map(double) | beam.GroupByKey()
            assert_that(pcoll, equal_to([(2, [1])]))
if __name__ == '__main__':
    unittest.main()