import math
import unittest
from prometheus_client.core import CollectorRegistry, CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily, Metric, Sample, SummaryMetricFamily
from prometheus_client.exposition import generate_latest
from prometheus_client.parser import text_string_to_metric_families

class TestParse(unittest.TestCase):

    def assertEqualMetrics(self, first, second, msg=None):
        if False:
            print('Hello World!')
        super().assertEqual(first, second, msg)
        for (a, b) in zip(first, second):
            for (sa, sb) in zip(a.samples, b.samples):
                assert sa.name == sb.name

    def test_simple_counter(self):
        if False:
            print('Hello World!')
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a help\na 1\n')
        self.assertEqualMetrics([CounterMetricFamily('a', 'help', value=1)], list(families))

    def test_simple_gauge(self):
        if False:
            while True:
                i = 10
        families = text_string_to_metric_families('# TYPE a gauge\n# HELP a help\na 1\n')
        self.assertEqualMetrics([GaugeMetricFamily('a', 'help', value=1)], list(families))

    def test_simple_summary(self):
        if False:
            i = 10
            return i + 15
        families = text_string_to_metric_families('# TYPE a summary\n# HELP a help\na_count 1\na_sum 2\n')
        summary = SummaryMetricFamily('a', 'help', count_value=1, sum_value=2)
        self.assertEqualMetrics([summary], list(families))

    def test_summary_quantiles(self):
        if False:
            return 10
        families = text_string_to_metric_families('# TYPE a summary\n# HELP a help\na_count 1\na_sum 2\na{quantile="0.5"} 0.7\n')
        metric_family = SummaryMetricFamily('a', 'help', count_value=1, sum_value=2)
        metric_family.add_sample('a', {'quantile': '0.5'}, 0.7)
        self.assertEqualMetrics([metric_family], list(families))

    def test_simple_histogram(self):
        if False:
            return 10
        families = text_string_to_metric_families('# TYPE a histogram\n# HELP a help\na_bucket{le="1"} 0\na_bucket{le="+Inf"} 3\na_count 3\na_sum 2\n')
        self.assertEqualMetrics([HistogramMetricFamily('a', 'help', sum_value=2, buckets=[('1', 0.0), ('+Inf', 3.0)])], list(families))

    def test_no_metadata(self):
        if False:
            return 10
        families = text_string_to_metric_families('a 1\n')
        metric_family = Metric('a', '', 'untyped')
        metric_family.add_sample('a', {}, 1)
        self.assertEqualMetrics([metric_family], list(families))

    def test_untyped(self):
        if False:
            for i in range(10):
                print('nop')
        families = text_string_to_metric_families('# HELP redis_connected_clients Redis connected clients\n# TYPE redis_connected_clients untyped\nredis_connected_clients{instance="rough-snowflake-web",port="6380"} 10.0\nredis_connected_clients{instance="rough-snowflake-web",port="6381"} 12.0\n')
        m = Metric('redis_connected_clients', 'Redis connected clients', 'untyped')
        m.samples = [Sample('redis_connected_clients', {'instance': 'rough-snowflake-web', 'port': '6380'}, 10), Sample('redis_connected_clients', {'instance': 'rough-snowflake-web', 'port': '6381'}, 12)]
        self.assertEqualMetrics([m], list(families))

    def test_type_help_switched(self):
        if False:
            print('Hello World!')
        families = text_string_to_metric_families('# HELP a help\n# TYPE a counter\na 1\n')
        self.assertEqualMetrics([CounterMetricFamily('a', 'help', value=1)], list(families))

    def test_blank_lines_and_comments(self):
        if False:
            while True:
                i = 10
        families = text_string_to_metric_families('\n# TYPE a counter\n# FOO a\n# BAR b\n# HELP a help\n\na 1\n')
        self.assertEqualMetrics([CounterMetricFamily('a', 'help', value=1)], list(families))

    def test_tabs(self):
        if False:
            i = 10
            return i + 15
        families = text_string_to_metric_families('#\tTYPE\ta\tcounter\n#\tHELP\ta\thelp\na\t1\n')
        self.assertEqualMetrics([CounterMetricFamily('a', 'help', value=1)], list(families))

    def test_labels_with_curly_braces(self):
        if False:
            for i in range(10):
                print('nop')
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a help\na{foo="bar", bar="b{a}z"} 1\n')
        metric_family = CounterMetricFamily('a', 'help', labels=['foo', 'bar'])
        metric_family.add_metric(['bar', 'b{a}z'], 1)
        self.assertEqualMetrics([metric_family], list(families))

    def test_empty_help(self):
        if False:
            for i in range(10):
                print('nop')
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a\na 1\n')
        self.assertEqualMetrics([CounterMetricFamily('a', '', value=1)], list(families))

    def test_labels_and_infinite(self):
        if False:
            for i in range(10):
                print('nop')
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a help\na{foo="bar"} +Inf\na{foo="baz"} -Inf\n')
        metric_family = CounterMetricFamily('a', 'help', labels=['foo'])
        metric_family.add_metric(['bar'], float('inf'))
        metric_family.add_metric(['baz'], float('-inf'))
        self.assertEqualMetrics([metric_family], list(families))

    def test_spaces(self):
        if False:
            while True:
                i = 10
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a help\na{ foo = "bar" } 1\na\t\t{\t\tfoo\t\t=\t\t"baz"\t\t}\t\t2\na   {    foo   =  "buz"   }    3\na\t {  \t foo\t = "biz"\t  } \t 4\na \t{\t foo   = "boz"\t}\t 5\na{foo="bez"}6\n')
        metric_family = CounterMetricFamily('a', 'help', labels=['foo'])
        metric_family.add_metric(['bar'], 1)
        metric_family.add_metric(['baz'], 2)
        metric_family.add_metric(['buz'], 3)
        metric_family.add_metric(['biz'], 4)
        metric_family.add_metric(['boz'], 5)
        metric_family.add_metric(['bez'], 6)
        self.assertEqualMetrics([metric_family], list(families))

    def test_commas(self):
        if False:
            return 10
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a help\na{foo="bar",} 1\na{foo="baz",  } 1\n# TYPE b counter\n# HELP b help\nb{,} 2\n# TYPE c counter\n# HELP c help\nc{  ,} 3\n# TYPE d counter\n# HELP d help\nd{,  } 4\n')
        a = CounterMetricFamily('a', 'help', labels=['foo'])
        a.add_metric(['bar'], 1)
        a.add_metric(['baz'], 1)
        b = CounterMetricFamily('b', 'help', value=2)
        c = CounterMetricFamily('c', 'help', value=3)
        d = CounterMetricFamily('d', 'help', value=4)
        self.assertEqualMetrics([a, b, c, d], list(families))

    def test_multiple_trailing_commas(self):
        if False:
            i = 10
            return i + 15
        text = '# TYPE a counter\n# HELP a help\na{foo="bar",, } 1\n'
        self.assertRaises(ValueError, lambda : list(text_string_to_metric_families(text)))

    def test_empty_brackets(self):
        if False:
            while True:
                i = 10
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a help\na{} 1\n')
        self.assertEqualMetrics([CounterMetricFamily('a', 'help', value=1)], list(families))

    def test_nan(self):
        if False:
            while True:
                i = 10
        families = text_string_to_metric_families('a NaN\n')
        self.assertTrue(math.isnan(list(families)[0].samples[0][2]))

    def test_empty_label(self):
        if False:
            print('Hello World!')
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a help\na{foo="bar"} 1\na{foo=""} 2\n')
        metric_family = CounterMetricFamily('a', 'help', labels=['foo'])
        metric_family.add_metric(['bar'], 1)
        metric_family.add_metric([''], 2)
        self.assertEqualMetrics([metric_family], list(families))

    def test_label_escaping(self):
        if False:
            while True:
                i = 10
        for (escaped_val, unescaped_val) in [('foo', 'foo'), ('\\foo', '\\foo'), ('\\\\foo', '\\foo'), ('foo\\\\', 'foo\\'), ('\\\\', '\\'), ('\\n', '\n'), ('\\\\n', '\\n'), ('\\\\\\n', '\\\n'), ('\\"', '"'), ('\\\\\\"', '\\"')]:
            families = list(text_string_to_metric_families('\n# TYPE a counter\n# HELP a help\na{foo="%s",bar="baz"} 1\n' % escaped_val))
            metric_family = CounterMetricFamily('a', 'help', labels=['foo', 'bar'])
            metric_family.add_metric([unescaped_val, 'baz'], 1)
            self.assertEqualMetrics([metric_family], list(families))

    def test_help_escaping(self):
        if False:
            print('Hello World!')
        for (escaped_val, unescaped_val) in [('foo', 'foo'), ('\\foo', '\\foo'), ('\\\\foo', '\\foo'), ('foo\\', 'foo\\'), ('foo\\\\', 'foo\\'), ('\\n', '\n'), ('\\\\n', '\\n'), ('\\\\\\n', '\\\n'), ('\\"', '\\"'), ('\\\\"', '\\"'), ('\\\\\\"', '\\\\"')]:
            families = list(text_string_to_metric_families('\n# TYPE a counter\n# HELP a %s\na{foo="bar"} 1\n' % escaped_val))
            metric_family = CounterMetricFamily('a', unescaped_val, labels=['foo'])
            metric_family.add_metric(['bar'], 1)
            self.assertEqualMetrics([metric_family], list(families))

    def test_escaping(self):
        if False:
            i = 10
            return i + 15
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a he\\n\\\\l\\tp\na{foo="b\\"a\\nr"} 1\na{foo="b\\\\a\\z"} 2\n')
        metric_family = CounterMetricFamily('a', 'he\n\\l\\tp', labels=['foo'])
        metric_family.add_metric(['b"a\nr'], 1)
        metric_family.add_metric(['b\\a\\z'], 2)
        self.assertEqualMetrics([metric_family], list(families))

    def test_timestamps(self):
        if False:
            print('Hello World!')
        families = text_string_to_metric_families('# TYPE a counter\n# HELP a help\na{foo="bar"} 1\t000\n# TYPE b counter\n# HELP b help\nb 2  1234567890\nb 88   1234566000   \n')
        a = CounterMetricFamily('a', 'help', labels=['foo'])
        a.add_metric(['bar'], 1, timestamp=0)
        b = CounterMetricFamily('b', 'help')
        b.add_metric([], 2, timestamp=1234567.89)
        b.add_metric([], 88, timestamp=1234566)
        self.assertEqualMetrics([a, b], list(families))

    def test_roundtrip(self):
        if False:
            while True:
                i = 10
        text = '# HELP go_gc_duration_seconds A summary of the GC invocation durations.\n# TYPE go_gc_duration_seconds summary\ngo_gc_duration_seconds{quantile="0"} 0.013300656000000001\ngo_gc_duration_seconds{quantile="0.25"} 0.013638736\ngo_gc_duration_seconds{quantile="0.5"} 0.013759906\ngo_gc_duration_seconds{quantile="0.75"} 0.013962066\ngo_gc_duration_seconds{quantile="1"} 0.021383540000000003\ngo_gc_duration_seconds_sum 56.12904785\ngo_gc_duration_seconds_count 7476.0\n# HELP go_goroutines Number of goroutines that currently exist.\n# TYPE go_goroutines gauge\ngo_goroutines 166.0\n# HELP prometheus_local_storage_indexing_batch_duration_milliseconds Quantiles for batch indexing duration in milliseconds.\n# TYPE prometheus_local_storage_indexing_batch_duration_milliseconds summary\nprometheus_local_storage_indexing_batch_duration_milliseconds{quantile="0.5"} NaN\nprometheus_local_storage_indexing_batch_duration_milliseconds{quantile="0.9"} NaN\nprometheus_local_storage_indexing_batch_duration_milliseconds{quantile="0.99"} NaN\nprometheus_local_storage_indexing_batch_duration_milliseconds_sum 871.5665949999999\nprometheus_local_storage_indexing_batch_duration_milliseconds_count 229.0\n# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.\n# TYPE process_cpu_seconds_total counter\nprocess_cpu_seconds_total 29323.4\n# HELP process_virtual_memory_bytes Virtual memory size in bytes.\n# TYPE process_virtual_memory_bytes gauge\nprocess_virtual_memory_bytes 2.478268416e+09\n# HELP prometheus_build_info A metric with a constant \'1\' value labeled by version, revision, and branch from which Prometheus was built.\n# TYPE prometheus_build_info gauge\nprometheus_build_info{branch="HEAD",revision="ef176e5",version="0.16.0rc1"} 1.0\n# HELP prometheus_local_storage_chunk_ops_total The total number of chunk operations by their type.\n# TYPE prometheus_local_storage_chunk_ops_total counter\nprometheus_local_storage_chunk_ops_total{type="clone"} 28.0\nprometheus_local_storage_chunk_ops_total{type="create"} 997844.0\nprometheus_local_storage_chunk_ops_total{type="drop"} 1.345758e+06\nprometheus_local_storage_chunk_ops_total{type="load"} 1641.0\nprometheus_local_storage_chunk_ops_total{type="persist"} 981408.0\nprometheus_local_storage_chunk_ops_total{type="pin"} 32662.0\nprometheus_local_storage_chunk_ops_total{type="transcode"} 980180.0\nprometheus_local_storage_chunk_ops_total{type="unpin"} 32662.0\n'
        families = list(text_string_to_metric_families(text))

        class TextCollector:

            def collect(self):
                if False:
                    while True:
                        i = 10
                return families
        registry = CollectorRegistry()
        registry.register(TextCollector())
        self.assertEqual(text.encode('utf-8'), generate_latest(registry))
if __name__ == '__main__':
    unittest.main()