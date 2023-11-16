import time
import unittest
from prometheus_client import CollectorRegistry, Counter, Enum, Gauge, Histogram, Info, Metric, Summary
from prometheus_client.core import Exemplar, GaugeHistogramMetricFamily, Timestamp
from prometheus_client.openmetrics.exposition import generate_latest

class TestGenerateText(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.registry = CollectorRegistry()
        self.old_time = time.time
        time.time = lambda : 123.456

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        time.time = self.old_time

    def custom_collector(self, metric_family):
        if False:
            i = 10
            return i + 15

        class CustomCollector:

            def collect(self):
                if False:
                    print('Hello World!')
                return [metric_family]
        self.registry.register(CustomCollector())

    def test_counter(self):
        if False:
            i = 10
            return i + 15
        c = Counter('cc', 'A counter', registry=self.registry)
        c.inc()
        self.assertEqual(b'# HELP cc A counter\n# TYPE cc counter\ncc_total 1.0\ncc_created 123.456\n# EOF\n', generate_latest(self.registry))

    def test_counter_total(self):
        if False:
            return 10
        c = Counter('cc_total', 'A counter', registry=self.registry)
        c.inc()
        self.assertEqual(b'# HELP cc A counter\n# TYPE cc counter\ncc_total 1.0\ncc_created 123.456\n# EOF\n', generate_latest(self.registry))

    def test_counter_unit(self):
        if False:
            print('Hello World!')
        c = Counter('cc_seconds', 'A counter', registry=self.registry, unit='seconds')
        c.inc()
        self.assertEqual(b'# HELP cc_seconds A counter\n# TYPE cc_seconds counter\n# UNIT cc_seconds seconds\ncc_seconds_total 1.0\ncc_seconds_created 123.456\n# EOF\n', generate_latest(self.registry))

    def test_gauge(self):
        if False:
            for i in range(10):
                print('nop')
        g = Gauge('gg', 'A gauge', registry=self.registry)
        g.set(17)
        self.assertEqual(b'# HELP gg A gauge\n# TYPE gg gauge\ngg 17.0\n# EOF\n', generate_latest(self.registry))

    def test_summary(self):
        if False:
            i = 10
            return i + 15
        s = Summary('ss', 'A summary', ['a', 'b'], registry=self.registry)
        s.labels('c', 'd').observe(17)
        self.assertEqual(b'# HELP ss A summary\n# TYPE ss summary\nss_count{a="c",b="d"} 1.0\nss_sum{a="c",b="d"} 17.0\nss_created{a="c",b="d"} 123.456\n# EOF\n', generate_latest(self.registry))

    def test_histogram(self):
        if False:
            print('Hello World!')
        s = Histogram('hh', 'A histogram', registry=self.registry)
        s.observe(0.05)
        self.assertEqual(b'# HELP hh A histogram\n# TYPE hh histogram\nhh_bucket{le="0.005"} 0.0\nhh_bucket{le="0.01"} 0.0\nhh_bucket{le="0.025"} 0.0\nhh_bucket{le="0.05"} 1.0\nhh_bucket{le="0.075"} 1.0\nhh_bucket{le="0.1"} 1.0\nhh_bucket{le="0.25"} 1.0\nhh_bucket{le="0.5"} 1.0\nhh_bucket{le="0.75"} 1.0\nhh_bucket{le="1.0"} 1.0\nhh_bucket{le="2.5"} 1.0\nhh_bucket{le="5.0"} 1.0\nhh_bucket{le="7.5"} 1.0\nhh_bucket{le="10.0"} 1.0\nhh_bucket{le="+Inf"} 1.0\nhh_count 1.0\nhh_sum 0.05\nhh_created 123.456\n# EOF\n', generate_latest(self.registry))

    def test_histogram_negative_buckets(self):
        if False:
            print('Hello World!')
        s = Histogram('hh', 'A histogram', buckets=[-1, -0.5, 0, 0.5, 1], registry=self.registry)
        s.observe(-0.5)
        self.assertEqual(b'# HELP hh A histogram\n# TYPE hh histogram\nhh_bucket{le="-1.0"} 0.0\nhh_bucket{le="-0.5"} 1.0\nhh_bucket{le="0.0"} 1.0\nhh_bucket{le="0.5"} 1.0\nhh_bucket{le="1.0"} 1.0\nhh_bucket{le="+Inf"} 1.0\nhh_count 1.0\nhh_created 123.456\n# EOF\n', generate_latest(self.registry))

    def test_histogram_exemplar(self):
        if False:
            return 10
        s = Histogram('hh', 'A histogram', buckets=[1, 2, 3, 4], registry=self.registry)
        s.observe(0.5, {'a': 'b'})
        s.observe(1.5, {'le': '7'})
        s.observe(2.5, {'a': 'b'})
        s.observe(3.5, {'a': '\n"\\'})
        print(generate_latest(self.registry))
        self.assertEqual(b'# HELP hh A histogram\n# TYPE hh histogram\nhh_bucket{le="1.0"} 1.0 # {a="b"} 0.5 123.456\nhh_bucket{le="2.0"} 2.0 # {le="7"} 1.5 123.456\nhh_bucket{le="3.0"} 3.0 # {a="b"} 2.5 123.456\nhh_bucket{le="4.0"} 4.0 # {a="\\n\\"\\\\"} 3.5 123.456\nhh_bucket{le="+Inf"} 4.0\nhh_count 4.0\nhh_sum 8.0\nhh_created 123.456\n# EOF\n', generate_latest(self.registry))

    def test_counter_exemplar(self):
        if False:
            return 10
        c = Counter('cc', 'A counter', registry=self.registry)
        c.inc(exemplar={'a': 'b'})
        self.assertEqual(b'# HELP cc A counter\n# TYPE cc counter\ncc_total 1.0 # {a="b"} 1.0 123.456\ncc_created 123.456\n# EOF\n', generate_latest(self.registry))

    def test_untyped_exemplar(self):
        if False:
            for i in range(10):
                print('nop')

        class MyCollector:

            def collect(self):
                if False:
                    i = 10
                    return i + 15
                metric = Metric('hh', 'help', 'untyped')
                metric.add_sample('hh_bucket', {}, 0, None, Exemplar({'a': 'b'}, 0.5))
                yield metric
        self.registry.register(MyCollector())
        with self.assertRaises(ValueError):
            generate_latest(self.registry)

    def test_histogram_non_bucket_exemplar(self):
        if False:
            return 10

        class MyCollector:

            def collect(self):
                if False:
                    return 10
                metric = Metric('hh', 'help', 'histogram')
                metric.add_sample('hh_count', {}, 0, None, Exemplar({'a': 'b'}, 0.5))
                yield metric
        self.registry.register(MyCollector())
        with self.assertRaises(ValueError):
            generate_latest(self.registry)

    def test_counter_non_total_exemplar(self):
        if False:
            print('Hello World!')

        class MyCollector:

            def collect(self):
                if False:
                    return 10
                metric = Metric('cc', 'A counter', 'counter')
                metric.add_sample('cc_total', {}, 1, None, None)
                metric.add_sample('cc_created', {}, 123.456, None, Exemplar({'a': 'b'}, 1.0, 123.456))
                yield metric
        self.registry.register(MyCollector())
        with self.assertRaises(ValueError):
            generate_latest(self.registry)

    def test_gaugehistogram(self):
        if False:
            i = 10
            return i + 15
        self.custom_collector(GaugeHistogramMetricFamily('gh', 'help', buckets=[('1.0', 4), ('+Inf', 5)], gsum_value=7))
        self.assertEqual(b'# HELP gh help\n# TYPE gh gaugehistogram\ngh_bucket{le="1.0"} 4.0\ngh_bucket{le="+Inf"} 5.0\ngh_gcount 5.0\ngh_gsum 7.0\n# EOF\n', generate_latest(self.registry))

    def test_gaugehistogram_negative_buckets(self):
        if False:
            for i in range(10):
                print('nop')
        self.custom_collector(GaugeHistogramMetricFamily('gh', 'help', buckets=[('-1.0', 4), ('+Inf', 5)], gsum_value=-7))
        self.assertEqual(b'# HELP gh help\n# TYPE gh gaugehistogram\ngh_bucket{le="-1.0"} 4.0\ngh_bucket{le="+Inf"} 5.0\ngh_gcount 5.0\ngh_gsum -7.0\n# EOF\n', generate_latest(self.registry))

    def test_info(self):
        if False:
            while True:
                i = 10
        i = Info('ii', 'A info', ['a', 'b'], registry=self.registry)
        i.labels('c', 'd').info({'foo': 'bar'})
        self.assertEqual(b'# HELP ii A info\n# TYPE ii info\nii_info{a="c",b="d",foo="bar"} 1.0\n# EOF\n', generate_latest(self.registry))

    def test_enum(self):
        if False:
            i = 10
            return i + 15
        i = Enum('ee', 'An enum', ['a', 'b'], registry=self.registry, states=['foo', 'bar'])
        i.labels('c', 'd').state('bar')
        self.assertEqual(b'# HELP ee An enum\n# TYPE ee stateset\nee{a="c",b="d",ee="foo"} 0.0\nee{a="c",b="d",ee="bar"} 1.0\n# EOF\n', generate_latest(self.registry))

    def test_unicode(self):
        if False:
            i = 10
            return i + 15
        c = Counter('cc', '䔀', ['l'], registry=self.registry)
        c.labels('䔀').inc()
        self.assertEqual(b'# HELP cc \xe4\x94\x80\n# TYPE cc counter\ncc_total{l="\xe4\x94\x80"} 1.0\ncc_created{l="\xe4\x94\x80"} 123.456\n# EOF\n', generate_latest(self.registry))

    def test_escaping(self):
        if False:
            return 10
        c = Counter('cc', 'A\ncount\\er"', ['a'], registry=self.registry)
        c.labels('\\x\n"').inc(1)
        self.assertEqual(b'# HELP cc A\\ncount\\\\er\\"\n# TYPE cc counter\ncc_total{a="\\\\x\\n\\""} 1.0\ncc_created{a="\\\\x\\n\\""} 123.456\n# EOF\n', generate_latest(self.registry))

    def test_nonnumber(self):
        if False:
            return 10

        class MyNumber:

            def __repr__(self):
                if False:
                    return 10
                return 'MyNumber(123)'

            def __float__(self):
                if False:
                    while True:
                        i = 10
                return 123.0

        class MyCollector:

            def collect(self):
                if False:
                    for i in range(10):
                        print('nop')
                metric = Metric('nonnumber', 'Non number', 'untyped')
                metric.add_sample('nonnumber', {}, MyNumber())
                yield metric
        self.registry.register(MyCollector())
        self.assertEqual(b'# HELP nonnumber Non number\n# TYPE nonnumber unknown\nnonnumber 123.0\n# EOF\n', generate_latest(self.registry))

    def test_timestamp(self):
        if False:
            i = 10
            return i + 15

        class MyCollector:

            def collect(self):
                if False:
                    print('Hello World!')
                metric = Metric('ts', 'help', 'unknown')
                metric.add_sample('ts', {'foo': 'a'}, 0, 123.456)
                metric.add_sample('ts', {'foo': 'b'}, 0, -123.456)
                metric.add_sample('ts', {'foo': 'c'}, 0, 123)
                metric.add_sample('ts', {'foo': 'd'}, 0, Timestamp(123, 456000000))
                metric.add_sample('ts', {'foo': 'e'}, 0, Timestamp(123, 456000))
                metric.add_sample('ts', {'foo': 'f'}, 0, Timestamp(123, 456))
                yield metric
        self.registry.register(MyCollector())
        self.assertEqual(b'# HELP ts help\n# TYPE ts unknown\nts{foo="a"} 0.0 123.456\nts{foo="b"} 0.0 -123.456\nts{foo="c"} 0.0 123\nts{foo="d"} 0.0 123.456000000\nts{foo="e"} 0.0 123.000456000\nts{foo="f"} 0.0 123.000000456\n# EOF\n', generate_latest(self.registry))
if __name__ == '__main__':
    unittest.main()