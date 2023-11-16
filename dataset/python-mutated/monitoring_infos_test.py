import unittest
from apache_beam.metrics import monitoring_infos
from apache_beam.metrics.cells import CounterCell
from apache_beam.metrics.cells import GaugeCell

class MonitoringInfosTest(unittest.TestCase):

    def test_parse_namespace_and_name_for_nonuser_metric(self):
        if False:
            return 10
        input = monitoring_infos.create_monitoring_info('beam:dummy:metric', 'typeurn', None)
        (namespace, name) = monitoring_infos.parse_namespace_and_name(input)
        self.assertEqual(namespace, 'beam')
        self.assertEqual(name, 'dummy:metric')

    def test_parse_namespace_and_name_for_user_counter_metric(self):
        if False:
            i = 10
            return i + 15
        urn = monitoring_infos.USER_COUNTER_URN
        labels = {}
        labels[monitoring_infos.NAMESPACE_LABEL] = 'counternamespace'
        labels[monitoring_infos.NAME_LABEL] = 'countername'
        input = monitoring_infos.create_monitoring_info(urn, 'typeurn', None, labels)
        (namespace, name) = monitoring_infos.parse_namespace_and_name(input)
        self.assertEqual(namespace, 'counternamespace')
        self.assertEqual(name, 'countername')

    def test_parse_namespace_and_name_for_user_distribution_metric(self):
        if False:
            while True:
                i = 10
        urn = monitoring_infos.USER_DISTRIBUTION_URN
        labels = {}
        labels[monitoring_infos.NAMESPACE_LABEL] = 'counternamespace'
        labels[monitoring_infos.NAME_LABEL] = 'countername'
        input = monitoring_infos.create_monitoring_info(urn, 'typeurn', None, labels)
        (namespace, name) = monitoring_infos.parse_namespace_and_name(input)
        self.assertEqual(namespace, 'counternamespace')
        self.assertEqual(name, 'countername')

    def test_parse_namespace_and_name_for_user_gauge_metric(self):
        if False:
            print('Hello World!')
        urn = monitoring_infos.USER_GAUGE_URN
        labels = {}
        labels[monitoring_infos.NAMESPACE_LABEL] = 'counternamespace'
        labels[monitoring_infos.NAME_LABEL] = 'countername'
        input = monitoring_infos.create_monitoring_info(urn, 'typeurn', None, labels)
        (namespace, name) = monitoring_infos.parse_namespace_and_name(input)
        self.assertEqual(namespace, 'counternamespace')
        self.assertEqual(name, 'countername')

    def test_int64_user_gauge(self):
        if False:
            i = 10
            return i + 15
        metric = GaugeCell().get_cumulative()
        result = monitoring_infos.int64_user_gauge('gaugenamespace', 'gaugename', metric)
        (_, gauge_value) = monitoring_infos.extract_gauge_value(result)
        self.assertEqual(0, gauge_value)

    def test_int64_user_counter(self):
        if False:
            return 10
        expected_labels = {}
        expected_labels[monitoring_infos.NAMESPACE_LABEL] = 'counternamespace'
        expected_labels[monitoring_infos.NAME_LABEL] = 'countername'
        metric = CounterCell().get_cumulative()
        result = monitoring_infos.int64_user_counter('counternamespace', 'countername', metric)
        counter_value = monitoring_infos.extract_counter_value(result)
        self.assertEqual(0, counter_value)
        self.assertEqual(result.labels, expected_labels)

    def test_int64_counter(self):
        if False:
            return 10
        expected_labels = {}
        expected_labels[monitoring_infos.PCOLLECTION_LABEL] = 'collectionname'
        expected_labels[monitoring_infos.PTRANSFORM_LABEL] = 'ptransformname'
        expected_labels[monitoring_infos.SERVICE_LABEL] = 'BigQuery'
        labels = {monitoring_infos.SERVICE_LABEL: 'BigQuery'}
        metric = CounterCell().get_cumulative()
        result = monitoring_infos.int64_counter(monitoring_infos.API_REQUEST_COUNT_URN, metric, ptransform='ptransformname', pcollection='collectionname', labels=labels)
        counter_value = monitoring_infos.extract_counter_value(result)
        self.assertEqual(0, counter_value)
        self.assertEqual(result.labels, expected_labels)
if __name__ == '__main__':
    unittest.main()