from __future__ import absolute_import
from chart_studio.api.v2 import plot_schema
from chart_studio.tests.test_plot_ly.test_api import PlotlyApiTestCase

class PlotSchemaTest(PlotlyApiTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(PlotSchemaTest, self).setUp()
        self.request_mock = self.mock('chart_studio.api.v2.utils.requests.request')
        self.request_mock.return_value = self.get_response()
        self.mock('chart_studio.api.v2.utils.validate_response')

    def test_retrieve(self):
        if False:
            while True:
                i = 10
        plot_schema.retrieve('some-hash', timeout=400)
        assert self.request_mock.call_count == 1
        (args, kwargs) = self.request_mock.call_args
        (method, url) = args
        self.assertEqual(method, 'get')
        self.assertEqual(url, '{}/v2/plot-schema'.format(self.plotly_api_domain))
        self.assertTrue(kwargs['timeout'])
        self.assertEqual(kwargs['params'], {'sha1': 'some-hash'})