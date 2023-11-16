import json
from django.test import TestCase
from unittest.mock import patch
from django.core.exceptions import ValidationError
from dojo.models import Test, Engagement, Product, Product_API_Scan_Configuration, Tool_Type, Tool_Configuration
from dojo.tools.api_edgescan.importer import EdgescanImporter

class TestEdgescanImporter(TestCase):

    @classmethod
    def setUpTestData(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.tool_type = Tool_Type()
        cls.tool_configuration = Tool_Configuration()
        cls.tool_configuration.tool_type = cls.tool_type
        cls.tool_configuration.authentication_type = 'API'
        cls.tool_configuration.api_key = 'API_KEY'
        cls.tool_configuration.extras = '{"extras": "EXTRAS"}'
        cls.product = Product()
        cls.product.name = 'Product'
        cls.engagement = Engagement()
        cls.engagement.product = cls.product
        cls.test = Test()
        cls.test.engagement = cls.engagement
        cls.api_scan_configuration = Product_API_Scan_Configuration()
        cls.api_scan_configuration.product = cls.product
        cls.api_scan_configuration.tool_configuration = cls.tool_configuration
        cls.product_2 = Product()
        cls.product_2.name = 'Product_2'
        cls.engagement_2 = Engagement()
        cls.engagement_2.product = cls.product_2
        cls.test_2 = Test()
        cls.test_2.engagement = cls.engagement_2
        cls.api_scan_configuration_2 = Product_API_Scan_Configuration()
        cls.test_2.api_scan_configuration = cls.api_scan_configuration_2
        cls.api_scan_configuration_2.product = cls.product_2
        cls.api_scan_configuration_2.tool_configuration = cls.tool_configuration
        cls.api_scan_configuration_2.service_key_1 = 'SERVICE_KEY_1'
        cls.findings = json.dumps({'a': 1, 'b': 2})

    def test_prepare_client_do_not_match(self):
        if False:
            return 10
        product_3 = Product()
        engagement_3 = Engagement()
        engagement_3.product = product_3
        test_3 = Test()
        test_3.engagement = engagement_3
        api_scan_configuration_3 = Product_API_Scan_Configuration()
        api_scan_configuration_3.product = self.product
        test_3.api_scan_configuration = api_scan_configuration_3
        with self.assertRaisesRegex(ValidationError, 'API Scan Configuration for Edgescan and Product do not match. Product: "" \\(None\\), config\\.product: "Product" \\(None\\)'):
            edgescan_importer = EdgescanImporter()
            edgescan_importer.prepare_client(test_3)

    @patch('dojo.models.Product_API_Scan_Configuration.objects')
    def test_prepare_client_more_than_one_configuration(self, mock_foo):
        if False:
            print('Hello World!')
        mock_foo.filter.return_value = mock_foo
        mock_foo.count.return_value = 2
        with self.assertRaisesRegex(ValidationError, 'More than one Product API Scan Configuration has been configured, but none of them has been chosen\\.\\\\nPlease specify at Test which one should be used\\. Product: "Product" \\(None\\)'):
            edgescan_importer = EdgescanImporter()
            edgescan_importer.prepare_client(self.test)
            mock_foo.filter.assert_called_with(product=self.product)

    @patch('dojo.models.Product_API_Scan_Configuration.objects')
    def test_prepare_client_no_configuration(self, mock_foo):
        if False:
            i = 10
            return i + 15
        mock_foo.filter.return_value = mock_foo
        mock_foo.count.return_value = 0
        with self.assertRaisesRegex(ValidationError, 'There are no API Scan Configurations for this Product\\.\\\\nPlease add at least one API Scan Configuration for Edgescan to this Product\\. Product: "Product" \\(None\\)'):
            edgescan_importer = EdgescanImporter()
            edgescan_importer.prepare_client(self.test)
            mock_foo.filter.assert_called_with(product=self.product)

    @patch('dojo.models.Product_API_Scan_Configuration.objects')
    def test_prepare_client_one_product_configuration(self, mock_foo):
        if False:
            return 10
        mock_foo.filter.return_value = mock_foo
        mock_foo.count.return_value = 1
        mock_foo.first.return_value = self.api_scan_configuration
        edgescan_importer = EdgescanImporter()
        (edgescan_api, api_scan_configuration) = edgescan_importer.prepare_client(self.test)
        mock_foo.filter.assert_called_with(product=self.product)
        self.assertEqual(api_scan_configuration, self.api_scan_configuration)
        self.assertEqual(edgescan_api.api_key, 'API_KEY')

    def test_prepare_client_one_test_configuration(self):
        if False:
            while True:
                i = 10
        edgescan_importer = EdgescanImporter()
        (edgescan_api, api_scan_configuration) = edgescan_importer.prepare_client(self.test_2)
        self.assertEqual(api_scan_configuration, self.api_scan_configuration_2)
        self.assertEqual(edgescan_api.api_key, 'API_KEY')

    @patch('dojo.tools.api_edgescan.importer.EdgescanAPI.get_findings')
    def test_get_findings(self, mock_foo):
        if False:
            return 10
        mock_foo.return_value = self.findings
        edgescan_importer = EdgescanImporter()
        my_findings = edgescan_importer.get_findings(self.test_2)
        mock_foo.assert_called_with('SERVICE_KEY_1')
        self.assertEqual(my_findings, self.findings)