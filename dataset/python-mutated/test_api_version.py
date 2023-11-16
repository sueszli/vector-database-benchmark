"""Tests for api_version."""
from django.urls import reverse
from InvenTree.api_version import INVENTREE_API_VERSION
from InvenTree.unit_test import InvenTreeAPITestCase
from InvenTree.version import inventreeApiText, parse_version_text

class ApiVersionTests(InvenTreeAPITestCase):
    """Tests for api_version functions and APIs."""

    def test_api(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the API text is correct.'
        url = reverse('api-version-text')
        response = self.client.get(url, format='json')
        data = response.json()
        self.assertEqual(len(data), 10)

    def test_inventree_api_text(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the inventreeApiText function works expected.'
        resp = inventreeApiText()
        self.assertEqual(len(resp), 10)
        resp = inventreeApiText(20)
        self.assertEqual(len(resp), 20)
        resp = inventreeApiText(start_version=5)
        self.assertEqual(list(resp)[0], 'v5')

    def test_parse_version_text(self):
        if False:
            print('Hello World!')
        'Test that api version text is correctly parsed.'
        resp = parse_version_text()
        self.assertEqual(len(resp), INVENTREE_API_VERSION - 1)