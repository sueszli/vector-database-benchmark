from django.test import TestCase
from django.urls import reverse
from wagtail.test.utils import WagtailTestUtils

class TestBulkActionDispatcher(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.user = self.login()

    def test_bulk_action_invalid_action(self):
        if False:
            i = 10
            return i + 15
        url = reverse('wagtail_bulk_action', args=('wagtailcore', 'page', 'ships'))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)