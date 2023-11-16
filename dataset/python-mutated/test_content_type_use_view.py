from django.test import TestCase
from django.urls import reverse
from django.utils.http import urlencode
from wagtail.test.testapp.models import EventPage
from wagtail.test.utils import WagtailTestUtils

class TestContentTypeUse(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            return 10
        self.user = self.login()
        self.christmas_page = EventPage.objects.get(title='Christmas')

    def test_content_type_use(self):
        if False:
            i = 10
            return i + 15
        request_url = reverse('wagtailadmin_pages:type_use', args=('tests', 'eventpage'))
        response = self.client.get(request_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/listing.html')
        self.assertTemplateUsed(response, 'wagtailadmin/pages/usage_results.html')
        self.assertContains(response, 'Christmas')
        delete_url = reverse('wagtailadmin_pages:delete', args=(self.christmas_page.id,)) + '?' + urlencode({'next': request_url})
        self.assertContains(response, delete_url)
        self.assertNotContains(response, 'data-bulk-action-select-all-checkbox')