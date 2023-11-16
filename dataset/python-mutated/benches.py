from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from wagtail.models import Page, Site
from wagtail.test.benchmark import Benchmark
from wagtail.test.testapp.models import SingleEventPage, StreamPage
from wagtail.test.utils import WagtailTestUtils

class BenchPageExplorerWith50LargePages(Benchmark, WagtailTestUtils, TestCase):
    """
    Creates 50 pages with large body content and benches the explorer.
    This will be slow if the body content is being loaded from the database.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.root_page = Page.objects.get(id=1)
        Site.objects.create(is_default_site=True, root_page=self.root_page)
        body = '[' + ','.join(['{"type": "text", "value": "%s"}' % ('foo' * 2000)] * 100) + ']'
        for i in range(50):
            self.root_page.add_child(instance=StreamPage(title=f'Page {i + 1}', slug=str(i + 1), body=body))
        self.login()

    def bench(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Page 1')
        self.assertContains(response, 'Page 49')

class BenchPageExplorerWithCustomURLPages(Benchmark, WagtailTestUtils, TestCase):
    """
    Creates 50 pages of a class with a customised the .url property.
    This will check how long it takes to generate URLs for all of these
    pages.
    """

    def setUp(self):
        if False:
            return 10
        self.root_page = Page.objects.get(id=1)
        Site.objects.create(is_default_site=True, root_page=self.root_page)
        for i in range(50):
            self.root_page.add_child(instance=SingleEventPage(title=f'Event {i + 1}', slug=str(i + 1), date_from=timezone.now(), audience='public', location='reykjavik', cost='cost'))
        self.login()

    def bench(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Event 1')
        self.assertContains(response, 'Event 49')
        self.assertContains(response, 'a href="http:///49/pointless-suffix/"')