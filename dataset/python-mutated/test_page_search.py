from io import StringIO
from django.contrib.auth.models import Permission
from django.core import management
from django.test import TransactionTestCase
from django.urls import reverse
from wagtail.models import Page
from wagtail.test.testapp.models import SimplePage, SingleEventPage
from wagtail.test.utils import WagtailTestUtils
from wagtail.test.utils.timestamps import local_datetime

class TestPageSearch(WagtailTestUtils, TransactionTestCase):
    fixtures = ['test_empty.json']

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        management.call_command('update_index', backend_name='default', stdout=StringIO(), chunk_size=50)
        self.user = self.login()

    def get(self, params=None, url_name='wagtailadmin_pages:search', **extra):
        if False:
            print('Hello World!')
        return self.client.get(reverse(url_name), params or {}, **extra)

    def test_view(self):
        if False:
            print('Hello World!')
        response = self.get()
        self.assertTemplateUsed(response, 'wagtailadmin/pages/search.html')
        self.assertEqual(response.status_code, 200)

    def test_search(self):
        if False:
            i = 10
            return i + 15
        response = self.get({'q': 'Hello'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/search.html')
        self.assertEqual(response.context['query_string'], 'Hello')

    def test_search_searchable_fields(self):
        if False:
            return 10
        root_page = Page.objects.get(id=2)
        root_page.add_child(instance=SimplePage(title='Greetings!', slug='hello', content='good morning', live=True, has_unpublished_changes=False))
        response = self.get({'q': 'hello'})
        self.assertNotContains(response, 'There is one matching page')
        response = self.get({'q': 'greetings'})
        self.assertContains(response, 'There is one matching page')

    def test_ajax(self):
        if False:
            return 10
        response = self.get({'q': 'Hello'}, url_name='wagtailadmin_pages:search_results')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateNotUsed(response, 'wagtailadmin/pages/search.html')
        self.assertTemplateUsed(response, 'wagtailadmin/pages/search_results.html')
        self.assertEqual(response.context['query_string'], 'Hello')

    def test_pagination(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get({'q': 'Hello', 'p': 1})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/search.html')
        response = self.get({'q': 'Hello', 'p': 9999})
        self.assertEqual(response.status_code, 404)

    def test_root_can_appear_in_search_results(self):
        if False:
            print('Hello World!')
        response = self.get({'q': 'root'})
        self.assertEqual(response.status_code, 200)
        results = response.context['pages']
        self.assertTrue(any((r.slug == 'root' for r in results)))

    def test_search_uses_admin_display_title_from_specific_class(self):
        if False:
            while True:
                i = 10
        root_page = Page.objects.get(id=2)
        new_event = SingleEventPage(title='Lunar event', location='the moon', audience='public', cost='free', date_from='2001-01-01', latest_revision_created_at=local_datetime(2016, 1, 1))
        root_page.add_child(instance=new_event)
        response = self.get({'q': 'lunar'})
        self.assertContains(response, 'Lunar event (single event)')

    def test_search_no_perms(self):
        if False:
            while True:
                i = 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        self.assertRedirects(self.get(), '/admin/')

    def test_search_order_by_title(self):
        if False:
            i = 10
            return i + 15
        root_page = Page.objects.get(id=2)
        new_event = SingleEventPage(title='Lunar event', location='the moon', audience='public', cost='free', date_from='2001-01-01', latest_revision_created_at=local_datetime(2016, 1, 1))
        root_page.add_child(instance=new_event)
        new_event_2 = SingleEventPage(title='A Lunar event', location='the moon', audience='public', cost='free', date_from='2001-01-01', latest_revision_created_at=local_datetime(2016, 1, 1))
        root_page.add_child(instance=new_event_2)
        response = self.get({'q': 'Lunar', 'ordering': 'title'})
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [new_event_2.id, new_event.id])
        response = self.get({'q': 'Lunar', 'ordering': '-title'})
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [new_event.id, new_event_2.id])

    def test_search_order_by_updated(self):
        if False:
            for i in range(10):
                print('nop')
        root_page = Page.objects.get(id=2)
        new_event = SingleEventPage(title='Lunar event', location='the moon', audience='public', cost='free', date_from='2001-01-01', latest_revision_created_at=local_datetime(2016, 1, 1))
        root_page.add_child(instance=new_event)
        new_event_2 = SingleEventPage(title='Lunar event 2', location='the moon', audience='public', cost='free', date_from='2001-01-01', latest_revision_created_at=local_datetime(2015, 1, 1))
        root_page.add_child(instance=new_event_2)
        response = self.get({'q': 'Lunar', 'ordering': 'latest_revision_created_at'})
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [new_event_2.id, new_event.id])
        response = self.get({'q': 'Lunar', 'ordering': '-latest_revision_created_at'})
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [new_event.id, new_event_2.id])

    def test_search_order_by_status(self):
        if False:
            return 10
        root_page = Page.objects.get(id=2)
        live_event = SingleEventPage(title='Lunar event', location='the moon', audience='public', cost='free', date_from='2001-01-01', latest_revision_created_at=local_datetime(2016, 1, 1), live=True)
        root_page.add_child(instance=live_event)
        draft_event = SingleEventPage(title='Lunar event', location='the moon', audience='public', cost='free', date_from='2001-01-01', latest_revision_created_at=local_datetime(2016, 1, 1), live=False)
        root_page.add_child(instance=draft_event)
        response = self.get({'q': 'Lunar', 'ordering': 'live'})
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [draft_event.id, live_event.id])
        response = self.get({'q': 'Lunar', 'ordering': '-live'})
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [live_event.id, draft_event.id])

    def test_search_filter_content_type(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get({'content_type': 'demosite.standardpage'})
        self.assertEqual(response.status_code, 200)
        response = self.get({'content_type': 'demosite.standardpage.error'})
        self.assertEqual(response.status_code, 404)