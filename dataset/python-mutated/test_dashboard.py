from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from wagtail.admin.views.home import RecentEditsPanel
from wagtail.coreutils import get_dummy_request
from wagtail.models import Page
from wagtail.test.testapp.models import SimplePage
from wagtail.test.utils import WagtailTestUtils

class TestRecentEditsPanel(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.root_page = Page.objects.get(id=2)
        child_page = SimplePage(title='Hello world!', slug='hello-world', content='Some content here')
        self.root_page.add_child(instance=child_page)
        self.revision = child_page.save_revision()
        self.revision.publish()
        self.child_page = SimplePage.objects.get(id=child_page.id)
        self.user_alice = self.create_superuser(username='alice', password='password')
        self.create_superuser(username='bob', password='password')

    def change_something(self, title):
        if False:
            i = 10
            return i + 15
        post_data = {'title': title, 'content': 'Some content', 'slug': 'hello-world'}
        response = self.client.post(reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)), post_data)
        self.assertRedirects(response, reverse('wagtailadmin_pages:edit', args=(self.child_page.id,)))
        child_page_new = SimplePage.objects.get(id=self.child_page.id)
        self.assertTrue(child_page_new.has_unpublished_changes)

    def go_to_dashboard_response(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_home'))
        self.assertEqual(response.status_code, 200)
        return response

    def test_your_recent_edits(self):
        if False:
            i = 10
            return i + 15
        self.login(username='bob', password='password')
        response = self.client.get(reverse('wagtailadmin_home'))
        self.assertNotIn('Your most recent edits', response.content.decode('utf-8'))
        self.client.logout()
        self.login(username='alice', password='password')
        self.change_something("Alice's edit")
        response = self.go_to_dashboard_response()
        self.assertIn('Your most recent edits', response.content.decode('utf-8'))
        self.login(username='bob', password='password')
        self.change_something("Bob's edit")
        response = self.go_to_dashboard_response()
        self.assertIn('Your most recent edits', response.content.decode('utf-8'))
        self.client.logout()
        self.login(username='alice', password='password')
        response = self.go_to_dashboard_response()
        self.assertIn('Your most recent edits', response.content.decode('utf-8'))

    def test_missing_page_record(self):
        if False:
            while True:
                i = 10
        self.revision.user = self.user_alice
        self.revision.object_id = '999999'
        self.revision.save()
        self.login(username='alice', password='password')
        response = self.client.get(reverse('wagtailadmin_home'))
        self.assertEqual(response.status_code, 200)

    def test_panel(self):
        if False:
            print('Hello World!')
        'Test if the panel actually returns expected pages'
        self.login(username='bob', password='password')
        self.change_something("Bob's edit")
        self.client.user = get_user_model().objects.get(email='bob@example.com')
        panel = RecentEditsPanel()
        ctx = panel.get_context_data({'request': self.client})
        page = Page.objects.get(pk=self.child_page.id).specific
        self.assertEqual(ctx['last_edits'][0][0].content_object, page)
        self.assertEqual(ctx['last_edits'][0][1], page)

class TestRecentEditsQueryCount(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.bob = self.create_superuser(username='bob', password='password')
        self.dummy_request = get_dummy_request()
        self.dummy_request.user = self.bob
        pages_to_edit = Page.objects.filter(id__in=[4, 5, 6, 9, 12, 13]).specific()
        for page in pages_to_edit:
            page.save_revision(user=self.bob)

    def test_panel_query_count(self):
        if False:
            return 10
        self.client.user = self.bob
        with self.assertNumQueries(4):
            panel = RecentEditsPanel()
            parent_context = {'request': self.dummy_request}
            panel.get_context_data(parent_context)
        html = panel.render_html(parent_context)
        self.assertIn('Ameristralia Day', html)