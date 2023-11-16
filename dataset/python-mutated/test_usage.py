from io import StringIO
from django.contrib.admin.utils import quote
from django.contrib.auth.models import Permission
from django.core import management
from django.test import TestCase
from django.urls import reverse
from wagtail.models import Page, ReferenceIndex
from wagtail.test.testapp.models import Advert, DraftStateModel, EventPage, GenericSnippetPage
from wagtail.test.utils import WagtailTestUtils

class TestUsageCount(TestCase):
    fixtures = ['test.json']

    @classmethod
    def setUpTestData(cls):
        if False:
            while True:
                i = 10
        super().setUpTestData()
        output = StringIO()
        management.call_command('rebuild_references_index', stdout=output)

    def test_snippet_usage_count(self):
        if False:
            print('Hello World!')
        advert = Advert.objects.get(pk=1)
        self.assertEqual(ReferenceIndex.get_grouped_references_to(advert).count(), 2)

class TestUsedBy(TestCase):
    fixtures = ['test.json']

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        super().setUpTestData()
        output = StringIO()
        management.call_command('rebuild_references_index', stdout=output)

    def test_snippet_used_by(self):
        if False:
            for i in range(10):
                print('nop')
        advert = Advert.objects.get(pk=1)
        usage = ReferenceIndex.get_grouped_references_to(advert)
        self.assertIsInstance(usage[0], tuple)
        self.assertIsInstance(usage[0][0], Page)
        self.assertIsInstance(usage[0][1], list)
        self.assertIsInstance(usage[0][1][0], ReferenceIndex)

class TestSnippetUsageView(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.user = self.login()

    def test_use_latest_draft_as_title(self):
        if False:
            for i in range(10):
                print('nop')
        snippet = DraftStateModel.objects.create(text='Draft-enabled Foo, Published')
        snippet.save_revision().publish()
        snippet.text = 'Draft-enabled Bar, In Draft'
        snippet.save_revision()
        response = self.client.get(reverse('wagtailsnippets_tests_draftstatemodel:usage', args=[quote(snippet.pk)]))
        self.assertContains(response, '<span class="w-header__subtitle">Draft-enabled Bar, In Draft</span>')

    def test_usage(self):
        if False:
            while True:
                i = 10
        page = Page.objects.get(pk=2)
        page.save()
        gfk_page = GenericSnippetPage(title='Foobar Title', snippet_content_object=Advert.objects.get(pk=1))
        page.add_child(instance=gfk_page)
        response = self.client.get(reverse('wagtailsnippets_tests_advert:usage', args=['1']))
        self.assertContains(response, 'Welcome to the Wagtail test site!')
        self.assertContains(response, 'Foobar Title')
        self.assertContains(response, '<td>Generic snippet page</td>', html=True)
        self.assertContains(response, 'Snippet content object')
        self.assertContains(response, '<th>Field</th>', html=True)
        self.assertNotContains(response, '<th>If you confirm deletion</th>', html=True)
        self.assertContains(response, 'Snippet content object')

    def test_usage_without_edit_permission_on_snippet(self):
        if False:
            print('Hello World!')
        user = self.create_user(username='basicadmin', email='basicadmin@example.com', password='password')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        user.user_permissions.add(admin_permission)
        self.login(username='basicadmin', password='password')
        response = self.client.get(reverse('wagtailsnippets_tests_advert:usage', args=['1']))
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('wagtailadmin_home'))

    def test_usage_without_edit_permission_on_page(self):
        if False:
            while True:
                i = 10
        page = Page.objects.get(pk=2)
        page.save()
        user = self.create_user(username='basicadmin', email='basicadmin@example.com', password='password')
        admin_permission = Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin')
        advert_permission = Permission.objects.get(content_type__app_label='tests', codename='change_advert')
        user.user_permissions.add(admin_permission)
        user.user_permissions.add(advert_permission)
        self.login(username='basicadmin', password='password')
        response = self.client.get(reverse('wagtailsnippets_tests_advert:usage', args=['1']))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'Welcome to the Wagtail test site!')
        self.assertContains(response, '(Private page)')
        self.assertContains(response, '<td>Page</td>', html=True)
        self.assertContains(response, '<th>Field</th>', html=True)
        self.assertNotContains(response, '<th>If you confirm deletion</th>', html=True)
        self.assertContains(response, '<li>Advert</li>', html=True)

    def test_usage_with_describe_on_delete_cascade(self):
        if False:
            print('Hello World!')
        page = Page.objects.get(pk=2)
        page.save()
        response = self.client.get(reverse('wagtailsnippets_tests_advert:usage', args=['1']) + '?describe_on_delete=1')
        self.assertContains(response, 'Welcome to the Wagtail test site!')
        self.assertContains(response, '<td>Page</td>', html=True)
        self.assertNotContains(response, '<th>Field</th>', html=True)
        self.assertContains(response, '<th>If you confirm deletion</th>', html=True)
        self.assertContains(response, 'Advert')
        self.assertContains(response, ': the advert placement will also be deleted')

    def test_usage_with_describe_on_delete_set_null(self):
        if False:
            return 10
        page = EventPage.objects.first()
        page.save()
        self.assertEqual(page.feed_image.get_usage().count(), 1)
        response = self.client.get(reverse('wagtailimages:image_usage', args=[page.feed_image.id]) + '?describe_on_delete=1')
        self.assertContains(response, page.title)
        self.assertContains(response, '<td>Event page</td>', html=True)
        self.assertNotContains(response, '<th>Field</th>', html=True)
        self.assertContains(response, '<th>If you confirm deletion</th>', html=True)
        self.assertContains(response, 'Feed image')
        self.assertContains(response, ': will unset the reference')

    def test_usage_with_describe_on_delete_gfk(self):
        if False:
            print('Hello World!')
        advert = Advert.objects.get(pk=1)
        gfk_page = GenericSnippetPage(title='Foobar Title', snippet_content_object=advert)
        Page.objects.get(pk=1).add_child(instance=gfk_page)
        self.assertEqual(ReferenceIndex.get_grouped_references_to(advert).count(), 1)
        response = self.client.get(reverse('wagtailsnippets_tests_advert:usage', args=['1']) + '?describe_on_delete=1')
        self.assertNotContains(response, 'Welcome to the Wagtail test site!')
        self.assertContains(response, 'Foobar Title')
        self.assertContains(response, '<td>Generic snippet page</td>', html=True)
        self.assertNotContains(response, '<th>Field</th>', html=True)
        self.assertContains(response, '<th>If you confirm deletion</th>', html=True)
        self.assertContains(response, 'Snippet content object')
        self.assertContains(response, ': will unset the reference')