from django.contrib.auth.models import AbstractBaseUser, Group, Permission
from django.contrib.contenttypes.models import ContentType
from django.core import paginator
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils.http import urlencode
from wagtail import hooks
from wagtail.admin.widgets import Button
from wagtail.models import GroupPagePermission, Locale, Page, Workflow
from wagtail.test.testapp.models import SimplePage, SingleEventPage, StandardIndex
from wagtail.test.utils import WagtailTestUtils
from wagtail.test.utils.timestamps import local_datetime
from wagtail.utils.deprecation import RemovedInWagtail70Warning

class TestPageExplorer(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.root_page = Page.objects.get(id=2)
        self.child_page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=self.child_page)
        self.old_page = StandardIndex(title='Old page', slug='old-page', latest_revision_created_at=local_datetime(2010, 1, 1))
        self.root_page.add_child(instance=self.old_page)
        self.new_page = SimplePage(title='New page', slug='new-page', content='hello', latest_revision_created_at=local_datetime(2016, 1, 1))
        self.root_page.add_child(instance=self.new_page)
        self.user = self.login()

    def test_explore(self):
        if False:
            while True:
                i = 10
        explore_url = reverse('wagtailadmin_explore', args=(self.root_page.id,))
        response = self.client.get(explore_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertEqual(self.root_page, response.context['parent_page'])
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.new_page.id, self.old_page.id, self.child_page.id])
        expected_new_page_copy_url = reverse('wagtailadmin_pages:copy', args=(self.new_page.id,)) + '?' + urlencode({'next': explore_url})
        self.assertContains(response, f'href="{expected_new_page_copy_url}"')
        self.assertContains(response, '1-3 of 3')

    def test_explore_results(self):
        if False:
            return 10
        explore_results_url = reverse('wagtailadmin_explore_results', args=(self.root_page.id,))
        response = self.client.get(explore_results_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index_results.html')
        self.assertEqual(self.root_page, response.context['parent_page'])
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.new_page.id, self.old_page.id, self.child_page.id])
        explore_url = reverse('wagtailadmin_explore', args=(self.root_page.id,))
        expected_new_page_copy_url = reverse('wagtailadmin_pages:copy', args=(self.new_page.id,)) + '?' + urlencode({'next': explore_url})
        self.assertContains(response, f'href="{expected_new_page_copy_url}"')
        self.assertContains(response, '1-3 of 3')

    def test_explore_root(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_explore_root'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertEqual(Page.objects.get(id=1), response.context['parent_page'])
        self.assertIn(self.root_page, response.context['pages'])

    def test_explore_root_shows_icon(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_explore_root'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<a href="/admin/sites/" title="Sites menu"><svg')

    def test_ordering(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'ordering': 'title'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertEqual(response.context['ordering'], 'title')
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.child_page.id, self.new_page.id, self.old_page.id])

    def test_reverse_ordering(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'ordering': '-title'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertEqual(response.context['ordering'], '-title')
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.old_page.id, self.new_page.id, self.child_page.id])

    def test_ordering_by_last_revision_forward(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'ordering': 'latest_revision_created_at'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertEqual(response.context['ordering'], 'latest_revision_created_at')
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.child_page.id, self.old_page.id, self.new_page.id])

    def test_invalid_ordering(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'ordering': 'invalid_order'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertEqual(response.context['ordering'], '-latest_revision_created_at')

    def test_reordering(self):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'ordering': 'ord'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertEqual(response.context['ordering'], 'ord')
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.child_page.id, self.old_page.id, self.new_page.id])
        self.assertNotIsInstance(response.context['pages'], paginator.Page)

    def test_construct_explorer_page_queryset_hook(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'polite_pages_only': 'yes_please'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.child_page.id])

    def test_construct_explorer_page_queryset_hook_with_ordering(self):
        if False:
            return 10

        def set_custom_ordering(parent_page, pages, request):
            if False:
                return 10
            return pages.order_by('-title')
        with hooks.register_temporarily('construct_explorer_page_queryset', set_custom_ordering):
            response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.old_page.id, self.new_page.id, self.child_page.id])

    def test_construct_page_listing_buttons_hook_with_old_signature(self):
        if False:
            while True:
                i = 10

        def add_dummy_button(buttons, page, page_perms, context=None):
            if False:
                while True:
                    i = 10
            item = Button(label='Dummy Button', url='/dummy-button', priority=10)
            buttons.append(item)
        with hooks.register_temporarily('construct_page_listing_buttons', add_dummy_button):
            with self.assertWarnsMessage(RemovedInWagtail70Warning, '`construct_page_listing_buttons` hook functions should accept a `user` argument instead of `page_perms`'):
                response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertContains(response, 'Dummy Button')
        self.assertContains(response, '/dummy-button')

    def test_construct_page_listing_buttons_hook_with_new_signature(self):
        if False:
            while True:
                i = 10

        def add_dummy_button(buttons, page, user, context=None):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(user, AbstractBaseUser):
                raise TypeError('expected a user instance')
            item = Button(label='Dummy Button', url='/dummy-button', priority=10)
            buttons.append(item)
        with hooks.register_temporarily('construct_page_listing_buttons', add_dummy_button):
            response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertContains(response, 'Dummy Button')
        self.assertContains(response, '/dummy-button')

    def make_pages(self):
        if False:
            while True:
                i = 10
        for i in range(150):
            self.root_page.add_child(instance=SimplePage(title='Page ' + str(i), slug='page-' + str(i), content='hello'))

    def test_pagination(self):
        if False:
            i = 10
            return i + 15
        self.make_pages()
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'p': 2})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        self.assertEqual(response.context['page_obj'].number, 2)
        self.assertContains(response, '51-100 of 153')

    def test_pagination_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        self.make_pages()
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'p': 'Hello World!'})
        self.assertEqual(response.status_code, 404)

    def test_pagination_out_of_range(self):
        if False:
            while True:
                i = 10
        self.make_pages()
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'p': 99999})
        self.assertEqual(response.status_code, 404)

    @override_settings(USE_L10N=True, USE_THOUSAND_SEPARATOR=True)
    def test_no_thousand_separators_in_bulk_action_checkbox(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the USE_THOUSAND_SEPARATOR setting does mess up object IDs in\n        bulk actions checkboxes\n        '
        self.root_page.add_child(instance=SimplePage(pk=1000, title='Page 1000', slug='page-1000', content='hello'))
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        expected = 'data-object-id="1000"'
        self.assertContains(response, expected)

    def test_listing_uses_specific_models(self):
        if False:
            while True:
                i = 10
        self.new_event = SingleEventPage(title='New event', location='the moon', audience='public', cost='free', date_from='2001-01-01', latest_revision_created_at=local_datetime(2016, 1, 1))
        self.root_page.add_child(instance=self.new_event)
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '/new-event/pointless-suffix/')

    def make_event_pages(self, count):
        if False:
            i = 10
            return i + 15
        for i in range(count):
            self.root_page.add_child(instance=SingleEventPage(title='New event ' + str(i), location='the moon', audience='public', cost='free', date_from='2001-01-01', latest_revision_created_at=local_datetime(2016, 1, 1)))

    def test_exploring_uses_specific_page_with_custom_display_title(self):
        if False:
            while True:
                i = 10
        self.make_event_pages(count=1)
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertContains(response, 'New event 0 (single event)')
        new_event = SingleEventPage.objects.latest('pk')
        response = self.client.get(reverse('wagtailadmin_explore', args=(new_event.id,)))
        self.assertContains(response, 'New event 0 (single event)')

    def test_parent_page_is_specific(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.child_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.context['parent_page'], SimplePage)

    def test_explorer_no_perms(self):
        if False:
            return 10
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        admin = reverse('wagtailadmin_home')
        self.assertRedirects(self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,))), admin)
        self.assertRedirects(self.client.get(reverse('wagtailadmin_explore_root')), admin)

    def test_explore_with_missing_page_model(self):
        if False:
            print('Hello World!')
        missing_page_content_type = ContentType.objects.create(app_label='tests', model='missingpage')
        Page.objects.filter(id=self.old_page.id).update(content_type=missing_page_content_type)
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.old_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')

    def test_search(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.root_page.id,)), {'q': 'old'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index.html')
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.old_page.id])
        self.assertContains(response, 'Search the whole site')

    def test_search_results(self):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('wagtailadmin_explore_results', args=(self.root_page.id,)), {'q': 'old'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/pages/index_results.html')
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.old_page.id])
        self.assertContains(response, '1-1 of 1')

    def test_search_searches_descendants(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_explore_root'), {'q': 'old'})
        self.assertEqual(response.status_code, 200)
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.old_page.id])
        self.assertContains(response, '<a href="/admin/pages/2/"><svg class="icon icon-arrow-right default" aria-hidden="true"><use href="#icon-arrow-right"></use></svg>Welcome to your new Wagtail site!</a>', html=True)
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.new_page.id,)), {'q': 'old'})
        self.assertEqual(response.status_code, 200)
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [])

    def test_search_whole_tree(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.new_page.id,)), {'q': 'old', 'search_all': '1'})
        self.assertEqual(response.status_code, 200)
        page_ids = [page.id for page in response.context['pages']]
        self.assertEqual(page_ids, [self.old_page.id])
        self.assertContains(response, "Search within 'New page (simple page)'")

class TestBreadcrumb(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def test_breadcrumb_next_present(self):
        if False:
            while True:
                i = 10
        self.user = self.login()
        page = Page.objects.get(url_path='/home/secret-plans/steal-underpants/')
        response = self.client.get(reverse('wagtailadmin_explore', args=(page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-controller="w-breadcrumbs"')

    def test_breadcrumb_uses_specific_titles(self):
        if False:
            for i in range(10):
                print('nop')
        self.user = self.login()
        page = Page.objects.get(url_path='/home/secret-plans/steal-underpants/')
        response = self.client.get(reverse('wagtailadmin_explore', args=(page.id,)))
        expected_url = reverse('wagtailadmin_explore', args=(Page.objects.get(url_path='/home/secret-plans/').id,))
        expected = '\n            <li class="w-h-full w-flex w-items-center w-overflow-hidden w-transition w-duration-300 w-whitespace-nowrap w-flex-shrink-0 w-max-w-0" data-w-breadcrumbs-target="content" hidden>\n                <a class="w-flex w-items-center w-h-full w-text-text-label w-pr-0.5 w-text-14 w-no-underline w-outline-offset-inside hover:w-underline hover:w-text-text-label w-h-full" href="%s">\n                    Secret plans (simple page)\n                </a>\n                <svg class="icon icon-arrow-right w-w-4 w-h-4 w-ml-3" aria-hidden="true">\n                    <use href="#icon-arrow-right"></use>\n                </svg>\n            </li>\n        ' % expected_url
        self.assertContains(response, expected, html=True)

class TestPageExplorerSidePanel(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def test_side_panel_present(self):
        if False:
            return 10
        self.user = self.login()
        page = Page.objects.get(url_path='/home/secret-plans/steal-underpants/')
        response = self.client.get(reverse('wagtailadmin_explore', args=(page.id,)))
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertTagInHTML('<aside data-form-side data-form-side-explorer>', html, allow_extra_attrs=True)

class TestPageExplorerSignposting(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            return 10
        self.root_page = Page.objects.get(id=1)
        self.site_page = Page.objects.get(id=2)
        self.no_site_page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=self.no_site_page)

    def test_admin_at_root(self):
        if False:
            print('Hello World!')
        self.login(username='superuser', password='password')
        response = self.client.get(reverse('wagtailadmin_explore_root'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'The root level is where you can add new sites to your Wagtail installation. Pages created here will not be accessible at any URL until they are associated with a site.')
        self.assertContains(response, '<a href="/admin/sites/">Configure a site now.</a>')

    def test_admin_at_non_site_page(self):
        if False:
            return 10
        self.login(username='superuser', password='password')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.no_site_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'There is no site set up for this location. Pages created here will not be accessible at any URL until a site is associated with this location.')
        self.assertContains(response, '<a href="/admin/sites/">Configure a site now.</a>')

    def test_admin_at_site_page(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='superuser', password='password')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.site_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'Pages created here will not be accessible')

    def test_nonadmin_at_root(self):
        if False:
            i = 10
            return i + 15
        GroupPagePermission.objects.create(group=Group.objects.get(name='Site-wide editors'), page=self.no_site_page, permission_type='add')
        self.login(username='siteeditor', password='password')
        response = self.client.get(reverse('wagtailadmin_explore_root'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Pages created here will not be accessible at any URL. To add pages to an existing site, create them as children of the homepage.')

    def test_nonadmin_at_non_site_page(self):
        if False:
            return 10
        GroupPagePermission.objects.create(group=Group.objects.get(name='Site-wide editors'), page=self.no_site_page, permission_type='add')
        self.login(username='siteeditor', password='password')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.no_site_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'There is no site record for this location. Pages created here will not be accessible at any URL.')

    def test_nonadmin_at_site_page(self):
        if False:
            i = 10
            return i + 15
        self.login(username='siteeditor', password='password')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.site_page.id,)))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'Pages created here will not be accessible')

    def test_bad_permissions_at_root(self):
        if False:
            return 10
        self.login(username='siteeditor', password='password')
        response = self.client.get(reverse('wagtailadmin_explore_root'))
        self.assertEqual((response.status_code, response['Location']), (302, reverse('wagtailadmin_explore', args=(self.site_page.pk,))))

    def test_bad_permissions_at_non_site_page(self):
        if False:
            return 10
        self.login(username='siteeditor', password='password')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.no_site_page.id,)))
        self.assertEqual((response.status_code, response['Location']), (302, reverse('wagtailadmin_explore', args=(self.site_page.pk,))))

    def test_bad_permissions_at_site_page(self):
        if False:
            print('Hello World!')
        Group.objects.get(name='Site-wide editors').page_permissions.update(page_id=self.no_site_page.id)
        self.login(username='siteeditor', password='password')
        response = self.client.get(reverse('wagtailadmin_explore', args=(self.site_page.id,)))
        self.assertEqual((response.status_code, response['Location']), (302, reverse('wagtailadmin_explore', args=(self.no_site_page.pk,))))

class TestExplorablePageVisibility(WagtailTestUtils, TestCase):
    """
    Test the way that the Explorable Pages functionality manifests within the Explorer.
    This is isolated in its own test case because it requires a custom page tree and custom set of
    users and groups.
    The fixture sets up this page tree:
    ========================================================
    ID Site          Path
    ========================================================
    1              /
    2  testserver  /home/
    3  testserver  /home/about-us/
    4  example.com /example-home/
    5  example.com /example-home/content/
    6  example.com /example-home/content/page-1/
    7  example.com /example-home/content/page-2/
    9  example.com /example-home/content/page-2/child-1
    8  example.com /example-home/other-content/
    10 example2.com /home-2/
    ========================================================
    Group 1 has explore and choose permissions rooted at testserver's homepage.
    Group 2 has explore and choose permissions rooted at example.com's page-1.
    Group 3 has explore and choose permissions rooted at example.com's other-content.
    User "jane" is in Group 1.
    User "bob" is in Group 2.
    User "sam" is in Groups 1 and 2.
    User "josh" is in Groups 2 and 3.
    User "mary" is is no Groups, but she has the "access wagtail admin" permission.
    User "superman" is an admin.
    """
    fixtures = ['test_explorable_pages.json']

    def test_admin_can_explore_every_page(self):
        if False:
            while True:
                i = 10
        self.login(username='superman', password='password')
        for page in Page.objects.all():
            response = self.client.get(reverse('wagtailadmin_explore', args=[page.pk]))
            self.assertEqual(response.status_code, 200)

    def test_admin_sees_root_page_as_explorer_root(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='superman', password='password')
        response = self.client.get(reverse('wagtailadmin_explore_root'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Welcome to testserver!')
        self.assertContains(response, 'Welcome to example.com!')

    def test_admin_sees_breadcrumbs_up_to_root_page(self):
        if False:
            return 10
        self.login(username='superman', password='password')
        response = self.client.get(reverse('wagtailadmin_explore', args=[6]))
        self.assertEqual(response.status_code, 200)
        expected = '\n            <li class="w-h-full w-flex w-items-center w-overflow-hidden w-transition w-duration-300 w-whitespace-nowrap w-flex-shrink-0 w-max-w-0" data-w-breadcrumbs-target="content" hidden>\n                <a class="w-flex w-items-center w-h-full w-text-text-label w-pr-0.5 w-text-14 w-no-underline w-outline-offset-inside hover:w-underline hover:w-text-text-label w-h-full" href="/admin/pages/">\n                    Root\n                </a>\n                <svg class="icon icon-arrow-right w-w-4 w-h-4 w-ml-3" aria-hidden="true">\n                    <use href="#icon-arrow-right"></use>\n                </svg>\n            </li>\n\n        '
        self.assertContains(response, expected, html=True)
        expected = '\n            <li class="w-h-full w-flex w-items-center w-overflow-hidden w-transition w-duration-300 w-whitespace-nowrap w-flex-shrink-0 w-max-w-0" data-w-breadcrumbs-target="content" hidden>\n                <a class="w-flex w-items-center w-h-full w-text-text-label w-pr-0.5 w-text-14 w-no-underline w-outline-offset-inside hover:w-underline hover:w-text-text-label w-h-full" href="/admin/pages/4/">\n                    Welcome to example.com!\n                </a>\n                <svg class="icon icon-arrow-right w-w-4 w-h-4 w-ml-3" aria-hidden="true">\n                    <use href="#icon-arrow-right"></use>\n                </svg>\n            </li>\n        '
        self.assertContains(response, expected, html=True)
        expected = '\n            <li class="w-h-full w-flex w-items-center w-overflow-hidden w-transition w-duration-300 w-whitespace-nowrap w-flex-shrink-0 w-max-w-0" data-w-breadcrumbs-target="content" hidden>\n                <a class="w-flex w-items-center w-h-full w-text-text-label w-pr-0.5 w-text-14 w-no-underline w-outline-offset-inside hover:w-underline hover:w-text-text-label w-h-full" href="/admin/pages/5/">\n                    Content\n                </a>\n                <svg class="icon icon-arrow-right w-w-4 w-h-4 w-ml-3" aria-hidden="true">\n                    <use href="#icon-arrow-right"></use>\n                </svg>\n            </li>\n        '
        self.assertContains(response, expected, html=True)

    def test_nonadmin_sees_breadcrumbs_up_to_cca(self):
        if False:
            while True:
                i = 10
        self.login(username='josh', password='password')
        response = self.client.get(reverse('wagtailadmin_explore', args=[6]))
        self.assertEqual(response.status_code, 200)
        expected = '\n            <li class="w-h-full w-flex w-items-center w-overflow-hidden w-transition w-duration-300 w-whitespace-nowrap w-flex-shrink-0 w-max-w-0" data-w-breadcrumbs-target="content" hidden>\n                <a class="w-flex w-items-center w-h-full w-text-text-label w-pr-0.5 w-text-14 w-no-underline w-outline-offset-inside hover:w-underline hover:w-text-text-label w-h-full" href="/admin/pages/4/">\n                    Root\n                </a>\n                <svg class="icon icon-arrow-right w-w-4 w-h-4 w-ml-3" aria-hidden="true">\n                    <use href="#icon-arrow-right"></use>\n                </svg>\n            </li>\n        '
        self.assertContains(response, expected, html=True)
        expected = '\n            <li class="w-h-full w-flex w-items-center w-overflow-hidden w-transition w-duration-300 w-whitespace-nowrap w-flex-shrink-0 w-max-w-0" data-w-breadcrumbs-target="content" hidden>\n                <a class="w-flex w-items-center w-h-full w-text-text-label w-pr-0.5 w-text-14 w-no-underline w-outline-offset-inside hover:w-underline hover:w-text-text-label w-h-full" href="/admin/pages/5/">\n                    Content\n                </a>\n                <svg class="icon icon-arrow-right w-w-4 w-h-4 w-ml-3" aria-hidden="true">\n                    <use href="#icon-arrow-right"></use>\n                </svg>\n            </li>\n        '
        self.assertContains(response, expected, html=True)
        self.assertNotContains(response, 'Welcome to example.com!')

    def test_admin_home_page_changes_with_permissions(self):
        if False:
            return 10
        self.login(username='bob', password='password')
        response = self.client.get(reverse('wagtailadmin_home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Welcome to the example.com Wagtail CMS')
        self.assertNotContains(response, 'testserver')

    def test_breadcrumb_with_no_user_permissions(self):
        if False:
            i = 10
            return i + 15
        self.login(username='mary', password='password')
        response = self.client.get(reverse('wagtailadmin_home'))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<li class="home breadcrumb-item"><a class="breadcrumb-link" href="/admin/pages/4/" class="icon icon-home text-replace">Home</a></li>')

@override_settings(WAGTAIL_I18N_ENABLED=True)
class TestLocaleSelector(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            return 10
        self.events_page = Page.objects.get(url_path='/home/events/')
        self.fr_locale = Locale.objects.create(language_code='fr')
        self.translated_events_page = self.events_page.copy_for_translation(self.fr_locale, copy_parents=True)
        self.user = self.login()

    def test_locale_selector(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_explore', args=[self.events_page.id]))
        html = response.content.decode()
        self.assertContains(response, 'id="status-sidebar-english"')
        self.assertContains(response, 'Switch locales')
        add_translation_url = reverse('wagtailadmin_explore', args=[self.translated_events_page.id])
        self.assertTagInHTML(f'<a href="{add_translation_url}" lang="fr">French</a>', html, allow_extra_attrs=True)

    @override_settings(WAGTAIL_I18N_ENABLED=False)
    def test_locale_selector_not_present_when_i18n_disabled(self):
        if False:
            return 10
        response = self.client.get(reverse('wagtailadmin_explore', args=[self.events_page.id]))
        html = response.content.decode()
        self.assertNotContains(response, 'Switch locales')
        add_translation_url = reverse('wagtailadmin_explore', args=[self.translated_events_page.id])
        self.assertTagInHTML(f'<a href="{add_translation_url}" lang="fr">French</a>', html, allow_extra_attrs=True, count=0)

class TestInWorkflowStatus(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    @classmethod
    def setUpTestData(cls):
        if False:
            while True:
                i = 10
        cls.event_index = Page.objects.get(url_path='/home/events/')
        cls.christmas = Page.objects.get(url_path='/home/events/christmas/').specific
        cls.saint_patrick = Page.objects.get(url_path='/home/events/saint-patrick/').specific
        cls.christmas.save_revision()
        cls.saint_patrick.save_revision()
        cls.url = reverse('wagtailadmin_explore', args=[cls.event_index.pk])

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = self.login()

    def test_in_workflow_status(self):
        if False:
            return 10
        workflow = Workflow.objects.first()
        workflow.start(self.christmas, self.user)
        workflow.start(self.saint_patrick, self.user)
        self.client.get(self.url)
        with self.assertNumQueries(47):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        soup = self.get_soup(response.content)
        for page in [self.christmas, self.saint_patrick]:
            status = soup.select_one(f'a.w-status[href="{page.url}"]')
            self.assertIsNotNone(status)
            self.assertEqual(status.text.strip(), 'Current page status: live + in moderation')
            self.assertEqual(page.status_string, 'live + in moderation')