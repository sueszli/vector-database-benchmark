from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from wagtail.contrib.redirects.models import Redirect
from wagtail.coreutils import get_dummy_request
from wagtail.models import Page, Site
from wagtail.test.routablepage.models import RoutablePageTest
from wagtail.test.testapp.models import EventIndex
from wagtail.test.utils import WagtailTestUtils
User = get_user_model()

@override_settings(WAGTAILREDIRECTS_AUTO_CREATE=True)
class TestAutocreateRedirects(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        cls.site = Site.objects.select_related('root_page').get(is_default_site=True)
        cls.user = User.objects.first()

    def setUp(self):
        if False:
            return 10
        self.home_page = self.site.root_page
        self.event_index = EventIndex.objects.get()
        self.other_page = Page.objects.get(url_path='/home/about-us/')

    def trigger_page_slug_changed_signal(self, page):
        if False:
            return 10
        page.slug += '-extra'
        with self.captureOnCommitCallbacks(execute=True):
            page.save(log_action='wagtail.publish', user=self.user, clean=False)

    def test_golden_path(self):
        if False:
            i = 10
            return i + 15
        test_subject = self.event_index
        drafts = test_subject.get_descendants().not_live()
        self.assertEqual(len(drafts), 4)
        request = get_dummy_request()
        branch_urls = []
        for page in test_subject.get_descendants(inclusive=True).live().specific(defer=True).iterator():
            main_url = page.get_url(request).rstrip('/')
            branch_urls.extend((main_url + path.rstrip('/') for path in page.get_cached_paths()))
        self.trigger_page_slug_changed_signal(test_subject)
        redirects = Redirect.objects.all()
        redirect_page_ids = {r.redirect_page_id for r in redirects}
        self.assertIn(test_subject.id, redirect_page_ids)
        for descendant in test_subject.get_descendants().live().iterator():
            self.assertIn(descendant.id, redirect_page_ids)
        for page in drafts:
            self.assertNotIn(page.id, redirect_page_ids)
        for r in redirects:
            self.assertIn(r.old_path, branch_urls)
            self.assertTrue(r.automatically_created)

    def test_no_redirects_created_when_page_is_root_for_all_sites_it_belongs_to(self):
        if False:
            print('Hello World!')
        self.trigger_page_slug_changed_signal(self.home_page)
        self.assertFalse(Redirect.objects.exists())

    def test_handling_of_existing_redirects(self):
        if False:
            print('Hello World!')
        test_subject = self.event_index
        descendants = test_subject.get_descendants().live()
        redirect1 = Redirect.objects.create(old_path=Redirect.normalise_path(descendants.first().specific.url), site=self.site, redirect_link='/some-place', automatically_created=False)
        redirect2 = Redirect.objects.create(old_path=Redirect.normalise_path(descendants.last().specific.url), site=self.site, redirect_link='/some-other-place', automatically_created=True)
        self.trigger_page_slug_changed_signal(test_subject)
        from_db = Redirect.objects.get(id=redirect1.id)
        self.assertEqual((redirect1.old_path, redirect1.site_id, redirect1.is_permanent, redirect1.redirect_link, redirect1.redirect_page), (from_db.old_path, from_db.site_id, from_db.is_permanent, from_db.redirect_link, from_db.redirect_page))
        self.assertFalse(Redirect.objects.filter(pk=redirect2.pk).exists())
        self.assertTrue(Redirect.objects.filter(old_path=redirect2.old_path, site_id=redirect2.site_id).exists())

    def test_redirect_creation_for_custom_route_paths(self):
        if False:
            for i in range(10):
                print('nop')
        homepage = Page.objects.get(id=2)
        routable_page = homepage.add_child(instance=RoutablePageTest(title='Routable Page', live=True))
        routable_page.move(self.event_index, pos='last-child')
        self.assertEqual(list(Redirect.objects.all().values_list('old_path', 'redirect_page', 'redirect_page_route_path').order_by('redirect_page_route_path')), [('/routable-page', routable_page.id, ''), ('/routable-page/not-a-valid-route', routable_page.id, '/not-a-valid-route'), ('/routable-page/render-method-test', routable_page.id, '/render-method-test/')])

    def test_no_redirects_created_when_pages_are_moved_to_a_different_site(self):
        if False:
            print('Hello World!')
        homepage_2 = Page(title='Second home', slug='second-home')
        root_page = Page.objects.get(depth=1)
        root_page.add_child(instance=homepage_2)
        Site.objects.create(root_page=homepage_2, hostname='newsite.com', port=80)
        self.event_index.move(homepage_2, pos='last-child')
        self.assertFalse(Redirect.objects.exists())

    @override_settings(WAGTAILREDIRECTS_AUTO_CREATE=False)
    def test_no_redirects_created_if_disabled(self):
        if False:
            while True:
                i = 10
        self.trigger_page_slug_changed_signal(self.event_index)
        self.assertFalse(Redirect.objects.exists())