from django.test import TestCase, override_settings
from django.urls import reverse
from wagtail.admin.admin_url_finder import AdminURLFinder
from wagtail.contrib.redirects import models
from wagtail.log_actions import registry as log_registry
from wagtail.models import Page, Site
from wagtail.test.routablepage.models import RoutablePageTest
from wagtail.test.utils import WagtailTestUtils

@override_settings(ALLOWED_HOSTS=['testserver', 'localhost', 'test.example.com', 'other.example.com'])
class TestRedirects(TestCase):
    fixtures = ['test.json']

    def test_path_normalisation(self):
        if False:
            while True:
                i = 10
        normalise_path = models.Redirect.normalise_path
        path = normalise_path('/Hello/world.html;fizz=three;buzz=five?foo=Bar&Baz=quux2')
        self.assertEqual(path, normalise_path('/Hello/world.html;fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertEqual(path, normalise_path('http://mywebsite.com:8000/Hello/world.html;fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertEqual(path, normalise_path('Hello/world.html;fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertEqual(path, normalise_path('Hello/world.html/;fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertEqual(path, normalise_path('/Hello/world.html;fizz=three;buzz=five?foo=Bar&Baz=quux2#cool'))
        self.assertEqual(path, normalise_path('/Hello/world.html;fizz=three;buzz=five?Baz=quux2&foo=Bar'))
        self.assertEqual(path, normalise_path('/Hello/world.html;buzz=five;fizz=three?foo=Bar&Baz=quux2'))
        self.assertEqual(path, normalise_path('  /Hello/world.html;fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertEqual(path, normalise_path('/Hello/world.html;fizz=three;buzz=five?foo=Bar&Baz=quux2  '))
        self.assertNotEqual(path, normalise_path('/hello/world.html;fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertNotEqual(path, normalise_path('/Hello/world;fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertNotEqual(path, normalise_path('/Hello/world.html;fizz=three;buzz=five?foo=bar&Baz=Quux2'))
        self.assertNotEqual(path, normalise_path('/Hello/world.html;fizz=three;buzz=five?foo=Bar&baz=quux2'))
        self.assertNotEqual(path, normalise_path('/Hello/world.html;fizz=three;buzz=Five?foo=Bar&Baz=quux2'))
        self.assertNotEqual(path, normalise_path('/Hello/world.html;Fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertNotEqual(path, normalise_path('/Hello/world.html?foo=Bar&Baz=quux2'))
        self.assertNotEqual(path, normalise_path('/Hello/WORLD.html;fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertNotEqual(path, normalise_path('/Hello/world.htm;fizz=three;buzz=five?foo=Bar&Baz=quux2'))
        self.assertEqual('/', normalise_path('/'))
        normalise_path('This is not a URL')
        normalise_path('//////hello/world')
        normalise_path('!#@%$*')
        normalise_path('C:\\Program Files (x86)\\Some random program\\file.txt')

    def test_unicode_path_normalisation(self):
        if False:
            while True:
                i = 10
        normalise_path = models.Redirect.normalise_path
        self.assertEqual('/here/tésting-ünicode', normalise_path('/here/tésting-ünicode'))
        self.assertNotEqual('/here/testing-unicode', normalise_path('/here/tésting-ünicode'))

    def test_route_path_normalisation(self):
        if False:
            while True:
                i = 10
        normalise_path = models.Redirect.normalise_page_route_path
        self.assertEqual('', normalise_path('/'))
        self.assertEqual('/test/', normalise_path('test/'))
        self.assertEqual('/multiple/segment/test', normalise_path('/multiple/segment/test'))
        self.assertEqual('/multiple/segment/test/', normalise_path('/multiple/segment/test/'))

    def test_basic_redirect(self):
        if False:
            return 10
        redirect = models.Redirect(old_path='/redirectme', redirect_link='/redirectto')
        redirect.save()
        response = self.client.get('/redirectme/')
        self.assertRedirects(response, '/redirectto', status_code=301, fetch_redirect_response=False)

    def test_temporary_redirect(self):
        if False:
            while True:
                i = 10
        redirect = models.Redirect(old_path='/redirectme', redirect_link='/redirectto', is_permanent=False)
        redirect.save()
        response = self.client.get('/redirectme/')
        self.assertRedirects(response, '/redirectto', status_code=302, fetch_redirect_response=False)

    def test_redirect_stripping_query_string(self):
        if False:
            return 10
        redirect_with_query_string = models.Redirect(old_path='/redirectme?foo=Bar', redirect_link='/with-query-string-only')
        redirect_with_query_string.save()
        redirect_without_query_string = models.Redirect(old_path='/redirectme', redirect_link='/without-query-string')
        redirect_without_query_string.save()
        r_matching_qs = self.client.get('/redirectme/?foo=Bar')
        self.assertRedirects(r_matching_qs, '/with-query-string-only', status_code=301, fetch_redirect_response=False)
        r_no_qs = self.client.get('/redirectme/?utm_source=irrelevant')
        self.assertRedirects(r_no_qs, '/without-query-string', status_code=301, fetch_redirect_response=False)

    def test_redirect_to_page(self):
        if False:
            return 10
        christmas_page = Page.objects.get(url_path='/home/events/christmas/')
        models.Redirect.objects.create(old_path='/xmas', redirect_page=christmas_page)
        response = self.client.get('/xmas/', HTTP_HOST='test.example.com')
        self.assertRedirects(response, '/events/christmas/', status_code=301, fetch_redirect_response=False)

    def test_redirect_to_specific_page_route(self):
        if False:
            print('Hello World!')
        homepage = Page.objects.get(id=2)
        routable_page = homepage.add_child(instance=RoutablePageTest(title='Routable Page', live=True))
        contact_page = Page.objects.get(url_path='/home/contact-us/')
        models.Redirect.add_redirect(old_path='/old-path-one', redirect_to=routable_page, page_route_path='/render-method-test-custom-template/')
        response = self.client.get('/old-path-one/', HTTP_HOST='test.example.com')
        self.assertRedirects(response, '/routable-page/render-method-test-custom-template/', status_code=301, fetch_redirect_response=False)
        models.Redirect.add_redirect(old_path='/old-path-two', redirect_to=routable_page, page_route_path='/invalid-route/')
        response = self.client.get('/old-path-two/', HTTP_HOST='test.example.com')
        self.assertRedirects(response, '/routable-page/', status_code=301, fetch_redirect_response=False)
        models.Redirect.add_redirect(old_path='/old-path-three', redirect_to=contact_page, page_route_path='/route-to-nowhere/')
        response = self.client.get('/old-path-three/', HTTP_HOST='test.example.com')
        self.assertRedirects(response, '/contact-us/', status_code=301, fetch_redirect_response=False)

    def test_redirect_from_any_site(self):
        if False:
            for i in range(10):
                print('nop')
        contact_page = Page.objects.get(url_path='/home/contact-us/')
        Site.objects.create(hostname='other.example.com', port=80, root_page=contact_page)
        christmas_page = Page.objects.get(url_path='/home/events/christmas/')
        models.Redirect.objects.create(old_path='/xmas', redirect_page=christmas_page)
        response = self.client.get('/xmas/', HTTP_HOST='localhost')
        self.assertRedirects(response, 'http://localhost/events/christmas/', status_code=301, fetch_redirect_response=False)
        response = self.client.get('/xmas/', HTTP_HOST='other.example.com')
        self.assertRedirects(response, 'http://localhost/events/christmas/', status_code=301, fetch_redirect_response=False)

    def test_redirect_from_specific_site(self):
        if False:
            print('Hello World!')
        contact_page = Page.objects.get(url_path='/home/contact-us/')
        other_site = Site.objects.create(hostname='other.example.com', port=80, root_page=contact_page)
        christmas_page = Page.objects.get(url_path='/home/events/christmas/')
        models.Redirect.objects.create(old_path='/xmas', redirect_page=christmas_page, site=other_site)
        response = self.client.get('/xmas/', HTTP_HOST='other.example.com')
        self.assertRedirects(response, 'http://localhost/events/christmas/', status_code=301, fetch_redirect_response=False)
        response = self.client.get('/xmas/', HTTP_HOST='localhost')
        self.assertEqual(response.status_code, 404)

    def test_redirect_without_page_or_link_target(self):
        if False:
            while True:
                i = 10
        models.Redirect.objects.create(old_path='/xmas/', redirect_link='')
        response = self.client.get('/xmas/', HTTP_HOST='localhost')
        self.assertEqual(response.status_code, 404)

    def test_redirect_to_page_without_site(self):
        if False:
            print('Hello World!')
        siteless_page = Page.objects.get(url_path='/does-not-exist/')
        models.Redirect.objects.create(old_path='/xmas', redirect_page=siteless_page)
        response = self.client.get('/xmas/', HTTP_HOST='localhost')
        self.assertEqual(response.status_code, 404)

    def test_duplicate_redirects_when_match_is_for_generic(self):
        if False:
            while True:
                i = 10
        contact_page = Page.objects.get(url_path='/home/contact-us/')
        site = Site.objects.create(hostname='other.example.com', port=80, root_page=contact_page)
        models.Redirect.objects.create(old_path='/xmas', redirect_link='/generic')
        models.Redirect.objects.create(site=site, old_path='/xmas', redirect_link='/site-specific')
        response = self.client.get('/xmas/')
        self.assertRedirects(response, '/generic', status_code=301, fetch_redirect_response=False)

    def test_duplicate_redirects_with_query_string_when_match_is_for_generic(self):
        if False:
            for i in range(10):
                print('nop')
        contact_page = Page.objects.get(url_path='/home/contact-us/')
        site = Site.objects.create(hostname='other.example.com', port=80, root_page=contact_page)
        models.Redirect.objects.create(old_path='/xmas?foo=Bar', redirect_link='/generic-with-query-string')
        models.Redirect.objects.create(site=site, old_path='/xmas?foo=Bar', redirect_link='/site-specific-with-query-string')
        models.Redirect.objects.create(old_path='/xmas', redirect_link='/generic')
        models.Redirect.objects.create(site=site, old_path='/xmas', redirect_link='/site-specific')
        response = self.client.get('/xmas/?foo=Bar')
        self.assertRedirects(response, '/generic-with-query-string', status_code=301, fetch_redirect_response=False)
        response = self.client.get('/xmas/?foo=Baz')
        self.assertRedirects(response, '/generic', status_code=301, fetch_redirect_response=False)

    def test_duplicate_redirects_when_match_is_for_specific(self):
        if False:
            print('Hello World!')
        contact_page = Page.objects.get(url_path='/home/contact-us/')
        site = Site.objects.create(hostname='other.example.com', port=80, root_page=contact_page)
        models.Redirect.objects.create(old_path='/xmas', redirect_link='/generic')
        models.Redirect.objects.create(site=site, old_path='/xmas', redirect_link='/site-specific')
        response = self.client.get('/xmas/', HTTP_HOST='other.example.com')
        self.assertRedirects(response, '/site-specific', status_code=301, fetch_redirect_response=False)

    def test_duplicate_redirects_with_query_string_when_match_is_for_specific_with_qs(self):
        if False:
            return 10
        contact_page = Page.objects.get(url_path='/home/contact-us/')
        site = Site.objects.create(hostname='other.example.com', port=80, root_page=contact_page)
        models.Redirect.objects.create(old_path='/xmas?foo=Bar', redirect_link='/generic-with-query-string')
        models.Redirect.objects.create(site=site, old_path='/xmas?foo=Bar', redirect_link='/site-specific-with-query-string')
        models.Redirect.objects.create(old_path='/xmas', redirect_link='/generic')
        models.Redirect.objects.create(site=site, old_path='/xmas', redirect_link='/site-specific')
        response = self.client.get('/xmas/?foo=Bar', HTTP_HOST='other.example.com')
        self.assertRedirects(response, '/site-specific-with-query-string', status_code=301, fetch_redirect_response=False)
        response = self.client.get('/xmas/?foo=Baz', HTTP_HOST='other.example.com')
        self.assertRedirects(response, '/site-specific', status_code=301, fetch_redirect_response=False)

    def test_duplicate_page_redirects_when_match_is_for_specific(self):
        if False:
            while True:
                i = 10
        contact_page = Page.objects.get(url_path='/home/contact-us/')
        site = Site.objects.create(hostname='other.example.com', port=80, root_page=contact_page)
        christmas_page = Page.objects.get(url_path='/home/events/christmas/')
        models.Redirect.objects.create(old_path='/xmas', redirect_page=contact_page)
        models.Redirect.objects.create(site=site, old_path='/xmas', redirect_page=christmas_page)
        response = self.client.get('/xmas/', HTTP_HOST='other.example.com')
        self.assertRedirects(response, 'http://localhost/events/christmas/', status_code=301, fetch_redirect_response=False)

    def test_redirect_with_unicode_in_url(self):
        if False:
            print('Hello World!')
        redirect = models.Redirect(old_path='/tésting-ünicode', redirect_link='/redirectto')
        redirect.save()
        response = self.client.get('/tésting-ünicode/')
        self.assertRedirects(response, '/redirectto', status_code=301, fetch_redirect_response=False)

    def test_redirect_with_encoded_url(self):
        if False:
            i = 10
            return i + 15
        redirect = models.Redirect(old_path='/t%C3%A9sting-%C3%BCnicode', redirect_link='/redirectto')
        redirect.save()
        response = self.client.get('/t%C3%A9sting-%C3%BCnicode/')
        self.assertRedirects(response, '/redirectto', status_code=301, fetch_redirect_response=False)

    def test_reject_null_characters(self):
        if False:
            print('Hello World!')
        response = self.client.get('/test%00test/')
        self.assertEqual(response.status_code, 404)
        response = self.client.get('/test\x00test/')
        self.assertEqual(response.status_code, 404)
        response = self.client.get('/test/?foo=%00bar')
        self.assertEqual(response.status_code, 404)
        response = self.client.get('/test/?foo=\x00bar')
        self.assertEqual(response.status_code, 404)

    def test_add_redirect_with_url(self):
        if False:
            for i in range(10):
                print('nop')
        add_redirect = models.Redirect.add_redirect
        old_path = '/old-path'
        redirect_to = '/new-path'
        redirect = add_redirect(old_path=old_path, redirect_to=redirect_to, is_permanent=False)
        self.assertEqual(redirect.old_path, old_path)
        self.assertEqual(redirect.link, redirect_to)
        self.assertIs(redirect.is_permanent, False)

    def test_add_redirect_with_page(self):
        if False:
            for i in range(10):
                print('nop')
        add_redirect = models.Redirect.add_redirect
        old_path = '/old-path'
        redirect_to = Page.objects.get(url_path='/home/events/christmas/')
        redirect = add_redirect(old_path=old_path, redirect_to=redirect_to)
        self.assertEqual(redirect.old_path, old_path)
        self.assertEqual(redirect.link, redirect_to.url)
        self.assertIs(redirect.is_permanent, True)

class TestRedirectsIndexView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.login()

    def get(self, params={}):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(reverse('wagtailredirects:index'), params)

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailredirects/index.html')

    def test_search(self):
        if False:
            return 10
        response = self.get({'q': 'Hello'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['query_string'], 'Hello')

    def test_search_results(self):
        if False:
            print('Hello World!')
        models.Redirect.objects.create(old_path='/aaargh', redirect_link='http://torchbox.com/')
        models.Redirect.objects.create(old_path='/torchbox', redirect_link='http://aaargh.com/')
        response = self.get({'q': 'aaargh'})
        self.assertEqual(len(response.context['redirects']), 2)

    def test_pagination(self):
        if False:
            while True:
                i = 10
        response = self.get({'p': 1})
        self.assertEqual(response.status_code, 200)
        response = self.get({'p': 9999})
        self.assertEqual(response.status_code, 404)

    def test_listing_order(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(0, 10):
            models.Redirect.objects.create(old_path='/redirect%d' % i, redirect_link='http://torchbox.com/')
        models.Redirect.objects.create(old_path='/aaargh', redirect_link='http://torchbox.com/')
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['redirects'][0].old_path, '/aaargh')

class TestRedirectsAddView(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            while True:
                i = 10
        self.login()

    def get(self, params={}):
        if False:
            while True:
                i = 10
        return self.client.get(reverse('wagtailredirects:add'), params)

    def post(self, post_data={}):
        if False:
            print('Hello World!')
        return self.client.post(reverse('wagtailredirects:add'), post_data)

    def test_simple(self):
        if False:
            return 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailredirects/add.html')

    def test_add(self):
        if False:
            print('Hello World!')
        response = self.post({'old_path': '/test', 'site': '', 'is_permanent': 'on', 'redirect_link': 'http://www.test.com/'})
        self.assertRedirects(response, reverse('wagtailredirects:index'))
        redirects = models.Redirect.objects.filter(old_path='/test')
        redirect = redirects.first()
        self.assertEqual(redirects.count(), 1)
        self.assertEqual(redirect.redirect_link, 'http://www.test.com/')
        self.assertIsNone(redirect.site)
        log_entry = log_registry.get_logs_for_instance(redirect).first()
        self.assertEqual(log_entry.action, 'wagtail.create')

    def test_add_with_site(self):
        if False:
            i = 10
            return i + 15
        localhost = Site.objects.get(hostname='localhost')
        response = self.post({'old_path': '/test', 'site': localhost.id, 'is_permanent': 'on', 'redirect_link': 'http://www.test.com/'})
        self.assertRedirects(response, reverse('wagtailredirects:index'))
        redirects = models.Redirect.objects.filter(old_path='/test')
        self.assertEqual(redirects.count(), 1)
        self.assertEqual(redirects.first().redirect_link, 'http://www.test.com/')
        self.assertEqual(redirects.first().site, localhost)

    def test_add_validation_error(self):
        if False:
            while True:
                i = 10
        response = self.post({'old_path': '', 'site': '', 'is_permanent': 'on', 'redirect_link': 'http://www.test.com/'})
        self.assertEqual(response.status_code, 200)

    def test_cannot_add_duplicate_with_no_site(self):
        if False:
            i = 10
            return i + 15
        models.Redirect.objects.create(old_path='/test', site=None, redirect_link='http://elsewhere.com/')
        response = self.post({'old_path': '/test', 'site': '', 'is_permanent': 'on', 'redirect_link': 'http://www.test.com/'})
        self.assertEqual(response.status_code, 200)

    def test_cannot_add_duplicate_on_same_site(self):
        if False:
            while True:
                i = 10
        localhost = Site.objects.get(hostname='localhost')
        models.Redirect.objects.create(old_path='/test', site=localhost, redirect_link='http://elsewhere.com/')
        response = self.post({'old_path': '/test', 'site': localhost.pk, 'is_permanent': 'on', 'redirect_link': 'http://www.test.com/'})
        self.assertEqual(response.status_code, 200)

    def test_can_reuse_path_on_other_site(self):
        if False:
            while True:
                i = 10
        localhost = Site.objects.get(hostname='localhost')
        contact_page = Page.objects.get(url_path='/home/contact-us/')
        other_site = Site.objects.create(hostname='other.example.com', port=80, root_page=contact_page)
        models.Redirect.objects.create(old_path='/test', site=localhost, redirect_link='http://elsewhere.com/')
        response = self.post({'old_path': '/test', 'site': other_site.pk, 'is_permanent': 'on', 'redirect_link': 'http://www.test.com/'})
        self.assertRedirects(response, reverse('wagtailredirects:index'))
        redirects = models.Redirect.objects.filter(redirect_link='http://www.test.com/')
        self.assertEqual(redirects.count(), 1)

    def test_add_long_redirect(self):
        if False:
            print('Hello World!')
        response = self.post({'old_path': '/test', 'site': '', 'is_permanent': 'on', 'redirect_link': 'https://www.google.com/search?q=this+is+a+very+long+url+because+it+has+a+huge+search+term+appended+to+the+end+of+it+even+though+someone+should+really+not+be+doing+something+so+crazy+without+first+seeing+a+psychiatrist'})
        self.assertRedirects(response, reverse('wagtailredirects:index'))
        redirects = models.Redirect.objects.filter(old_path='/test')
        self.assertEqual(redirects.count(), 1)
        self.assertEqual(redirects.first().redirect_link, 'https://www.google.com/search?q=this+is+a+very+long+url+because+it+has+a+huge+search+term+appended+to+the+end+of+it+even+though+someone+should+really+not+be+doing+something+so+crazy+without+first+seeing+a+psychiatrist')
        self.assertIsNone(redirects.first().site)

class TestRedirectsEditView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            return 10
        self.redirect = models.Redirect(old_path='/test', redirect_link='http://www.test.com/')
        self.redirect.save()
        self.user = self.login()

    def get(self, params={}, redirect_id=None):
        if False:
            i = 10
            return i + 15
        return self.client.get(reverse('wagtailredirects:edit', args=(redirect_id or self.redirect.id,)), params)

    def post(self, post_data={}, redirect_id=None):
        if False:
            while True:
                i = 10
        return self.client.post(reverse('wagtailredirects:edit', args=(redirect_id or self.redirect.id,)), post_data)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailredirects/edit.html')
        url_finder = AdminURLFinder(self.user)
        expected_url = '/admin/redirects/%d/' % self.redirect.id
        self.assertEqual(url_finder.get_edit_url(self.redirect), expected_url)

    def test_nonexistant_redirect(self):
        if False:
            return 10
        self.assertEqual(self.get(redirect_id=100000).status_code, 404)

    def test_edit(self):
        if False:
            print('Hello World!')
        response = self.post({'old_path': '/test', 'is_permanent': 'on', 'site': '', 'redirect_link': 'http://www.test.com/ive-been-edited'})
        self.assertRedirects(response, reverse('wagtailredirects:index'))
        redirects = models.Redirect.objects.filter(old_path='/test')
        self.assertEqual(redirects.count(), 1)
        self.assertEqual(redirects.first().redirect_link, 'http://www.test.com/ive-been-edited')
        self.assertIsNone(redirects.first().site)

    def test_edit_with_site(self):
        if False:
            for i in range(10):
                print('nop')
        localhost = Site.objects.get(hostname='localhost')
        response = self.post({'old_path': '/test', 'is_permanent': 'on', 'site': localhost.id, 'redirect_link': 'http://www.test.com/ive-been-edited'})
        self.assertRedirects(response, reverse('wagtailredirects:index'))
        redirects = models.Redirect.objects.filter(old_path='/test')
        self.assertEqual(redirects.count(), 1)
        self.assertEqual(redirects.first().redirect_link, 'http://www.test.com/ive-been-edited')
        self.assertEqual(redirects.first().site, localhost)

    def test_edit_validation_error(self):
        if False:
            i = 10
            return i + 15
        response = self.post({'old_path': '', 'is_permanent': 'on', 'site': '', 'redirect_link': 'http://www.test.com/ive-been-edited'})
        self.assertEqual(response.status_code, 200)

    def test_edit_duplicate(self):
        if False:
            print('Hello World!')
        models.Redirect.objects.create(old_path='/othertest', site=None, redirect_link='http://elsewhere.com/')
        response = self.post({'old_path': '/othertest', 'is_permanent': 'on', 'site': '', 'redirect_link': 'http://www.test.com/ive-been-edited'})
        self.assertEqual(response.status_code, 200)

class TestRedirectsDeleteView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.redirect = models.Redirect(old_path='/test', redirect_link='http://www.test.com/')
        self.redirect.save()
        self.login()

    def get(self, params={}, redirect_id=None):
        if False:
            return 10
        return self.client.get(reverse('wagtailredirects:delete', args=(redirect_id or self.redirect.id,)), params)

    def post(self, redirect_id=None):
        if False:
            i = 10
            return i + 15
        return self.client.post(reverse('wagtailredirects:delete', args=(redirect_id or self.redirect.id,)))

    def test_simple(self):
        if False:
            while True:
                i = 10
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailredirects/confirm_delete.html')

    def test_nonexistant_redirect(self):
        if False:
            return 10
        self.assertEqual(self.get(redirect_id=100000).status_code, 404)

    def test_delete(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.post()
        self.assertRedirects(response, reverse('wagtailredirects:index'))
        redirects = models.Redirect.objects.filter(old_path='/test')
        self.assertEqual(redirects.count(), 0)