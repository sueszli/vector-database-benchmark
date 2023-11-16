from unittest import mock
from django.conf import settings
from wagtail.models import Page
from wagtail.test.routablepage.models import RoutablePageTest
from wagtail.test.utils import WagtailPageTestCase

class TestCustomPageAssertions(WagtailPageTestCase):

    @classmethod
    def setUpTestData(cls):
        if False:
            while True:
                i = 10
        cls.superuser = cls.create_superuser('super')

    def setUp(self):
        if False:
            print('Hello World!')
        self.parent = Page.objects.get(id=2)
        self.page = RoutablePageTest(title='Hello world!', slug='hello-world')
        self.parent.add_child(instance=self.page)

    def test_is_routable(self):
        if False:
            return 10
        self.assertPageIsRoutable(self.page)

    def test_is_routable_with_alternative_route(self):
        if False:
            return 10
        self.assertPageIsRoutable(self.page, 'archive/year/1984/')

    def test_is_routable_fails_for_draft_page(self):
        if False:
            while True:
                i = 10
        self.page.live = False
        self.page.save()
        with self.assertRaises(self.failureException):
            self.assertPageIsRoutable(self.page)

    def test_is_routable_fails_for_invalid_route_path(self):
        if False:
            print('Hello World!')
        with self.assertRaises(self.failureException):
            self.assertPageIsRoutable(self.page, 'invalid-route-path/')

    @mock.patch('django.test.testcases.Client.get')
    @mock.patch('django.test.testcases.Client.force_login')
    def test_is_renderable(self, mocked_force_login, mocked_get):
        if False:
            return 10
        self.assertPageIsRenderable(self.page)
        mocked_force_login.assert_not_called()
        mocked_get.assert_called_once_with('/hello-world/', data=None)

    @mock.patch('django.test.testcases.Client.get')
    @mock.patch('django.test.testcases.Client.force_login')
    def test_is_renderable_for_alternative_route(self, mocked_force_login, mocked_get):
        if False:
            return 10
        self.assertPageIsRenderable(self.page, 'archive/year/1984/')
        mocked_force_login.assert_not_called()
        mocked_get.assert_called_once_with('/hello-world/archive/year/1984/', data=None)

    @mock.patch('django.test.testcases.Client.get')
    @mock.patch('django.test.testcases.Client.force_login')
    def test_is_renderable_for_user(self, mocked_force_login, mocked_get):
        if False:
            print('Hello World!')
        self.assertPageIsRenderable(self.page, user=self.superuser)
        mocked_force_login.assert_called_once_with(self.superuser, settings.AUTHENTICATION_BACKENDS[0])
        mocked_get.assert_called_once_with('/hello-world/', data=None)

    @mock.patch('django.test.testcases.Client.get')
    def test_is_renderable_with_query_data(self, mocked_get):
        if False:
            i = 10
            return i + 15
        query_data = {'p': 1, 'q': 'test'}
        self.assertPageIsRenderable(self.page, query_data=query_data)
        mocked_get.assert_called_once_with('/hello-world/', data=query_data)

    @mock.patch('django.test.testcases.Client.post')
    def test_is_renderable_with_query_and_post_data(self, mocked_post):
        if False:
            while True:
                i = 10
        query_data = {'p': 1, 'q': 'test'}
        post_data = {'subscribe': True}
        self.assertPageIsRenderable(self.page, query_data=query_data, post_data=post_data)
        mocked_post.assert_called_once_with('/hello-world/', data=post_data, QUERYSTRING='p=1&q=test')

    def test_is_renderable_for_draft_page(self):
        if False:
            for i in range(10):
                print('nop')
        self.page.live = False
        self.page.save()
        with self.assertRaises(self.failureException):
            self.assertPageIsRenderable(self.page)
        self.assertPageIsRenderable(self.page, accept_404=True)

    def test_is_renderable_for_invalid_route_path(self):
        if False:
            print('Hello World!')
        with self.assertRaises(self.failureException):
            self.assertPageIsRenderable(self.page, 'invalid-route-path/')
        self.assertPageIsRenderable(self.page, 'invalid-route-path/', accept_404=True)

    def test_is_rendereable_accept_redirect(self):
        if False:
            print('Hello World!')
        redirect_route_paths = ['permanant-homepage-redirect/', 'temporary-homepage-redirect/']
        for route_path in redirect_route_paths:
            with self.assertRaises(self.failureException):
                self.assertPageIsRenderable(self.page, route_path)
        for route_path in redirect_route_paths:
            self.assertPageIsRenderable(self.page, route_path, accept_redirect=True)

    def test_is_editable(self):
        if False:
            while True:
                i = 10
        self.assertPageIsEditable(self.page)

    @mock.patch('django.test.testcases.Client.force_login')
    def test_is_editable_always_authenticates(self, mocked_force_login):
        if False:
            return 10
        try:
            self.assertPageIsEditable(self.page)
        except self.failureException:
            pass
        mocked_force_login.assert_called_with(self._pageiseditable_superuser, settings.AUTHENTICATION_BACKENDS[0])
        try:
            self.assertPageIsEditable(self.page, user=self.superuser)
        except self.failureException:
            pass
        mocked_force_login.assert_called_with(self.superuser, settings.AUTHENTICATION_BACKENDS[0])

    @mock.patch('django.test.testcases.Client.get')
    @mock.patch('django.test.testcases.Client.force_login')
    def test_is_editable_with_permission_lacking_user(self, mocked_force_login, mocked_get):
        if False:
            return 10
        user = self.create_user('bob')
        with self.assertRaises(self.failureException):
            self.assertPageIsEditable(self.page, user=user)
        mocked_force_login.assert_not_called()
        mocked_get.assert_not_called()

    def test_is_editable_with_post_data(self):
        if False:
            return 10
        self.assertPageIsEditable(self.page, post_data={'title': 'Goodbye world?', 'slug': 'goodbye-world', 'content': 'goodbye'})

    def test_is_previewable(self):
        if False:
            while True:
                i = 10
        self.assertPageIsPreviewable(self.page)

    def test_is_previewable_with_post_data(self):
        if False:
            print('Hello World!')
        self.assertPageIsPreviewable(self.page, post_data={'title': 'test', 'slug': 'test'})

    def test_is_previewable_with_custom_user(self):
        if False:
            print('Hello World!')
        self.assertPageIsPreviewable(self.page, user=self.superuser)

    def test_is_previewable_for_alternative_mode(self):
        if False:
            i = 10
            return i + 15
        self.assertPageIsPreviewable(self.page, mode='extra')

    def test_is_previewable_for_broken_mode(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(self.failureException):
            self.assertPageIsPreviewable(self.page, mode='broken')