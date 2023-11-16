import logging
import pytest
from django.core.exceptions import ImproperlyConfigured
from django.http import HttpResponse
from django.test import RequestFactory, TestCase
from django.views.generic import View
from oauthlib.oauth2 import Server
from oauth2_provider.oauth2_backends import OAuthLibCore
from oauth2_provider.oauth2_validators import OAuth2Validator
from oauth2_provider.views.mixins import OAuthLibMixin, OIDCLogoutOnlyMixin, OIDCOnlyMixin, ProtectedResourceMixin, ScopedResourceMixin
from . import presets

@pytest.mark.usefixtures('oauth2_settings')
class BaseTest(TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.request_factory = RequestFactory()
        super().setUpClass()

class TestOAuthLibMixin(BaseTest):

    def test_missing_oauthlib_backend_class_uses_fallback(self):
        if False:
            while True:
                i = 10

        class CustomOauthLibBackend:

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                pass
        self.oauth2_settings.OAUTH2_BACKEND_CLASS = CustomOauthLibBackend

        class TestView(OAuthLibMixin, View):
            server_class = Server
            validator_class = OAuth2Validator
        test_view = TestView()
        self.assertEqual(CustomOauthLibBackend, test_view.get_oauthlib_backend_class())
        core = test_view.get_oauthlib_core()
        self.assertTrue(isinstance(core, CustomOauthLibBackend))

    def test_missing_server_class_uses_fallback(self):
        if False:
            return 10

        class CustomServer:

            def __init__(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                pass
        self.oauth2_settings.OAUTH2_SERVER_CLASS = CustomServer

        class TestView(OAuthLibMixin, View):
            validator_class = OAuth2Validator
            oauthlib_backend_class = OAuthLibCore
        test_view = TestView()
        self.assertEqual(CustomServer, test_view.get_server_class())
        core = test_view.get_oauthlib_core()
        self.assertTrue(isinstance(core.server, CustomServer))

    def test_missing_validator_class_uses_fallback(self):
        if False:
            while True:
                i = 10

        class CustomValidator:
            pass
        self.oauth2_settings.OAUTH2_VALIDATOR_CLASS = CustomValidator

        class TestView(OAuthLibMixin, View):
            server_class = Server
            oauthlib_backend_class = OAuthLibCore
        test_view = TestView()
        self.assertEqual(CustomValidator, test_view.get_validator_class())
        core = test_view.get_oauthlib_core()
        self.assertTrue(isinstance(core.server.request_validator, CustomValidator))

    def test_correct_server(self):
        if False:
            while True:
                i = 10

        class TestView(OAuthLibMixin, View):
            server_class = Server
            validator_class = OAuth2Validator
            oauthlib_backend_class = OAuthLibCore
        request = self.request_factory.get('/fake-req')
        request.user = 'fake'
        test_view = TestView()
        self.assertIsInstance(test_view.get_server(), Server)

    def test_custom_backend(self):
        if False:
            while True:
                i = 10

        class AnotherOauthLibBackend:
            pass

        class TestView(OAuthLibMixin, View):
            server_class = Server
            validator_class = OAuth2Validator
            oauthlib_backend_class = AnotherOauthLibBackend
        request = self.request_factory.get('/fake-req')
        request.user = 'fake'
        test_view = TestView()
        self.assertEqual(test_view.get_oauthlib_backend_class(), AnotherOauthLibBackend)

class TestScopedResourceMixin(BaseTest):

    def test_missing_required_scopes(self):
        if False:
            while True:
                i = 10

        class TestView(ScopedResourceMixin, View):
            pass
        test_view = TestView()
        self.assertRaises(ImproperlyConfigured, test_view.get_scopes)

    def test_correct_required_scopes(self):
        if False:
            for i in range(10):
                print('nop')

        class TestView(ScopedResourceMixin, View):
            required_scopes = ['scope1', 'scope2']
        test_view = TestView()
        self.assertEqual(test_view.get_scopes(), ['scope1', 'scope2'])

class TestProtectedResourceMixin(BaseTest):

    def test_options_shall_pass(self):
        if False:
            i = 10
            return i + 15

        class TestView(ProtectedResourceMixin, View):
            server_class = Server
            validator_class = OAuth2Validator
        request = self.request_factory.options('/fake-req')
        view = TestView.as_view()
        response = view(request)
        self.assertEqual(response.status_code, 200)

@pytest.fixture
def oidc_only_view():
    if False:
        return 10

    class TView(OIDCOnlyMixin, View):

        def get(self, *args, **kwargs):
            if False:
                return 10
            return HttpResponse('OK')
    return TView.as_view()

@pytest.fixture
def oidc_logout_only_view():
    if False:
        print('Hello World!')

    class TView(OIDCLogoutOnlyMixin, View):

        def get(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            return HttpResponse('OK')
    return TView.as_view()

@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RW)
def test_oidc_only_mixin_oidc_enabled(oauth2_settings, rf, oidc_only_view):
    if False:
        for i in range(10):
            print('nop')
    assert oauth2_settings.OIDC_ENABLED
    rsp = oidc_only_view(rf.get('/'))
    assert rsp.status_code == 200
    assert rsp.content.decode('utf-8') == 'OK'

@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RP_LOGOUT)
def test_oidc_logout_only_mixin_oidc_enabled(oauth2_settings, rf, oidc_only_view):
    if False:
        for i in range(10):
            print('nop')
    assert oauth2_settings.OIDC_RP_INITIATED_LOGOUT_ENABLED
    rsp = oidc_only_view(rf.get('/'))
    assert rsp.status_code == 200
    assert rsp.content.decode('utf-8') == 'OK'

def test_oidc_only_mixin_oidc_disabled_debug(oauth2_settings, rf, settings, oidc_only_view):
    if False:
        print('Hello World!')
    assert oauth2_settings.OIDC_ENABLED is False
    settings.DEBUG = True
    with pytest.raises(ImproperlyConfigured) as exc:
        oidc_only_view(rf.get('/'))
    assert 'OIDC views are not enabled' in str(exc.value)

def test_oidc_logout_only_mixin_oidc_disabled_debug(oauth2_settings, rf, settings, oidc_logout_only_view):
    if False:
        i = 10
        return i + 15
    assert oauth2_settings.OIDC_RP_INITIATED_LOGOUT_ENABLED is False
    settings.DEBUG = True
    with pytest.raises(ImproperlyConfigured) as exc:
        oidc_logout_only_view(rf.get('/'))
        assert str(exc.value) == OIDCLogoutOnlyMixin.debug_error_message

def test_oidc_only_mixin_oidc_disabled_no_debug(oauth2_settings, rf, settings, oidc_only_view, caplog):
    if False:
        for i in range(10):
            print('nop')
    assert oauth2_settings.OIDC_ENABLED is False
    settings.DEBUG = False
    with caplog.at_level(logging.WARNING, logger='oauth2_provider'):
        rsp = oidc_only_view(rf.get('/'))
    assert rsp.status_code == 404
    assert len(caplog.records) == 1
    assert 'OIDC views are not enabled' in caplog.records[0].message

def test_oidc_logout_only_mixin_oidc_disabled_no_debug(oauth2_settings, rf, settings, oidc_logout_only_view, caplog):
    if False:
        while True:
            i = 10
    assert oauth2_settings.OIDC_RP_INITIATED_LOGOUT_ENABLED is False
    settings.DEBUG = False
    with caplog.at_level(logging.WARNING, logger='oauth2_provider'):
        rsp = oidc_logout_only_view(rf.get('/'))
        assert rsp.status_code == 404
        assert len(caplog.records) == 1
        assert caplog.records[0].message == OIDCLogoutOnlyMixin.debug_error_message