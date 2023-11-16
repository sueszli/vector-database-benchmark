"""API endpoints for social authentication with allauth."""
import logging
from importlib import import_module
from django.urls import include, path, reverse
from allauth.account.models import EmailAddress
from allauth.socialaccount import providers
from allauth.socialaccount.models import SocialApp
from allauth.socialaccount.providers.keycloak.views import KeycloakOAuth2Adapter
from allauth.socialaccount.providers.oauth2.views import OAuth2Adapter, OAuth2LoginView
from drf_spectacular.utils import OpenApiResponse, extend_schema
from rest_framework.exceptions import NotFound
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from common.models import InvenTreeSetting
from InvenTree.mixins import CreateAPI, ListAPI, ListCreateAPI
from InvenTree.serializers import InvenTreeModelSerializer
logger = logging.getLogger('inventree')

class GenericOAuth2ApiLoginView(OAuth2LoginView):
    """Api view to login a user with a social account"""

    def dispatch(self, request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Dispatch the regular login view directly.'
        return self.login(request, *args, **kwargs)

class GenericOAuth2ApiConnectView(GenericOAuth2ApiLoginView):
    """Api view to connect a social account to the current user"""

    def dispatch(self, request, *args, **kwargs):
        if False:
            print('Hello World!')
        'Dispatch the connect request directly.'
        request.GET = request.GET.copy()
        request.GET['process'] = 'connect'
        return super().dispatch(request, *args, **kwargs)

def handle_oauth2(adapter: OAuth2Adapter):
    if False:
        while True:
            i = 10
    'Define urls for oauth2 endpoints.'
    return [path('login/', GenericOAuth2ApiLoginView.adapter_view(adapter), name=f'{provider.id}_api_login'), path('connect/', GenericOAuth2ApiConnectView.adapter_view(adapter), name=f'{provider.id}_api_connect')]

def handle_keycloak():
    if False:
        print('Hello World!')
    'Define urls for keycloak.'
    return [path('login/', GenericOAuth2ApiLoginView.adapter_view(KeycloakOAuth2Adapter), name='keycloak_api_login'), path('connect/', GenericOAuth2ApiConnectView.adapter_view(KeycloakOAuth2Adapter), name='keycloak_api_connet')]
legacy = {'twitter': 'twitter_oauth2', 'bitbucket': 'bitbucket_oauth2', 'linkedin': 'linkedin_oauth2', 'vimeo': 'vimeo_oauth2', 'openid': 'openid_connect'}
social_auth_urlpatterns = []
provider_urlpatterns = []
for provider in providers.registry.get_list():
    try:
        prov_mod = import_module(provider.get_package() + '.views')
    except ImportError:
        continue
    adapters = [cls for cls in prov_mod.__dict__.values() if isinstance(cls, type) and (not cls == OAuth2Adapter) and issubclass(cls, OAuth2Adapter)]
    urls = []
    if len(adapters) == 1:
        urls = handle_oauth2(adapter=adapters[0])
    elif provider.id in legacy:
        logger.warning('`%s` is not supported on platform UI. Use `%s` instead.', provider.id, legacy[provider.id])
        continue
    elif provider.id == 'keycloak':
        urls = handle_keycloak()
    else:
        logger.error('Found handler that is not yet ready for platform UI: `%s`. Open an feature request on GitHub if you need it implemented.', provider.id)
        continue
    provider_urlpatterns += [path(f'{provider.id}/', include(urls))]
social_auth_urlpatterns += provider_urlpatterns

class SocialProviderListView(ListAPI):
    """List of available social providers."""
    permission_classes = (AllowAny,)

    def get(self, request, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Get the list of providers.'
        provider_list = []
        for provider in providers.registry.get_list():
            provider_data = {'id': provider.id, 'name': provider.name, 'login': request.build_absolute_uri(reverse(f'{provider.id}_api_login')), 'connect': request.build_absolute_uri(reverse(f'{provider.id}_api_connect')), 'configured': False}
            try:
                provider_app = provider.get_app(request)
                provider_data['display_name'] = provider_app.name
                provider_data['configured'] = True
            except SocialApp.DoesNotExist:
                provider_data['display_name'] = provider.name
            provider_list.append(provider_data)
        data = {'sso_enabled': InvenTreeSetting.get_setting('LOGIN_ENABLE_SSO'), 'sso_registration': InvenTreeSetting.get_setting('LOGIN_ENABLE_SSO_REG'), 'mfa_required': InvenTreeSetting.get_setting('LOGIN_ENFORCE_MFA'), 'providers': provider_list}
        return Response(data)

class EmailAddressSerializer(InvenTreeModelSerializer):
    """Serializer for the EmailAddress model."""

    class Meta:
        """Meta options for EmailAddressSerializer."""
        model = EmailAddress
        fields = '__all__'

class EmptyEmailAddressSerializer(InvenTreeModelSerializer):
    """Empty Serializer for the EmailAddress model."""

    class Meta:
        """Meta options for EmailAddressSerializer."""
        model = EmailAddress
        fields = []

class EmailListView(ListCreateAPI):
    """List of registered email addresses for current users."""
    permission_classes = (IsAuthenticated,)
    serializer_class = EmailAddressSerializer

    def get_queryset(self):
        if False:
            return 10
        'Only return data for current user.'
        return EmailAddress.objects.filter(user=self.request.user)

class EmailActionMixin(CreateAPI):
    """Mixin to modify email addresses for current users."""
    serializer_class = EmptyEmailAddressSerializer
    permission_classes = (IsAuthenticated,)

    def get_queryset(self):
        if False:
            for i in range(10):
                print('nop')
        'Filter queryset for current user.'
        return EmailAddress.objects.filter(user=self.request.user, pk=self.kwargs['pk']).first()

    @extend_schema(responses={200: OpenApiResponse(response=EmailAddressSerializer)})
    def post(self, request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Filter item, run action and return data.'
        email = self.get_queryset()
        if not email:
            raise NotFound
        self.special_action(email, request, *args, **kwargs)
        return Response(EmailAddressSerializer(email).data)

class EmailVerifyView(EmailActionMixin):
    """Re-verify an email for a currently logged in user."""

    def special_action(self, email, request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Send confirmation.'
        if email.verified:
            return
        email.send_confirmation(request)

class EmailPrimaryView(EmailActionMixin):
    """Make an email for a currently logged in user primary."""

    def special_action(self, email, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Mark email as primary.'
        if email.primary:
            return
        email.set_as_primary()

class EmailRemoveView(EmailActionMixin):
    """Remove an email for a currently logged in user."""

    def special_action(self, email, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Delete email.'
        email.delete()