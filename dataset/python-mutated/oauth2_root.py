from datetime import timedelta
from django.utils.timezone import now
from django.conf import settings
from django.urls import re_path
from oauthlib import oauth2
from oauth2_provider import views
from awx.main.models import RefreshToken
from awx.api.views.root import ApiOAuthAuthorizationRootView

class TokenView(views.TokenView):

    def create_token_response(self, request):
        if False:
            while True:
                i = 10
        if request.POST.get('grant_type') == 'refresh_token' and 'refresh_token' in request.POST:
            refresh_token = RefreshToken.objects.filter(token=request.POST['refresh_token']).first()
            if refresh_token:
                expire_seconds = settings.OAUTH2_PROVIDER.get('REFRESH_TOKEN_EXPIRE_SECONDS', 0)
                if refresh_token.created + timedelta(seconds=expire_seconds) < now():
                    return (request.build_absolute_uri(), {}, 'The refresh token has expired.', '403')
        try:
            return super(TokenView, self).create_token_response(request)
        except oauth2.AccessDeniedError as e:
            return (request.build_absolute_uri(), {}, str(e), '403')
urls = [re_path('^$', ApiOAuthAuthorizationRootView.as_view(), name='oauth_authorization_root_view'), re_path('^authorize/$', views.AuthorizationView.as_view(), name='authorize'), re_path('^token/$', TokenView.as_view(), name='token'), re_path('^revoke_token/$', views.RevokeTokenView.as_view(), name='revoke-token')]
__all__ = ['urls']