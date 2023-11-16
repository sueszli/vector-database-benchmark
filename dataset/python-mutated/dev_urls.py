import os
from urllib.parse import urlsplit
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.views import serve as staticfiles_serve
from django.http.request import HttpRequest
from django.http.response import FileResponse
from django.urls import path
from django.views.generic import TemplateView
from django.views.static import serve
from zerver.views.auth import login_page
from zerver.views.development.cache import remove_caches
from zerver.views.development.camo import handle_camo_url
from zerver.views.development.dev_login import api_dev_fetch_api_key, api_dev_list_users, dev_direct_login
from zerver.views.development.email_log import clear_emails, email_page, generate_all_emails
from zerver.views.development.integrations import check_send_webhook_fixture_message, dev_panel, get_fixtures, send_all_webhook_fixture_messages
from zerver.views.development.registration import confirmation_key, register_demo_development_realm, register_development_realm, register_development_user
from zerver.views.errors import config_error
use_prod_static = not settings.DEBUG
urls = [path('coverage/<path:path>', serve, {'document_root': os.path.join(settings.DEPLOY_ROOT, 'var/coverage'), 'show_indexes': True}), path('node-coverage/<path:path>', serve, {'document_root': os.path.join(settings.DEPLOY_ROOT, 'var/node-coverage/lcov-report'), 'show_indexes': True}), path('docs/<path:path>', serve, {'document_root': os.path.join(settings.DEPLOY_ROOT, 'docs/_build/html')}), path('devlogin/', login_page, {'template_name': 'zerver/development/dev_login.html'}, name='login_page'), path('emails/', email_page), path('emails/generate/', generate_all_emails), path('emails/clear/', clear_emails), path('devtools/', TemplateView.as_view(template_name='zerver/development/dev_tools.html')), path('devtools/register_user/', register_development_user, name='register_dev_user'), path('devtools/register_realm/', register_development_realm, name='register_dev_realm'), path('devtools/register_demo_realm/', register_demo_development_realm, name='register_demo_dev_realm'), path('errors/404/', TemplateView.as_view(template_name='404.html')), path('errors/5xx/', TemplateView.as_view(template_name='500.html')), path('devtools/integrations/', dev_panel), path('devtools/integrations/check_send_webhook_fixture_message', check_send_webhook_fixture_message), path('devtools/integrations/send_all_webhook_fixture_messages', send_all_webhook_fixture_messages), path('devtools/integrations/<integration_name>/fixtures', get_fixtures), path('config-error/<error_name>', config_error, name='config_error'), path('flush_caches', remove_caches), path('external_content/<digest>/<received_url>', handle_camo_url)]
v1_api_mobile_patterns = [path('dev_fetch_api_key', api_dev_fetch_api_key), path('dev_list_users', api_dev_list_users)]
if use_prod_static:
    urls += [path('static/<path:path>', serve, {'document_root': settings.STATIC_ROOT})]
else:

    def serve_static(request: HttpRequest, path: str) -> FileResponse:
        if False:
            i = 10
            return i + 15
        response = staticfiles_serve(request, path)
        response['Access-Control-Allow-Origin'] = '*'
        return response
    assert settings.STATIC_URL is not None
    urls += static(urlsplit(settings.STATIC_URL).path, view=serve_static)
i18n_urls = [path('accounts/login/local/', dev_direct_login, name='login-local'), path('confirmation_key/', confirmation_key)]
urls += i18n_urls