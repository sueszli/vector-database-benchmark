from django.core.exceptions import PermissionDenied
from django.template.response import TemplateResponse
from django.test import SimpleTestCase, modify_settings, override_settings
from django.urls import path

class MiddlewareAccessingContent:

    def __init__(self, get_response):
        if False:
            return 10
        self.get_response = get_response

    def __call__(self, request):
        if False:
            print('Hello World!')
        response = self.get_response(request)
        assert response.content
        return response

def template_response_error_handler(request, exception=None):
    if False:
        print('Hello World!')
    return TemplateResponse(request, 'test_handler.html', status=403)

def permission_denied_view(request):
    if False:
        i = 10
        return i + 15
    raise PermissionDenied
urlpatterns = [path('', permission_denied_view)]
handler403 = template_response_error_handler

@override_settings(ROOT_URLCONF='handlers.tests_custom_error_handlers')
@modify_settings(MIDDLEWARE={'append': 'handlers.tests_custom_error_handlers.MiddlewareAccessingContent'})
class CustomErrorHandlerTests(SimpleTestCase):

    def test_handler_renders_template_response(self):
        if False:
            return 10
        '\n        BaseHandler should render TemplateResponse if necessary.\n        '
        response = self.client.get('/')
        self.assertContains(response, 'Error handler content', status_code=403)