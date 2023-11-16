from django.http import HttpRequest, HttpResponse
from django.views.generic import View
from sentry.web.helpers import render_to_response

class Error404View(View):

    def dispatch(self, request: HttpRequest, exception=None) -> HttpResponse:
        if False:
            for i in range(10):
                print('nop')
        return render_to_response('sentry/404.html', status=404, request=request)