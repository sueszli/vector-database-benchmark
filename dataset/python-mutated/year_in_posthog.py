from django.http import HttpResponse
from django.template.loader import get_template
from django.views.decorators.cache import cache_control

@cache_control(public=True, max_age=300)
def render_2022(request, user_uuid: str) -> HttpResponse:
    if False:
        print('Hello World!')
    template = get_template('hibernating.html')
    html = template.render({}, request=request)
    return HttpResponse(html)