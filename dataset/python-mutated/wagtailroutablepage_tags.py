from django import template
from wagtail.models import Site
register = template.Library()

@register.simple_tag(takes_context=True)
def routablepageurl(context, page, url_name, *args, **kwargs):
    if False:
        return 10
    '\n    ``routablepageurl`` is similar to ``pageurl``, but works with\n    pages using ``RoutablePageMixin``. It behaves like a hybrid between the built-in\n    ``reverse``, and ``pageurl`` from Wagtail.\n\n    ``page`` is the RoutablePage that URLs will be generated from.\n\n    ``url_name`` is a URL name defined in ``page.subpage_urls``.\n\n    Positional arguments and keyword arguments should be passed as normal\n    positional arguments and keyword arguments.\n    '
    request = context['request']
    site = Site.find_for_request(request)
    base_url = page.relative_url(site, request)
    routed_url = page.reverse_subpage(url_name, args=args, kwargs=kwargs)
    if not base_url.endswith('/'):
        base_url += '/'
    return base_url + routed_url