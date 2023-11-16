from pyramid.httpexceptions import HTTPMethodNotAllowed
from pyramid.viewderivers import INGRESS, csrf_view
SAFE_METHODS = {'GET', 'HEAD', 'OPTIONS'}

def require_method_view(view, info):
    if False:
        i = 10
        return i + 15
    require_methods = info.options.get('require_methods', SAFE_METHODS)
    explicit = bool(info.options.get('require_methods'))
    if not require_methods:
        return view

    def wrapped(context, request):
        if False:
            while True:
                i = 10
        if request.method not in require_methods and (getattr(request, 'exception', None) is None or explicit):
            raise HTTPMethodNotAllowed(headers={'Allow': ', '.join(sorted(require_methods))})
        return view(context, request)
    return wrapped
require_method_view.options = {'require_methods'}

def includeme(config):
    if False:
        for i in range(10):
            print('nop')
    config.set_default_csrf_options(require_csrf=True)
    config.add_view_deriver(csrf_view, under=INGRESS, over='secured_view')
    config.add_view_deriver(require_method_view, under=INGRESS, over='csrf_view')