import urllib.parse
from pyramid.config import PHASE3_CONFIG
from pyramid.httpexceptions import HTTPBadRequest, HTTPException
from pyramid.interfaces import ITweens

def junk_encoding(request):
    if False:
        while True:
            i = 10
    try:
        request.GET.get('', None)
    except UnicodeDecodeError:
        raise HTTPBadRequest('Invalid bytes in query string.')
    try:
        request.path_info
    except UnicodeDecodeError:
        raise HTTPBadRequest('Invalid bytes in URL.')

def invalid_forms(request):
    if False:
        return 10
    if request.method == 'POST':
        try:
            request.POST.get('', None)
        except ValueError:
            raise HTTPBadRequest('Invalid Form Data.')

def unicode_redirects(response):
    if False:
        for i in range(10):
            print('nop')
    if response.location:
        try:
            response.location.encode('ascii')
        except UnicodeEncodeError:
            response.location = '/'.join([urllib.parse.quote_plus(x) for x in response.location.split('/')])
    return response

def sanity_tween_factory_ingress(handler, registry):
    if False:
        return 10

    def sanity_tween_ingress(request):
        if False:
            return 10
        try:
            junk_encoding(request)
            invalid_forms(request)
        except HTTPException as exc:
            return exc
        return handler(request)
    return sanity_tween_ingress

def sanity_tween_factory_egress(handler, registry):
    if False:
        for i in range(10):
            print('nop')

    def sanity_tween_egress(request):
        if False:
            for i in range(10):
                print('nop')
        return unicode_redirects(handler(request))
    return sanity_tween_egress

def _add_tween(config):
    if False:
        return 10
    tweens = config.registry.queryUtility(ITweens)
    tweens.add_explicit('warehouse.sanity.sanity_tween_factory_ingress', sanity_tween_factory_ingress)
    for (tween_name, tween_factory) in tweens.implicit():
        tweens.add_explicit(tween_name, tween_factory)
    tweens.add_explicit('warehouse.sanity.sanity_tween_factory_egress', sanity_tween_factory_egress)

def includeme(config):
    if False:
        for i in range(10):
            print('nop')
    config.action(('tween', 'warehouse.sanity.sanity_tween_factory', True), _add_tween, args=(config,), order=PHASE3_CONFIG)