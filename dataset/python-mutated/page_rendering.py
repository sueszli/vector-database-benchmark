from django.conf import settings
from django.http import Http404
from django.shortcuts import render
from django.template.response import TemplateResponse
from django.urls import Resolver404, resolve, reverse
from cms import __version__
from cms.cache.page import set_page_cache
from cms.models import Page
from cms.utils.conf import get_cms_setting
from cms.utils.page import get_page_template_from_request
from cms.utils.page_permissions import user_can_change_page, user_can_view_page

def render_page(request, page, current_language, slug):
    if False:
        print('Hello World!')
    '\n    Renders a page\n    '
    context = {}
    context['lang'] = current_language
    context['current_page'] = page
    context['has_change_permissions'] = user_can_change_page(request.user, page)
    context['has_view_permissions'] = user_can_view_page(request.user, page)
    if not context['has_view_permissions']:
        return _handle_no_page(request)
    template = get_page_template_from_request(request)
    response = TemplateResponse(request, template, context)
    response.add_post_render_callback(set_page_cache)
    xframe_options = page.get_xframe_options()
    if xframe_options == Page.X_FRAME_OPTIONS_INHERIT or xframe_options is None:
        return response
    response.xframe_options_exempt = True
    if xframe_options == Page.X_FRAME_OPTIONS_ALLOW:
        return response
    elif xframe_options == Page.X_FRAME_OPTIONS_SAMEORIGIN:
        response['X-Frame-Options'] = 'SAMEORIGIN'
    elif xframe_options == Page.X_FRAME_OPTIONS_DENY:
        response['X-Frame-Options'] = 'DENY'
    return response

def render_object_structure(request, obj):
    if False:
        return 10
    context = {'object': obj, 'cms_toolbar': request.toolbar}
    return render(request, 'cms/toolbar/structure.html', context)

def _handle_no_page(request):
    if False:
        for i in range(10):
            print('nop')
    try:
        resolve('%s$' % request.path)
    except Resolver404 as e:
        exc = Http404({'path': request.path, 'tried': e.args[0]['tried']})
        raise exc
    raise Http404('CMS Page not found: %s' % request.path)

def _render_welcome_page(request):
    if False:
        print('Hello World!')
    context = {'cms_version': __version__, 'cms_edit_on': get_cms_setting('CMS_TOOLBAR_URL__EDIT_ON'), 'django_debug': settings.DEBUG, 'next_url': reverse('pages-root')}
    return TemplateResponse(request, 'cms/welcome.html', context)