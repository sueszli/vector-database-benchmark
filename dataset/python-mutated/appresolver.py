from collections import OrderedDict
from importlib import import_module
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import OperationalError, ProgrammingError
from django.urls import Resolver404, URLResolver, reverse
from django.urls.resolvers import RegexPattern, URLPattern
from django.utils.translation import get_language, override
from cms.apphook_pool import apphook_pool
from cms.models.pagemodel import Page
from cms.utils import get_current_site
from cms.utils.i18n import get_language_list
from cms.utils.moderator import _use_draft
APP_RESOLVERS = []

def clear_app_resolvers():
    if False:
        i = 10
        return i + 15
    global APP_RESOLVERS
    APP_RESOLVERS = []

def applications_page_check(request, path=None):
    if False:
        print('Hello World!')
    'Tries to find if given path was resolved over application.\n    Applications have higher priority than other cms pages.\n    '
    if path is None:
        path = request.path_info.replace(reverse('pages-root'), '', 1)
    for lang in get_language_list():
        if path.startswith(lang + '/'):
            path = path[len(lang + '/'):]
    use_public = not _use_draft(request)
    for resolver in APP_RESOLVERS:
        try:
            page_id = resolver.resolve_page_id(path)
            page = Page.objects.public().get(id=page_id)
            return page if use_public else page.publisher_public
        except Resolver404:
            pass
        except Page.DoesNotExist:
            pass
    return None

class AppRegexURLResolver(URLResolver):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.page_id = None
        self.url_patterns_dict = {}
        super().__init__(*args, **kwargs)

    @property
    def urlconf_module(self):
        if False:
            for i in range(10):
                print('nop')
        return self.url_patterns_dict.get(get_language(), [])

    @property
    def url_patterns(self):
        if False:
            return 10
        return self.urlconf_module

    def resolve_page_id(self, path):
        if False:
            while True:
                i = 10
        'Resolves requested path similar way how resolve does, but instead\n        of return callback,.. returns page_id to which was application\n        assigned.\n        '
        tried = []
        pattern = getattr(self, 'pattern', self)
        match = pattern.regex.search(path)
        if match:
            new_path = path[match.end():]
            for pattern in self.url_patterns:
                if isinstance(pattern, AppRegexURLResolver):
                    try:
                        return pattern.resolve_page_id(new_path)
                    except Resolver404:
                        pass
                else:
                    try:
                        sub_match = pattern.resolve(new_path)
                    except Resolver404 as e:
                        tried_match = e.args[0].get('tried')
                        if tried_match is not None:
                            tried.extend([[pattern] + t for t in tried_match])
                        else:
                            tried.extend([pattern])
                    else:
                        if sub_match:
                            return getattr(pattern, 'page_id', None)
                        pattern = getattr(pattern, 'pattern', pattern)
                        tried.append(pattern.regex.pattern)
            raise Resolver404({'tried': tried, 'path': new_path})

def recurse_patterns(path, pattern_list, page_id, default_args=None, nested=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Recurse over a list of to-be-hooked patterns for a given path prefix\n    '
    newpatterns = []
    for pattern in pattern_list:
        app_pat = getattr(pattern, 'pattern', pattern).regex.pattern
        app_pat = app_pat.lstrip('^')
        path = path.lstrip('^')
        regex = '^%s%s' % (path, app_pat) if not nested else '^%s' % app_pat
        if isinstance(pattern, URLResolver):
            args = pattern.default_kwargs
            if default_args:
                args.update(default_args)
            urlconf_module = recurse_patterns(regex, pattern.url_patterns, page_id, args, nested=True)
            regex_pattern = RegexPattern(regex)
            resolver = URLResolver(regex_pattern, urlconf_module, pattern.default_kwargs, pattern.app_name, pattern.namespace)
        else:
            args = pattern.default_args
            if default_args:
                args.update(default_args)
            regex_pattern = RegexPattern(regex, name=pattern.name, is_endpoint=True)
            resolver = URLPattern(regex_pattern, pattern.callback, args, pattern.name)
        resolver.page_id = page_id
        newpatterns.append(resolver)
    return newpatterns

def _set_permissions(patterns, exclude_permissions):
    if False:
        for i in range(10):
            print('nop')
    for pattern in patterns:
        if isinstance(pattern, URLResolver):
            if pattern.namespace in exclude_permissions:
                continue
            _set_permissions(pattern.url_patterns, exclude_permissions)
        else:
            from cms.utils.decorators import cms_perms
            pattern.callback = cms_perms(pattern.callback)

def get_app_urls(urls):
    if False:
        return 10
    for urlconf in urls:
        if isinstance(urlconf, str):
            mod = import_module(urlconf)
            if not hasattr(mod, 'urlpatterns'):
                raise ImproperlyConfigured('URLConf `%s` has no urlpatterns attribute' % urlconf)
            yield mod.urlpatterns
        elif isinstance(urlconf, (list, tuple)):
            yield urlconf
        else:
            yield [urlconf]

def get_patterns_for_title(path, title):
    if False:
        while True:
            i = 10
    '\n    Resolve the urlconf module for a path+title combination\n    Returns a list of url objects.\n    '
    app = apphook_pool.get_apphook(title.page.application_urls)
    url_patterns = []
    for pattern_list in get_app_urls(app.get_urls(title.page, title.language)):
        if path and (not path.endswith('/')):
            path += '/'
        page_id = title.page.id
        url_patterns += recurse_patterns(path, pattern_list, page_id)
    return url_patterns

def get_app_patterns():
    if False:
        while True:
            i = 10
    try:
        site = get_current_site()
        return _get_app_patterns(site)
    except (OperationalError, ProgrammingError):
        return []

def _get_app_patterns(site):
    if False:
        while True:
            i = 10
    "\n    Get a list of patterns for all hooked apps.\n\n    How this works:\n\n    By looking through all titles with an app hook (application_urls) we find\n    all urlconf modules we have to hook into titles.\n\n    If we use the ML URL Middleware, we namespace those patterns with the title\n    language.\n\n    All 'normal' patterns from the urlconf get re-written by prefixing them with\n    the title path and then included into the cms url patterns.\n\n    If the app is still configured, but is no longer installed/available, then\n    this method returns a degenerate patterns object: patterns('')\n    "
    from cms.models import Title
    included = []
    title_qs = Title.objects.public().filter(page__node__site=site)
    hooked_applications = OrderedDict()
    titles = title_qs.exclude(page__application_urls=None).exclude(page__application_urls='').order_by('-page__node__path').select_related()
    for title in titles:
        path = title.path
        mix_id = '%s:%s:%s' % (path + '/', title.page.application_urls, title.language)
        if mix_id in included:
            continue
        if not settings.APPEND_SLASH:
            path += '/'
        app = apphook_pool.get_apphook(title.page.application_urls)
        if not app:
            continue
        if title.page_id not in hooked_applications:
            hooked_applications[title.page_id] = {}
        app_ns = (app.app_name, title.page.application_namespace)
        with override(title.language):
            hooked_applications[title.page_id][title.language] = (app_ns, get_patterns_for_title(path, title), app)
        included.append(mix_id)
    app_patterns = []
    for page_id in hooked_applications.keys():
        resolver = None
        for lang in hooked_applications[page_id].keys():
            ((app_ns, inst_ns), current_patterns, app) = hooked_applications[page_id][lang]
            if not resolver:
                regex_pattern = RegexPattern('')
                resolver = AppRegexURLResolver(regex_pattern, 'app_resolver', app_name=app_ns, namespace=inst_ns)
                resolver.page_id = page_id
            if app.permissions:
                _set_permissions(current_patterns, app.exclude_permissions)
            resolver.url_patterns_dict[lang] = current_patterns
        app_patterns.append(resolver)
        APP_RESOLVERS.append(resolver)
    return app_patterns