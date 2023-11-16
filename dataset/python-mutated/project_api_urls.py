import logging
import re
from django.conf.urls import include
from django.urls import URLPattern, URLResolver, re_path
from sentry.plugins.base import plugins
logger = logging.getLogger('sentry.plugins')

def load_plugin_urls(plugins):
    if False:
        i = 10
        return i + 15
    urlpatterns = []
    for plugin in plugins:
        urls = plugin.get_project_urls()
        if not urls:
            continue
        try:
            for u in urls:
                if not isinstance(u, (URLResolver, URLPattern)):
                    raise TypeError('url must be URLResolver or URLPattern, not {!r}: {!r}'.format(type(u).__name__, u))
        except Exception:
            logger.exception('routes.failed', extra={'plugin': type(plugin).__name__})
        else:
            urlpatterns.append(re_path('^%s/' % re.escape(plugin.slug), include(urls)))
    return urlpatterns
urlpatterns = load_plugin_urls(plugins.all())