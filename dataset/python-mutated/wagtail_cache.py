from django import template
from django.template import Variable
from django.template.exceptions import TemplateSyntaxError
from django.templatetags.cache import CacheNode as DjangoCacheNode
from wagtail.models import PAGE_TEMPLATE_VAR, Site
register = template.Library()

class WagtailCacheNode(DjangoCacheNode):
    """
    A modified version of Django's `CacheNode` which is aware of Wagtail's
    page previews.
    """

    def render(self, context):
        if False:
            for i in range(10):
                print('nop')
        try:
            request = context['request']
        except KeyError:
            return self.nodelist.render(context)
        if getattr(request, 'is_preview', False):
            return self.nodelist.render(context)
        return super().render(context)

class WagtailPageCacheNode(WagtailCacheNode):
    """
    A modified version of Django's `CacheNode` designed for caching fragments
    of pages.

    This tag intentionally makes assumptions about what context is available.
    If these assumptions aren't valid, it's recommended to just use `{% wagtailcache %}`.
    """
    CACHE_SITE_TEMPLATE_VAR = 'wagtail_page_cache_site'

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.vary_on.extend([Variable(f'{PAGE_TEMPLATE_VAR}.cache_key'), Variable(f'{self.CACHE_SITE_TEMPLATE_VAR}.pk')])

    def render(self, context):
        if False:
            while True:
                i = 10
        if 'request' in context:
            with context.update({self.CACHE_SITE_TEMPLATE_VAR: Site.find_for_request(context['request'])}):
                return super().render(context)
        return super().render(context)

def register_cache_tag(tag_name, node_class):
    if False:
        return 10
    '\n    A helper function to define cache tags without duplicating `do_cache`.\n    '

    @register.tag(tag_name)
    def do_cache(parser, token):
        if False:
            return 10
        nodelist = parser.parse((f'end{tag_name}',))
        parser.delete_first_token()
        tokens = token.split_contents()
        if len(tokens) < 3:
            raise TemplateSyntaxError(f"'{tokens[0]}' tag requires at least 2 arguments.")
        if len(tokens) > 3 and tokens[-1].startswith('using='):
            cache_name = parser.compile_filter(tokens[-1][len('using='):])
            tokens = tokens[:-1]
        else:
            cache_name = None
        return node_class(nodelist, parser.compile_filter(tokens[1]), tokens[2], [parser.compile_filter(t) for t in tokens[3:]], cache_name)
register_cache_tag('wagtailcache', WagtailCacheNode)
register_cache_tag('wagtailpagecache', WagtailPageCacheNode)