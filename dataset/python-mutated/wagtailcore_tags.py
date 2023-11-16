from django import template
from django.shortcuts import resolve_url
from django.template.defaulttags import token_kwargs
from django.template.loader import render_to_string
from django.utils.encoding import force_str
from django.utils.html import conditional_escape
from wagtail import VERSION, __version__
from wagtail.models import Page, Site
from wagtail.rich_text import RichText, expand_db_html
from wagtail.utils.version import get_main_version
register = template.Library()

@register.simple_tag(takes_context=True)
def pageurl(context, page, fallback=None):
    if False:
        print('Hello World!')
    "\n    Outputs a page's URL as relative (/foo/bar/) if it's within the same site as the\n    current page, or absolute (http://example.com/foo/bar/) if not.\n    If kwargs contains a fallback view name and page is None, the fallback view url will be returned.\n    "
    if page is None and fallback:
        return resolve_url(fallback)
    if not isinstance(page, Page):
        raise ValueError('pageurl tag expected a Page object, got %r' % page)
    return page.get_url(request=context.get('request'))

@register.simple_tag(takes_context=True)
def fullpageurl(context, page, fallback=None):
    if False:
        return 10
    "\n    Outputs a page's absolute URL (http://example.com/foo/bar/)\n    If kwargs contains a fallback view name and page is None, the fallback view url will be returned.\n    "
    if page is None and fallback:
        fallback_url = resolve_url(fallback)
        if fallback_url and 'request' in context and (fallback_url[0] == '/'):
            fallback_url = context['request'].build_absolute_uri(fallback_url)
        return fallback_url
    if not isinstance(page, Page):
        raise ValueError('fullpageurl tag expected a Page object, got %r' % page)
    return page.get_full_url(request=context.get('request'))

@register.simple_tag(takes_context=True)
def slugurl(context, slug):
    if False:
        print('Hello World!')
    '\n    Returns the URL for the page that has the given slug.\n\n    First tries to find a page on the current site. If that fails or a request\n    is not available in the context, then returns the URL for the first page\n    that matches the slug on any site.\n    '
    page = None
    try:
        site = Site.find_for_request(context['request'])
        current_site = site
    except KeyError:
        pass
    else:
        if current_site is not None:
            page = Page.objects.in_site(current_site).filter(slug=slug).first()
    if page is None:
        page = Page.objects.filter(slug=slug).first()
    if page:
        return pageurl(context, page)

@register.simple_tag
def wagtail_version():
    if False:
        while True:
            i = 10
    return __version__

@register.simple_tag
def wagtail_documentation_path():
    if False:
        for i in range(10):
            print('nop')
    (major, minor, patch, release, num) = VERSION
    if release == 'final':
        return 'https://docs.wagtail.org/en/v%s' % __version__
    else:
        return 'https://docs.wagtail.org/en/latest'

@register.simple_tag
def wagtail_release_notes_path():
    if False:
        for i in range(10):
            print('nop')
    return '%s.html' % get_main_version(VERSION)

@register.simple_tag
def wagtail_feature_release_whats_new_link():
    if False:
        print('Hello World!')
    return 'https://guide.wagtail.org/en-latest/releases/latest/'

@register.simple_tag
def wagtail_feature_release_editor_guide_link():
    if False:
        return 10
    return 'https://guide.wagtail.org/'

@register.filter
def richtext(value):
    if False:
        i = 10
        return i + 15
    if isinstance(value, RichText):
        return value
    elif value is None:
        html = ''
    elif isinstance(value, str):
        html = expand_db_html(value)
    else:
        raise TypeError("'richtext' template filter received an invalid value; expected string, got {}.".format(type(value)))
    return render_to_string('wagtailcore/shared/richtext.html', {'html': html})

class IncludeBlockNode(template.Node):

    def __init__(self, block_var, extra_context, use_parent_context):
        if False:
            i = 10
            return i + 15
        self.block_var = block_var
        self.extra_context = extra_context
        self.use_parent_context = use_parent_context

    def render(self, context):
        if False:
            return 10
        try:
            value = self.block_var.resolve(context)
        except template.VariableDoesNotExist:
            return ''
        if hasattr(value, 'render_as_block'):
            if self.use_parent_context:
                new_context = context.flatten()
            else:
                new_context = {}
            if self.extra_context:
                for (var_name, var_value) in self.extra_context.items():
                    new_context[var_name] = var_value.resolve(context)
            output = value.render_as_block(context=new_context)
        else:
            output = value
        if context.autoescape:
            return conditional_escape(output)
        else:
            return force_str(output)

@register.tag
def include_block(parser, token):
    if False:
        while True:
            i = 10
    "\n    Render the passed item of StreamField content, passing the current template context\n    if there's an identifiable way of doing so (i.e. if it has a `render_as_block` method).\n    "
    tokens = token.split_contents()
    try:
        tag_name = tokens.pop(0)
        block_var_token = tokens.pop(0)
    except IndexError:
        raise template.TemplateSyntaxError('%r tag requires at least one argument' % tag_name)
    block_var = parser.compile_filter(block_var_token)
    if tokens and tokens[0] == 'with':
        tokens.pop(0)
        extra_context = token_kwargs(tokens, parser)
    else:
        extra_context = None
    use_parent_context = True
    if tokens and tokens[0] == 'only':
        tokens.pop(0)
        use_parent_context = False
    if tokens:
        raise template.TemplateSyntaxError(f'Unexpected argument to {tag_name!r} tag: {tokens[0]!r}')
    return IncludeBlockNode(block_var, extra_context, use_parent_context)

@register.simple_tag(takes_context=True)
def wagtail_site(context):
    if False:
        i = 10
        return i + 15
    '\n    Returns the Site object for the given request\n    '
    try:
        request = context['request']
    except KeyError:
        return None
    return Site.find_for_request(request=request)