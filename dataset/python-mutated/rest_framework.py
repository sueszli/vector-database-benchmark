import re
from django import template
from django.template import loader
from django.urls import NoReverseMatch, reverse
from django.utils.encoding import iri_to_uri
from django.utils.html import escape, format_html, smart_urlquote
from django.utils.safestring import mark_safe
from rest_framework.compat import apply_markdown, pygments_highlight
from rest_framework.renderers import HTMLFormRenderer
from rest_framework.utils.urls import replace_query_param
register = template.Library()
class_re = re.compile('(?<=class=["\\\'])(.*)(?=["\\\'])')

@register.tag(name='code')
def highlight_code(parser, token):
    if False:
        return 10
    code = token.split_contents()[-1]
    nodelist = parser.parse(('endcode',))
    parser.delete_first_token()
    return CodeNode(code, nodelist)

class CodeNode(template.Node):
    style = 'emacs'

    def __init__(self, lang, code):
        if False:
            print('Hello World!')
        self.lang = lang
        self.nodelist = code

    def render(self, context):
        if False:
            while True:
                i = 10
        text = self.nodelist.render(context)
        return pygments_highlight(text, self.lang, self.style)

@register.filter()
def with_location(fields, location):
    if False:
        print('Hello World!')
    return [field for field in fields if field.location == location]

@register.simple_tag
def form_for_link(link):
    if False:
        print('Hello World!')
    import coreschema
    properties = {field.name: field.schema or coreschema.String() for field in link.fields}
    required = [field.name for field in link.fields if field.required]
    schema = coreschema.Object(properties=properties, required=required)
    return mark_safe(coreschema.render_to_form(schema))

@register.simple_tag
def render_markdown(markdown_text):
    if False:
        while True:
            i = 10
    if apply_markdown is None:
        return markdown_text
    return mark_safe(apply_markdown(markdown_text))

@register.simple_tag
def get_pagination_html(pager):
    if False:
        print('Hello World!')
    return pager.to_html()

@register.simple_tag
def render_form(serializer, template_pack=None):
    if False:
        for i in range(10):
            print('nop')
    style = {'template_pack': template_pack} if template_pack else {}
    renderer = HTMLFormRenderer()
    return renderer.render(serializer.data, None, {'style': style})

@register.simple_tag
def render_field(field, style):
    if False:
        return 10
    renderer = style.get('renderer', HTMLFormRenderer())
    return renderer.render_field(field, style)

@register.simple_tag
def optional_login(request):
    if False:
        i = 10
        return i + 15
    "\n    Include a login snippet if REST framework's login view is in the URLconf.\n    "
    try:
        login_url = reverse('rest_framework:login')
    except NoReverseMatch:
        return ''
    snippet = "<li><a href='{href}?next={next}'>Log in</a></li>"
    snippet = format_html(snippet, href=login_url, next=escape(request.path))
    return mark_safe(snippet)

@register.simple_tag
def optional_docs_login(request):
    if False:
        i = 10
        return i + 15
    "\n    Include a login snippet if REST framework's login view is in the URLconf.\n    "
    try:
        login_url = reverse('rest_framework:login')
    except NoReverseMatch:
        return 'log in'
    snippet = "<a href='{href}?next={next}'>log in</a>"
    snippet = format_html(snippet, href=login_url, next=escape(request.path))
    return mark_safe(snippet)

@register.simple_tag
def optional_logout(request, user):
    if False:
        return 10
    "\n    Include a logout snippet if REST framework's logout view is in the URLconf.\n    "
    try:
        logout_url = reverse('rest_framework:logout')
    except NoReverseMatch:
        snippet = format_html('<li class="navbar-text">{user}</li>', user=escape(user))
        return mark_safe(snippet)
    snippet = '<li class="dropdown">\n        <a href="#" class="dropdown-toggle" data-toggle="dropdown">\n            {user}\n            <b class="caret"></b>\n        </a>\n        <ul class="dropdown-menu">\n            <li><a href=\'{href}?next={next}\'>Log out</a></li>\n        </ul>\n    </li>'
    snippet = format_html(snippet, user=escape(user), href=logout_url, next=escape(request.path))
    return mark_safe(snippet)

@register.simple_tag
def add_query_param(request, key, val):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add a query parameter to the current request url, and return the new url.\n    '
    iri = request.get_full_path()
    uri = iri_to_uri(iri)
    return escape(replace_query_param(uri, key, val))

@register.filter
def as_string(value):
    if False:
        return 10
    if value is None:
        return ''
    return '%s' % value

@register.filter
def as_list_of_strings(value):
    if False:
        for i in range(10):
            print('nop')
    return ['' if item is None else '%s' % item for item in value]

@register.filter
def add_class(value, css_class):
    if False:
        while True:
            i = 10
    '\n    https://stackoverflow.com/questions/4124220/django-adding-css-classes-when-rendering-form-fields-in-a-template\n\n    Inserts classes into template variables that contain HTML tags,\n    useful for modifying forms without needing to change the Form objects.\n\n    Usage:\n\n        {{ field.label_tag|add_class:"control-label" }}\n\n    In the case of REST Framework, the filter is used to add Bootstrap-specific\n    classes to the forms.\n    '
    html = str(value)
    match = class_re.search(html)
    if match:
        m = re.search('^%s$|^%s\\s|\\s%s\\s|\\s%s$' % (css_class, css_class, css_class, css_class), match.group(1))
        if not m:
            return mark_safe(class_re.sub(match.group(1) + ' ' + css_class, html))
    else:
        return mark_safe(html.replace('>', ' class="%s">' % css_class, 1))
    return value

@register.filter
def format_value(value):
    if False:
        print('Hello World!')
    if getattr(value, 'is_hyperlink', False):
        name = str(value.obj)
        return mark_safe('<a href=%s>%s</a>' % (value, escape(name)))
    if value is None or isinstance(value, bool):
        return mark_safe('<code>%s</code>' % {True: 'true', False: 'false', None: 'null'}[value])
    elif isinstance(value, list):
        if any((isinstance(item, (list, dict)) for item in value)):
            template = loader.get_template('rest_framework/admin/list_value.html')
        else:
            template = loader.get_template('rest_framework/admin/simple_list_value.html')
        context = {'value': value}
        return template.render(context)
    elif isinstance(value, dict):
        template = loader.get_template('rest_framework/admin/dict_value.html')
        context = {'value': value}
        return template.render(context)
    elif isinstance(value, str):
        if (value.startswith('http:') or value.startswith('https:') or value.startswith('/')) and (not re.search('\\s', value)):
            return mark_safe('<a href="{value}">{value}</a>'.format(value=escape(value)))
        elif '@' in value and (not re.search('\\s', value)):
            return mark_safe('<a href="mailto:{value}">{value}</a>'.format(value=escape(value)))
        elif '\n' in value:
            return mark_safe('<pre>%s</pre>' % escape(value))
    return str(value)

@register.filter
def items(value):
    if False:
        while True:
            i = 10
    "\n    Simple filter to return the items of the dict. Useful when the dict may\n    have a key 'items' which is resolved first in Django template dot-notation\n    lookup.  See issue #4931\n    Also see: https://stackoverflow.com/questions/15416662/django-template-loop-over-dictionary-items-with-items-as-key\n    "
    if value is None:
        return []
    return value.items()

@register.filter
def data(value):
    if False:
        return 10
    '\n    Simple filter to access `data` attribute of object,\n    specifically coreapi.Document.\n\n    As per `items` filter above, allows accessing `document.data` when\n    Document contains Link keyed-at "data".\n\n    See issue #5395\n    '
    return value.data

@register.filter
def schema_links(section, sec_key=None):
    if False:
        print('Hello World!')
    '\n    Recursively find every link in a schema, even nested.\n    '
    NESTED_FORMAT = '%s > %s'
    links = section.links
    if section.data:
        data = section.data.items()
        for (sub_section_key, sub_section) in data:
            new_links = schema_links(sub_section, sec_key=sub_section_key)
            links.update(new_links)
    if sec_key is not None:
        new_links = {}
        for (link_key, link) in links.items():
            new_key = NESTED_FORMAT % (sec_key, link_key)
            new_links.update({new_key: link})
        return new_links
    return links

@register.filter
def add_nested_class(value):
    if False:
        i = 10
        return i + 15
    if isinstance(value, dict):
        return 'class=nested'
    if isinstance(value, list) and any((isinstance(item, (list, dict)) for item in value)):
        return 'class=nested'
    return ''
TRAILING_PUNCTUATION = ['.', ',', ':', ';', '.)', '"', "']", "'}", "'"]
WRAPPING_PUNCTUATION = [('(', ')'), ('<', '>'), ('[', ']'), ('&lt;', '&gt;'), ('"', '"'), ("'", "'")]
word_split_re = re.compile('(\\s+)')
simple_url_re = re.compile('^https?://\\[?\\w', re.IGNORECASE)
simple_url_2_re = re.compile('^www\\.|^(?!http)\\w[^@]+\\.(com|edu|gov|int|mil|net|org)$', re.IGNORECASE)
simple_email_re = re.compile('^\\S+@\\S+\\.\\S+$')

def smart_urlquote_wrapper(matched_url):
    if False:
        print('Hello World!')
    '\n    Simple wrapper for smart_urlquote. ValueError("Invalid IPv6 URL") can\n    be raised here, see issue #1386\n    '
    try:
        return smart_urlquote(matched_url)
    except ValueError:
        return None

@register.filter
def break_long_headers(header):
    if False:
        i = 10
        return i + 15
    '\n    Breaks headers longer than 160 characters (~page length)\n    when possible (are comma separated)\n    '
    if len(header) > 160 and ',' in header:
        header = mark_safe('<br> ' + ', <br>'.join(header.split(',')))
    return header