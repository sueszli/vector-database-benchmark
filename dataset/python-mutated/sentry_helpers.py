import functools
import os.path
import random
from collections import namedtuple
from datetime import datetime, timedelta
from random import randint
from urllib.parse import quote, urlencode
from django import template
from django.template.defaultfilters import stringfilter
from django.utils import timezone
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.utils.translation import gettext as _
from packaging.version import parse as parse_version
from sentry import options
from sentry.api.serializers import serialize as serialize_func
from sentry.utils import json
from sentry.utils.strings import soft_break as _soft_break
from sentry.utils.strings import soft_hyphenate, to_unicode, truncatechars
SentryVersion = namedtuple('SentryVersion', ['current', 'latest', 'update_available', 'build'])
register = template.Library()
truncatechars = register.filter(stringfilter(truncatechars))
truncatechars.is_safe = True

@register.filter
def to_json(obj, request=None):
    if False:
        for i in range(10):
            print('nop')
    return json.dumps_htmlsafe(obj)

@register.filter
def multiply(x, y):
    if False:
        while True:
            i = 10

    def coerce(value):
        if False:
            i = 10
            return i + 15
        if isinstance(value, ((int,), float)):
            return value
        try:
            return int(value)
        except ValueError:
            return float(value)
    return coerce(x) * coerce(y)

class AbsoluteUriNode(template.Node):

    def __init__(self, args, target_var):
        if False:
            return 10
        self.args = args
        self.target_var = target_var

    def render(self, context):
        if False:
            for i in range(10):
                print('nop')
        from sentry.utils.http import absolute_uri
        args = []
        for arg in self.args:
            try:
                arg = template.Variable(arg).resolve(context)
            except template.VariableDoesNotExist:
                arg = ''
            args.append(arg)
        if not args:
            rv = ''
        elif len(args) == 1:
            rv = args[0]
        else:
            rv = args[0].format(*args[1:])
        rv = absolute_uri(rv)
        if self.target_var is not None:
            context[self.target_var] = rv
            rv = ''
        return rv

@register.tag
def absolute_uri(parser, token):
    if False:
        while True:
            i = 10
    bits = token.split_contents()[1:]
    if len(bits) >= 2 and bits[-2] == 'as':
        target_var = bits[-1]
        bits = bits[:-2]
    else:
        target_var = None
    return AbsoluteUriNode(bits, target_var)

@register.simple_tag
def org_url(organization, path, query=None, fragment=None) -> str:
    if False:
        return 10
    '\n    Generate an absolute url for an organization\n    '
    if not hasattr(organization, 'absolute_url'):
        raise RuntimeError('organization parameter is not an Organization instance')
    return organization.absolute_url(path, query=query, fragment=fragment)

@register.simple_tag
def loading_message():
    if False:
        while True:
            i = 10
    options = ['Please wait while we load an obnoxious amount of JavaScript.', 'Escaping node_modules gravity well.', 'Parallelizing webpack builders.', 'Awaiting solution to the halting problem.', 'Collapsing wavefunctions.']
    return random.choice(options)

@register.simple_tag
def querystring(**kwargs):
    if False:
        while True:
            i = 10
    return urlencode(kwargs, doseq=False)

@register.simple_tag
def system_origin():
    if False:
        for i in range(10):
            print('nop')
    from sentry.utils.http import absolute_uri, origin_from_url
    return origin_from_url(absolute_uri())

@register.simple_tag
def security_contact():
    if False:
        for i in range(10):
            print('nop')
    return options.get('system.security-email') or options.get('system.admin-email')

@register.filter
def pprint(value, break_after=10):
    if False:
        return 10
    '\n    break_after is used to define how often a <span> is\n    inserted (for soft wrapping).\n    '
    value = to_unicode(value)
    return mark_safe('<span></span>'.join((escape(value[i:i + break_after]) for i in range(0, len(value), break_after))))

@register.filter
def is_url(value):
    if False:
        return 10
    if not isinstance(value, str):
        return False
    if not value.startswith(('http://', 'https://')):
        return False
    if ' ' in value:
        return False
    return True

@register.filter
def absolute_value(value):
    if False:
        for i in range(10):
            print('nop')
    return abs(int(value) if isinstance(value, int) else float(value))

@register.filter
def as_sorted(value):
    if False:
        return 10
    return sorted(value)

@register.filter
def small_count(v, precision=1):
    if False:
        i = 10
        return i + 15
    if not v:
        return 0
    z = [(1000000000, _('b')), (1000000, _('m')), (1000, _('k'))]
    v = int(v)
    for (x, y) in z:
        (o, p) = divmod(v, x)
        if o:
            if len(str(o)) > 2 or not p:
                return '%d%s' % (o, y)
            return f'%.{precision}f%s' % (v / float(x), y)
    return v

@register.filter
def as_tag_alias(v):
    if False:
        return 10
    return {'sentry:release': 'release', 'sentry:dist': 'dist', 'sentry:user': 'user'}.get(v, v)

@register.simple_tag(takes_context=True)
def serialize(context, value):
    if False:
        i = 10
        return i + 15
    value = serialize_func(value, context['request'].user)
    return json.dumps_htmlsafe(value)

@register.simple_tag(takes_context=True)
def get_sentry_version(context):
    if False:
        for i in range(10):
            print('nop')
    import sentry
    current = sentry.VERSION
    latest = options.get('sentry:latest_version') or current
    update_available = parse_version(latest) > parse_version(current)
    build = sentry.__build__ or current
    context['sentry_version'] = SentryVersion(current, latest, update_available, build)
    return ''

@register.filter
def timesince(value, now=None):
    if False:
        while True:
            i = 10
    from django.utils.timesince import timesince
    if now is None:
        now = timezone.now()
    if not value:
        return _('never')
    if value < now - timedelta(days=5):
        return value.date()
    value = ' '.join(timesince(value, now).split(' ')[0:2]).strip(',')
    if value == _('0 minutes'):
        return _('just now')
    if value == _('1 day'):
        return _('yesterday')
    return _('%s ago') % value

@register.filter
def duration(value):
    if False:
        print('Hello World!')
    if not value:
        return '0s'
    value = value / 1000.0
    (hours, minutes, seconds) = (0, 0, 0)
    if value > 3600:
        hours = value / 3600
        value = value % 3600
    if value > 60:
        minutes = value / 60
        value = value % 60
    seconds = value
    output = []
    if hours:
        output.append('%dh' % hours)
    if minutes:
        output.append('%dm' % minutes)
    if seconds > 1:
        output.append('%0.2fs' % seconds)
    elif seconds:
        output.append('%dms' % (seconds * 1000))
    return ''.join(output)

@register.filter
def date(dt, arg=None):
    if False:
        for i in range(10):
            print('nop')
    from django.template.defaultfilters import date
    if isinstance(dt, datetime) and (not timezone.is_aware(dt)):
        dt = dt.replace(tzinfo=timezone.utc)
    return date(dt, arg)

@register.simple_tag
def percent(value, total, format=None):
    if False:
        print('Hello World!')
    if not (value and total):
        result = 0
    else:
        result = int(value) / float(total) * 100
    if format is None:
        return int(result)
    else:
        return '%%%s' % format % result

@register.filter
def titleize(value):
    if False:
        while True:
            i = 10
    return value.replace('_', ' ').title()

@register.filter
def split(value, delim=''):
    if False:
        i = 10
        return i + 15
    return value.split(delim)

@register.filter
def urlquote(value, safe=''):
    if False:
        while True:
            i = 10
    return quote(value.encode('utf8'), safe)

@register.filter
def basename(value):
    if False:
        i = 10
        return i + 15
    return os.path.basename(value)

@register.filter
def soft_break(value, length):
    if False:
        for i in range(10):
            print('nop')
    return _soft_break(value, length, functools.partial(soft_hyphenate, length=max(length // 10, 10)))

@register.simple_tag
def random_int(a, b=None):
    if False:
        while True:
            i = 10
    if b is None:
        (a, b) = (0, a)
    return randint(a, b)

@register.filter
def get_item(dictionary, key):
    if False:
        for i in range(10):
            print('nop')
    return dictionary.get(key, '')