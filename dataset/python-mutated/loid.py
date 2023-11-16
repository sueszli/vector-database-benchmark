from datetime import datetime, timedelta
from dateutil.parser import parse as date_parse
import pytz
import string
from urllib import quote, unquote
from pylons import app_globals as g
from . import hooks
from .utils import randstr, to_epoch_milliseconds
LOID_COOKIE = 'loid'
LOID_CREATED_COOKIE = 'loidcreated'
EXPIRES_RELATIVE = timedelta(days=2 * 365)
GLOBAL_VERSION = 0
LOID_LENGTH = 18
LOID_CHARSPACE = string.uppercase + string.lowercase + string.digits

def isodate(d):
    if False:
        return 10
    d = d.astimezone(pytz.UTC)
    milliseconds = ('%06d' % d.microsecond)[0:3]
    return d.strftime('%Y-%m-%dT%H:%M:%S.') + milliseconds + 'Z'

def ensure_unquoted(cookie_str):
    if False:
        while True:
            i = 10
    "Keep unquoting.  Never surrender.\n\n    Some of the cookies issued in the first version of this patch ended up\n    doubly quote()d.  As a preventative measure, unquote several times.\n    [This could be a while loop, because every iteration will cause the str\n    to at worst get shorter and at best stay the same and break the loop.  I\n    just don't want to replace an escaping error with a possible infinite\n    loop.]\n\n    :param str cookie_str: Cookie string.\n    "
    for _ in range(3):
        new_str = unquote(cookie_str)
        if new_str == cookie_str:
            return new_str
        cookie_str = new_str

class LoId(object):
    """Container for holding and validating logged out ids.

    The primary accessor functions for this class are:

     * :py:meth:`load` to pull the ``LoId`` out of the request cookies
     * :py:meth:`save` to save an ``LoId`` to cookies
     * :py:meth:`to_dict` to serialize this object's data to the event pipe
    """

    def __init__(self, request, context, loid=None, new=None, version=GLOBAL_VERSION, created=None, serializable=True):
        if False:
            return 10
        self.context = context
        self.request = request
        self.new = new
        self.loid = loid and str(loid)
        self.created = created or datetime.now(pytz.UTC)
        self.version = version
        self.serializable = serializable
        self.dirty = new

    def _trigger_event(self, action):
        if False:
            for i in range(10):
                print('nop')
        g.events.loid_event(loid=self, action_name=action, request=self.request, context=self.context)

    @classmethod
    def _create(cls, request, context):
        if False:
            return 10
        'Create and return a new logged out id and timestamp.\n\n        This also triggers an loid_event in the event pipeline.\n\n        :param request: current :py:module:`pylons` request object\n        :param context: current :py:module:`pylons` context object\n        :rtype: :py:class:`LoId`\n        :returns: new ``LoId``\n        '
        loid = cls(request=request, context=context, new=True, loid=randstr(LOID_LENGTH, LOID_CHARSPACE))
        loid._trigger_event('create_loid')
        return loid

    @classmethod
    def load(cls, request, context, create=True):
        if False:
            i = 10
            return i + 15
        'Load loid (and timestamp) from cookie or optionally create one.\n\n        :param request: current :py:module:`pylons` request object\n        :param context: current :py:module:`pylons` context object\n        :param bool create: On failure to load from a cookie,\n        :rtype: :py:class:`LoId`\n        '
        stub = cls(request, context, serializable=False)
        loid = request.cookies.get(LOID_COOKIE)
        if loid:
            (loid, _, _) = unquote(loid).partition('.')
            try:
                created = ensure_unquoted(request.cookies.get(LOID_CREATED_COOKIE, ''))
                created = date_parse(created)
            except ValueError:
                created = None
            return cls(request, context, new=False, loid=loid, version=0, created=created)
        elif create:
            return cls._create(request, context)
        else:
            return stub

    def save(self, **cookie_attrs):
        if False:
            while True:
                i = 10
        'Write to cookie if serializable and dirty (generally new).\n\n        :param dict cookie_attrs: additional cookie attrs.\n        '
        if self.serializable and self.dirty:
            expires = datetime.utcnow() + EXPIRES_RELATIVE
            for (name, value) in ((LOID_COOKIE, self.loid), (LOID_CREATED_COOKIE, isodate(self.created))):
                d = cookie_attrs.copy()
                d.setdefault('expires', expires)
                self.context.cookies.add(name, value, **d)

    def to_dict(self, prefix=None):
        if False:
            while True:
                i = 10
        'Serialize LoId, generally for use in the event pipeline.'
        if not self.serializable:
            return {}
        d = {'loid': self.loid, 'loid_created': to_epoch_milliseconds(self.created), 'loid_new': self.new, 'loid_version': self.version}
        hook = hooks.get_hook('loid.to_dict')
        hook.call(loid=self, data=d)
        if prefix:
            d = {'{}{}'.format(prefix, k): v for (k, v) in d.iteritems()}
        return d