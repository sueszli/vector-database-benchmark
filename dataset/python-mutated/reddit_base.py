import collections
import json
import re
import simplejson
import socket
import itertools
from Cookie import CookieError
from copy import copy
from datetime import datetime, timedelta
from functools import wraps
from hashlib import sha1, md5
from urllib import quote, unquote
from urlparse import urlparse
import babel.core
import pylibmc
from mako.filters import url_escape
from pylons import request, response
from pylons import tmpl_context as c
from pylons import app_globals as g
from pylons.i18n import _
from pylons.i18n.translation import LanguageError
from r2.config import feature
from r2.config.extensions import is_api, set_extension
from r2.lib import baseplate_integration, filters, geoip, hooks, pages, ratelimit, utils
from r2.lib.base import BaseController, abort
from r2.lib.cookies import change_user_cookie_security, Cookies, Cookie, delete_secure_session_cookie, have_secure_session_cookie, upgrade_cookie_security, NEVER, DELETE
from r2.lib.errors import ErrorSet, BadRequestError, ForbiddenError, errors, reddit_http_error
from r2.lib.filters import _force_utf8, _force_unicode, scriptsafe_dumps
from r2.lib.loid import LoId
from r2.lib.require import RequirementException, require, require_split
from r2.lib.strings import strings
from r2.lib.template_helpers import add_sr, JSPreload
from r2.lib.tracking import encrypt, decrypt, get_pageview_pixel_url
from r2.lib.translation import set_lang
from r2.lib.utils import SimpleSillyStub, UniqueIterator, extract_subdomain, http_utils, is_subdomain, is_throttled, tup, UrlParser
from r2.lib.validator import build_arg_list, fullname_regex, valid_jsonp_callback, validate, VBoolean, VByName, VCount, VLang, VLength, VLimit, VTarget
from r2.models import Account, All, AllFiltered, AllMinus, DefaultSR, DomainSR, FakeAccount, FakeSubreddit, Friends, Frontpage, LabeledMulti, Link, Mod, ModFiltered, ModMinus, MultiReddit, NotFound, OAuth2AccessToken, OAuth2Client, OAuth2Scope, Random, RandomNSFW, RandomSubscription, Subreddit, valid_admin_cookie, valid_feed, valid_otp_cookie
from r2.lib.db import tdb_cassandra
CACHEABLE_COOKIES = ()

class UnloggedUser(FakeAccount):
    COOKIE_NAME = '_options'
    allowed_prefs = {'pref_lang': VLang.validate_lang, 'pref_hide_locationbar': bool, 'pref_use_global_defaults': bool}

    def __init__(self, browser_langs, *a, **kw):
        if False:
            return 10
        FakeAccount.__init__(self, *a, **kw)
        lang = browser_langs[0] if browser_langs else g.lang
        self._defaults = self._defaults.copy()
        self._defaults['pref_lang'] = lang
        self._defaults['pref_hide_locationbar'] = False
        self._defaults['pref_use_global_defaults'] = False
        self._t.update(self._from_cookie())

    @property
    def name(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def _decode_json(self, json_blob):
        if False:
            for i in range(10):
                print('nop')
        data = json.loads(json_blob)
        validated = {}
        for (k, v) in data.iteritems():
            validator = self.allowed_prefs.get(k)
            if validator:
                try:
                    validated[k] = validator(v)
                except ValueError:
                    pass
        return validated

    def _from_cookie(self):
        if False:
            return 10
        cookie = c.cookies.get(self.COOKIE_NAME)
        if not cookie:
            return {}
        try:
            return self._decode_json(cookie.value)
        except ValueError:
            try:
                plaintext = decrypt(cookie.value)
                values = self._decode_json(plaintext)
            except (TypeError, ValueError):
                c.cookies[self.COOKIE_NAME] = Cookie(value='', expires=DELETE)
                return {}
            else:
                self._to_cookie(values)
                return values

    def _to_cookie(self, data):
        if False:
            print('Hello World!')
        allowed_data = {k: v for (k, v) in data.iteritems() if k in self.allowed_prefs}
        jsonified = json.dumps(allowed_data, sort_keys=True)
        c.cookies[self.COOKIE_NAME] = Cookie(value=jsonified)

    def _subscribe(self, sr):
        if False:
            return 10
        pass

    def _unsubscribe(self, sr):
        if False:
            while True:
                i = 10
        pass

    def _commit(self):
        if False:
            return 10
        if self._dirty:
            for (k, (oldv, newv)) in self._dirties.iteritems():
                self._t[k] = newv
            self._to_cookie(self._t)

def read_user_cookie(name):
    if False:
        print('Hello World!')
    uname = c.user.name if c.user_is_loggedin else ''
    cookie_name = uname + '_' + name
    if cookie_name in c.cookies:
        return c.cookies[cookie_name].value
    else:
        return ''

def set_user_cookie(name, val, **kwargs):
    if False:
        return 10
    uname = c.user.name if c.user_is_loggedin else ''
    c.cookies[uname + '_' + name] = Cookie(value=val, **kwargs)
valid_click_cookie = fullname_regex(Link, True).match

def set_recent_clicks():
    if False:
        i = 10
        return i + 15
    c.recent_clicks = []
    if not c.user_is_loggedin:
        return
    click_cookie = read_user_cookie('recentclicks2')
    if click_cookie:
        if valid_click_cookie(click_cookie):
            names = [x for x in UniqueIterator(click_cookie.split(',')) if x]
            if len(names) > 5:
                names = names[:5]
                set_user_cookie('recentclicks2', ','.join(names))
            names = names[:5]
            try:
                c.recent_clicks = Link._by_fullname(names, data=True, return_dict=False)
            except NotFound:
                set_user_cookie('recentclicks2', '')
        else:
            set_user_cookie('recentclicks2', '')

def delete_obsolete_cookies():
    if False:
        i = 10
        return i + 15
    for cookie_name in c.cookies:
        if cookie_name.endswith(('_last_thing', '_mod')):
            c.cookies[cookie_name] = Cookie('', expires=DELETE)

def over18():
    if False:
        i = 10
        return i + 15
    if c.user_is_loggedin:
        return c.user.pref_over_18 or c.user_is_admin
    elif 'over18' in c.cookies:
        cookie = c.cookies['over18'].value
        if cookie == '1':
            return True
        else:
            delete_over18_cookie()

def set_over18_cookie():
    if False:
        print('Hello World!')
    c.cookies.add('over18', '1')

def delete_over18_cookie():
    if False:
        i = 10
        return i + 15
    c.cookies['over18'] = Cookie(value='', expires=DELETE)

def set_obey_over18():
    if False:
        i = 10
        return i + 15
    'querystring parameter for API to obey over18 filtering rules'
    c.obey_over18 = request.GET.get('obey_over18') == 'true'
valid_ascii_domain = re.compile('\\A(\\w[-\\w]*\\.)+[\\w]+\\Z')

def set_subreddit():
    if False:
        for i in range(10):
            print('nop')
    sr_name = request.environ.get('subreddit', request.params.get('r'))
    domain = request.environ.get('domain')
    can_stale = request.method.upper() in ('GET', 'HEAD')
    c.site = Frontpage
    if not sr_name:
        cname = request.environ.get('legacy-cname')
        if cname:
            sr = Subreddit._by_domain(cname) or Frontpage
            domain = g.domain
            if g.domain_prefix:
                domain = '.'.join((g.domain_prefix, domain))
            path = '%s://%s%s' % (g.default_scheme, domain, sr.path)
            abort(301, location=BaseController.format_output_url(path))
    elif '+' in sr_name:
        name_filter = lambda name: Subreddit.is_valid_name(name, allow_language_srs=True)
        sr_names = filter(name_filter, sr_name.split('+'))
        srs = Subreddit._by_name(sr_names, stale=can_stale).values()
        if All in srs:
            c.site = All
        elif Friends in srs:
            c.site = Friends
        else:
            srs = [sr for sr in srs if not isinstance(sr, FakeSubreddit)]
            if len(srs) == 1:
                c.site = srs[0]
            elif srs:
                found = {sr.name.lower() for sr in srs}
                sr_names = filter(lambda name: name.lower() in found, sr_names)
                sr_name = '+'.join(sr_names)
                multi_path = '/r/' + sr_name
                c.site = MultiReddit(multi_path, srs)
            elif not c.error_page:
                abort(404)
    elif '-' in sr_name:
        sr_names = sr_name.split('-')
        (base_sr_name, exclude_sr_names) = (sr_names[0], sr_names[1:])
        srs = Subreddit._by_name(sr_names, stale=can_stale)
        base_sr = srs.pop(base_sr_name, None)
        exclude_srs = [sr for sr in srs.itervalues() if not isinstance(sr, FakeSubreddit)]
        if base_sr == All:
            if exclude_srs:
                c.site = AllMinus(exclude_srs)
            else:
                c.site = All
        elif base_sr == Mod:
            if exclude_srs:
                c.site = ModMinus(exclude_srs)
            else:
                c.site = Mod
        else:
            path = '/subreddits/search?q=%s' % sr_name
            abort(302, location=BaseController.format_output_url(path))
    else:
        try:
            c.site = Subreddit._by_name(sr_name, stale=can_stale)
        except NotFound:
            if Subreddit.is_valid_name(sr_name):
                path = '/subreddits/search?q=%s' % sr_name
                abort(302, location=BaseController.format_output_url(path))
            elif not c.error_page and (not request.path.startswith('/api/login/')):
                abort(404)
    if not sr_name and isinstance(c.site, DefaultSR) and domain:
        try:
            idna = _force_unicode(domain).encode('idna')
            if idna != domain:
                path_info = request.environ['PATH_INFO']
                path = '/domain/%s%s' % (idna, path_info)
                abort(302, location=BaseController.format_output_url(path))
        except UnicodeError:
            domain = ''
        if not c.error_page and (not valid_ascii_domain.match(domain)):
            abort(404)
        c.site = DomainSR(domain)
    if isinstance(c.site, FakeSubreddit):
        c.default_sr = True
_FILTER_SRS = {'mod': ModFiltered, 'all': AllFiltered}

def set_multireddit():
    if False:
        while True:
            i = 10
    routes_dict = request.environ['pylons.routes_dict']
    if 'multipath' in routes_dict or ('m' in request.GET and is_api()):
        fullpath = routes_dict.get('multipath', '').lower()
        multipaths = fullpath.split('+')
        multi_ids = None
        logged_in_username = c.user.name.lower() if c.user_is_loggedin else None
        multiurl = None
        if c.user_is_loggedin and routes_dict.get('my_multi'):
            multi_ids = ['/user/%s/m/%s' % (logged_in_username, multipath) for multipath in multipaths]
            multiurl = '/me/m/' + fullpath
        elif 'username' in routes_dict:
            username = routes_dict['username'].lower()
            if c.user_is_loggedin:
                if username == logged_in_username and (not is_api()):
                    url_parts = request.path_qs.split('/')[5:]
                    url_parts.insert(0, '/me/m/%s' % fullpath)
                    path = '/'.join(url_parts)
                    abort(302, location=BaseController.format_output_url(path))
            multiurl = '/user/' + username + '/m/' + fullpath
            multi_ids = ['/user/%s/m/%s' % (username, multipath) for multipath in multipaths]
        elif 'm' in request.GET and is_api():
            multi_ids = [m.lower() for m in request.GET.getall('m') if m]
            multiurl = ''
        if multi_ids is not None:
            multis = LabeledMulti._byID(multi_ids, return_dict=False) or []
            multis = [m for m in multis if m.can_view(c.user)]
            if not multis:
                abort(404)
            elif len(multis) == 1:
                c.site = multis[0]
            else:
                sr_ids = Subreddit.random_reddits(logged_in_username, list(set(itertools.chain.from_iterable((multi.sr_ids for multi in multis)))), LabeledMulti.MAX_SR_COUNT)
                srs = Subreddit._byID(sr_ids, data=True, return_dict=False)
                c.site = MultiReddit(multiurl, srs)
                if any((m.weighting_scheme == 'fresh' for m in multis)):
                    c.site.weighting_scheme = 'fresh'
    elif 'filtername' in routes_dict:
        if not c.user_is_loggedin:
            abort(404)
        filtername = routes_dict['filtername'].lower()
        filtersr = _FILTER_SRS.get(filtername)
        if not filtersr:
            abort(404)
        c.site = filtersr()

def set_content_type():
    if False:
        for i in range(10):
            print('nop')
    e = request.environ
    c.render_style = e['render_style']
    response.content_type = e['content_type']
    if e.has_key('extension'):
        c.extension = ext = e['extension']
        if ext in ('embed', 'widget'):
            wrapper = request.params.get('callback', 'document.write')
            wrapper = filters._force_utf8(wrapper)
            if not valid_jsonp_callback(wrapper):
                abort(BadRequestError(errors.BAD_JSONP_CALLBACK))
            c.user = UnloggedUser(get_browser_langs())
            c.user_is_loggedin = False
            c.forced_loggedout = True

            def to_js(content):
                if False:
                    i = 10
                    return i + 15
                return '/**/' + wrapper + '(' + utils.string2js(content) + ');'
            c.response_wrapper = to_js
        if ext in ('rss', 'api', 'json') and request.method.upper() == 'GET':
            user = valid_feed(request.GET.get('user'), request.GET.get('feed'), request.path)
            if user and (not g.read_only_mode):
                c.user = user
                c.user_is_loggedin = True
        if ext in ('mobile', 'm') and (not request.GET.get('keep_extension')):
            try:
                if request.cookies['reddit_mobility'] == 'compact':
                    c.extension = 'compact'
                    c.render_style = 'compact'
            except (ValueError, KeyError):
                c.suggest_compact = True
        if ext in ('mobile', 'm', 'compact'):
            if request.GET.get('keep_extension'):
                c.cookies['reddit_mobility'] = Cookie(ext, expires=NEVER)
    if is_api() or c.render_style in ('html', 'mobile', 'compact'):
        c.loid = LoId.load(request, c)
    callback = request.GET.get('jsonp')
    if is_api() and request.method.upper() == 'GET' and callback:
        if not valid_jsonp_callback(callback):
            abort(BadRequestError(errors.BAD_JSONP_CALLBACK))
        c.allowed_callback = callback
        c.user = UnloggedUser(get_browser_langs())
        c.user_is_loggedin = False
        c.forced_loggedout = True
        response.content_type = 'application/javascript'

def get_browser_langs():
    if False:
        print('Hello World!')
    browser_langs = []
    langs = request.environ.get('HTTP_ACCEPT_LANGUAGE')
    if langs:
        langs = langs.split(',')
        browser_langs = []
        seen_langs = set()
        for l in langs:
            if ';' in l:
                l = l.split(';')[0]
            if l not in seen_langs and l in g.languages:
                browser_langs.append(l)
                seen_langs.add(l)
            if '-' in l:
                l = l.split('-')[0]
            if l not in seen_langs and l in g.languages:
                browser_langs.append(l)
                seen_langs.add(l)
    return browser_langs

def set_iface_lang():
    if False:
        while True:
            i = 10
    host_lang = request.environ.get('reddit-prefer-lang')
    lang = host_lang or c.user.pref_lang
    if getattr(g, 'lang_override') and lang == 'en':
        lang = g.lang_override
    c.lang = lang
    try:
        set_lang(lang, fallback_lang=g.lang)
    except LanguageError:
        lang = g.lang
        set_lang(lang, graceful_fail=True)
    try:
        c.locale = babel.core.Locale.parse(lang, sep='-')
    except (babel.core.UnknownLocaleError, ValueError):
        c.locale = babel.core.Locale.parse(g.lang, sep='-')

def set_colors():
    if False:
        print('Hello World!')
    theme_rx = re.compile('')
    color_rx = re.compile('\\A([a-fA-F0-9]){3}(([a-fA-F0-9]){3})?\\Z')
    c.theme = None
    if color_rx.match(request.GET.get('bgcolor') or ''):
        c.bgcolor = request.GET.get('bgcolor')
    if color_rx.match(request.GET.get('bordercolor') or ''):
        c.bordercolor = request.GET.get('bordercolor')

def ratelimit_agent(agent, limit=10, slice_size=10):
    if False:
        return 10
    h = md5()
    h.update(agent)
    hashed_agent = h.hexdigest()
    slice_size = min(slice_size, 60)
    time_slice = ratelimit.get_timeslice(slice_size)
    usage = ratelimit.record_usage('rl-agent-' + hashed_agent, time_slice)
    if usage > limit:
        request.environ['retry_after'] = time_slice.remaining
        abort(429)
appengine_re = re.compile('AppEngine-Google; \\(\\+http://code.google.com/appengine; appid: (?:dev|s)~([a-z0-9-]{6,30})\\)\\Z')

def ratelimit_agents():
    if False:
        print('Hello World!')
    user_agent = request.user_agent
    if not user_agent:
        return
    appengine_match = appengine_re.search(user_agent)
    if appengine_match:
        appid = appengine_match.group(1)
        ratelimit_agent(appid)
        return
    for (agent_re, limit) in g.user_agent_ratelimit_regexes.iteritems():
        if agent_re.search(user_agent):
            ratelimit_agent(agent_re.pattern, limit)
            return

def ratelimit_throttled():
    if False:
        print('Hello World!')
    ip = request.ip.strip()
    if is_throttled(ip):
        abort(429)

def paginated_listing(default_page_size=25, max_page_size=100, backend='sql'):
    if False:
        i = 10
        return i + 15

    def decorator(fn):
        if False:
            for i in range(10):
                print('nop')

        @validate(num=VLimit('limit', default=default_page_size, max_limit=max_page_size), after=VByName('after', backend=backend), before=VByName('before', backend=backend), count=VCount('count'), target=VTarget('target'), sr_detail=VBoolean('sr_detail', docs={'sr_detail': '(optional) expand subreddits'}), show=VLength('show', 3, empty_error=None, docs={'show': '(optional) the string `all`'}))
        @wraps(fn)
        def new_fn(self, before, **env):
            if False:
                print('Hello World!')
            if c.render_style == 'htmllite':
                c.link_target = env.get('target')
            elif 'target' in env:
                del env['target']
            if 'show' in env and env['show'] == 'all':
                c.ignore_hide_rules = True
            kw = build_arg_list(fn, env)
            kw['reverse'] = False
            if before:
                kw['after'] = before
                kw['reverse'] = True
            return fn(self, **kw)
        if hasattr(fn, '_api_doc'):
            notes = fn._api_doc['notes'] or []
            if paginated_listing.doc_note not in notes:
                notes.append(paginated_listing.doc_note)
            fn._api_doc['notes'] = notes
        return new_fn
    return decorator
paginated_listing.doc_note = '*This endpoint is [a listing](#listings).*'

def base_listing(fn):
    if False:
        i = 10
        return i + 15
    return paginated_listing()(fn)

def is_trusted_origin(origin):
    if False:
        i = 10
        return i + 15
    try:
        origin = urlparse(origin)
    except ValueError:
        return False
    return any((is_subdomain(origin.hostname, domain) for domain in g.trusted_domains))

def cross_domain(origin_check=is_trusted_origin, **options):
    if False:
        return 10
    'Set up cross domain validation and hoisting for a request handler.'

    def cross_domain_wrap(fn):
        if False:
            while True:
                i = 10
        cors_perms = {'origin_check': origin_check, 'allow_credentials': bool(options.get('allow_credentials'))}

        @wraps(fn)
        def cross_domain_handler(self, *args, **kwargs):
            if False:
                print('Hello World!')
            if request.params.get('hoist') == 'cookie':
                if cors_perms['origin_check'](g.origin):
                    name = request.environ['pylons.routes_dict']['action_name']
                    resp = fn(self, *args, **kwargs)
                    c.cookies.add('hoist_%s' % name, ''.join(tup(resp)))
                    response.content_type = 'text/html'
                    return ''
                else:
                    abort(403)
            else:
                self.check_cors()
                return fn(self, *args, **kwargs)
        cross_domain_handler.cors_perms = cors_perms
        return cross_domain_handler
    return cross_domain_wrap

def make_url_https(url):
    if False:
        for i in range(10):
            print('nop')
    'Turn a possibly relative URL into a fully-qualified HTTPS URL.'
    new_url = UrlParser(url)
    new_url.scheme = 'https'
    if not new_url.hostname:
        new_url.hostname = request.host.lower()
    return new_url.unparse()

def generate_modhash():
    if False:
        print('Hello World!')
    if c.oauth_user:
        return None
    modhash = hooks.get_hook('modhash.generate').call_until_return()
    if modhash is not None:
        return modhash
    return c.user.name

def enforce_https():
    if False:
        print('Hello World!')
    'Enforce policy for forced usage of HTTPS.'
    if c.oauth_user:
        return
    if c.forced_loggedout or c.render_style == 'js':
        return
    if request.environ.get('pylons.error_call', False):
        return
    is_api_request = is_api() or request.path.startswith('/api/')
    redirect_url = None
    if is_api_request and (not c.secure):
        ua = request.user_agent
        g.stats.count_string('https.security_violation', ua)
        if c.user_is_loggedin:
            g.stats.count_string('https.loggedin_security_violation', ua)
        if have_secure_session_cookie() and (not c.user_is_loggedin):
            redirect_url = make_url_https(request.fullurl)
    if c.render_style in {'html', 'compact', 'mobile'} and (not is_api_request):
        want_redirect = feature.is_enabled('force_https') or feature.is_enabled('https_redirect')
        if not c.secure and want_redirect:
            redirect_url = make_url_https(request.fullurl)
    if redirect_url:
        headers = {'Cache-Control': 'private, no-cache', 'Pragma': 'no-cache'}
        status_code = 301 if request.method == 'GET' else 307
        abort(status_code, location=redirect_url, headers=headers)

def require_https():
    if False:
        print('Hello World!')
    if not c.secure:
        abort(ForbiddenError(errors.HTTPS_REQUIRED))

def require_domain(required_domain):
    if False:
        while True:
            i = 10
    if not is_subdomain(request.host, required_domain):
        abort(ForbiddenError(errors.WRONG_DOMAIN))

def disable_subreddit_css():
    if False:
        return 10

    def wrap(f):
        if False:
            print('Hello World!')

        @wraps(f)
        def no_funny_business(*args, **kwargs):
            if False:
                return 10
            c.allow_styles = False
            return f(*args, **kwargs)
        return no_funny_business
    return wrap

def request_timer_name(action):
    if False:
        while True:
            i = 10
    return 'service_time.web.' + action

def flatten_response(content):
    if False:
        i = 10
        return i + 15
    'Convert a content iterable to a string, properly handling unicode.'
    return ''.join((_force_utf8(x) for x in tup(content) if x))

def abort_with_error(error, code=None):
    if False:
        for i in range(10):
            print('nop')
    if not code and (not error.code):
        raise ValueError('Error %r missing status code' % error)
    abort(reddit_http_error(code=code or error.code, error_name=error.name, explanation=error.message, fields=error.fields))

class MinimalController(BaseController):
    allow_stylesheets = False
    defer_ratelimiting = False

    def run_sitewide_ratelimits(self):
        if False:
            while True:
                i = 10
        "Ratelimit users and add ratelimit headers to the response.\n\n        Headers added are:\n        X-Ratelimit-Used: Number of requests used in this period\n        X-Ratelimit-Remaining: Number of requests left to use\n        X-Ratelimit-Reset: Approximate number of seconds to end of period\n\n        This function only has an effect if one of\n        g.RL_SITEWIDE_ENABLED or g.RL_OAUTH_SITEWIDE_ENABLED\n        are set to 'true' in the app configuration\n\n        If the ratelimit is exceeded, a 429 response will be sent,\n        unless the app configuration has g.ENFORCE_RATELIMIT off.\n        Headers will be sent even on aborted requests.\n\n        "
        if c.error_page:
            return
        if c.oauth_user and g.RL_OAUTH_SITEWIDE_ENABLED:
            type_ = 'oauth'
            period = g.RL_OAUTH_RESET_SECONDS
            max_reqs = c.oauth2_client._max_reqs
            client_id = c.oauth2_access_token.client_id.encode('ascii')
            key = 'siterl-oauth-' + c.user._id36 + ':' + client_id
        elif c.cdn_cacheable:
            type_ = 'cdn'
        elif not is_api():
            type_ = 'web'
        elif g.RL_SITEWIDE_ENABLED:
            type_ = 'api'
            max_reqs = g.RL_MAX_REQS
            period = g.RL_RESET_SECONDS
            key = 'siterl-api-' + request.ip
        else:
            type_ = 'none'
        g.stats.event_count('ratelimit.type', type_, sample_rate=0.01)
        if type_ in ('cdn', 'web', 'none'):
            return
        time_slice = ratelimit.get_timeslice(period)
        try:
            recent_reqs = ratelimit.record_usage(key, time_slice)
        except ratelimit.RatelimitError as e:
            g.log.info('ratelimit error: %s', e)
            return
        reqs_remaining = max(0, max_reqs - recent_reqs)
        c.ratelimit_headers = {'X-Ratelimit-Used': str(recent_reqs), 'X-Ratelimit-Reset': str(time_slice.remaining), 'X-Ratelimit-Remaining': str(reqs_remaining)}
        event_type = None
        if reqs_remaining <= 0:
            if recent_reqs > 2 * max_reqs:
                event_type = 'hyperbolic'
            else:
                event_type = 'over'
            if g.ENFORCE_RATELIMIT:
                request.environ['retry_after'] = time_slice.remaining
                response.headers.update(c.ratelimit_headers)
                abort(429)
        elif reqs_remaining < 0.1 * max_reqs:
            event_type = 'close'
        if event_type is not None:
            g.stats.event_count('ratelimit.exceeded', event_type)
            if type_ == 'oauth':
                g.stats.count_string('oauth.{}'.format(event_type), client_id)

    def pre(self):
        if False:
            return 10
        action = request.environ['pylons.routes_dict'].get('action')
        if action:
            if not self._get_action_handler():
                action = 'invalid'
            controller = request.environ['pylons.routes_dict']['controller']
            key = '{}.{}'.format(controller, action)
            c.request_timer = g.stats.get_timer(request_timer_name(key))
        else:
            c.request_timer = SimpleSillyStub()
        baseplate_integration.make_server_span(span_name=key).start()
        c.response_wrapper = None
        c.start_time = datetime.now(g.tz)
        c.request_timer.start()
        g.reset_caches()
        c.domain_prefix = request.environ.get('reddit-domain-prefix', g.domain_prefix)
        c.secure = request.environ['wsgi.url_scheme'] == 'https'
        c.request_origin = request.host_url
        if not c.error_page:
            ratelimit_throttled()
            ratelimit_agents()
        if 'WANT_RAW_JSON' not in request.environ:
            want_raw_json = request.params.get('raw_json', '') == '1'
            request.environ['WANT_RAW_JSON'] = want_raw_json
        c.allow_framing = False
        c.referrer_policy = 'origin'
        c.cdn_cacheable = request.via_cdn and g.login_cookie not in request.cookies
        c.extension = request.environ.get('extension')
        set_subreddit()
        c.subdomain = extract_subdomain()
        c.errors = ErrorSet()
        c.cookies = Cookies()
        set_content_type()
        c.request_timer.intermediate('minimal-pre')
        c.update_last_visit = None
        if is_subdomain(request.host, g.oauth_domain):
            self.check_cors()
        if not self.defer_ratelimiting:
            self.run_sitewide_ratelimits()
            c.request_timer.intermediate('minimal-ratelimits')
        hooks.get_hook('reddit.request.minimal_begin').call()

    def post(self):
        if False:
            return 10
        c.request_timer.intermediate('action')
        c.is_exception_response = getattr(response, '_exception', False)
        if c.response_wrapper and (not c.is_exception_response):
            content = flatten_response(response.content)
            wrapped_content = c.response_wrapper(content)
            response.content = wrapped_content
        if not c.allow_framing:
            response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        if feature.is_enabled('force_https') and feature.is_enabled('upgrade_cookies'):
            upgrade_cookie_security()
        dirty_cookies = (k for (k, v) in c.cookies.iteritems() if v.dirty)
        would_poison = any((k not in CACHEABLE_COOKIES for k in dirty_cookies))
        if c.user_is_loggedin or would_poison:
            cache_control = ('private', 's-maxage=0', 'max-age=0', 'must-revalidate')
            response.headers['Expires'] = '-1'
            response.headers['Cache-Control'] = ', '.join(cache_control)
        if c.ratelimit_headers:
            response.headers.update(c.ratelimit_headers)
        if c.loid:
            c.loid.save(domain=g.domain)
        secure_cookies = feature.is_enabled('force_https')
        for (k, v) in c.cookies.iteritems():
            if v.dirty:
                v_secure = v.secure if v.secure is not None else secure_cookies
                response.set_cookie(key=k, value=quote(v.value), domain=v.domain, expires=v.expires, secure=v_secure, httponly=getattr(v, 'httponly', False))
        if not isinstance(c.site, FakeSubreddit) and (not g.disallow_db_writes):
            if c.user_is_loggedin:
                c.site.record_visitor_activity('logged_in', c.user._fullname)
        if self.should_update_last_visit():
            c.user.update_last_visit(c.start_time)
        hooks.get_hook('reddit.request.end').call()
        g.reset_caches()
        c.request_timer.intermediate('post')
        c.trace.set_tag('user', c.user._fullname if c.user_is_loggedin else None)
        c.trace.set_tag('render_style', c.render_style)
        baseplate_integration.finish_server_span()
        c.request_timer.stop()
        g.stats.flush()

    def on_validation_error(self, error):
        if False:
            i = 10
            return i + 15
        if error.name == errors.USER_REQUIRED:
            self.intermediate_redirect('/login')
        elif error.name == errors.VERIFIED_USER_REQUIRED:
            self.intermediate_redirect('/verify')

    def abort404(self):
        if False:
            while True:
                i = 10
        abort(404, 'not found')

    def abort403(self):
        if False:
            return 10
        abort(403, 'forbidden')
    COMMON_REDDIT_HEADERS = ', '.join(('X-Ratelimit-Used', 'X-Ratelimit-Remaining', 'X-Ratelimit-Reset', 'X-Moose'))

    def check_cors(self):
        if False:
            while True:
                i = 10
        origin = request.headers.get('Origin')
        if c.cors_checked or not origin:
            return
        method = request.method
        if method == 'OPTIONS':
            method = request.headers.get('Access-Control-Request-Method')
            if not method:
                self.abort403()
        via_oauth = is_subdomain(request.host, g.oauth_domain)
        if via_oauth:
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE'
            response.headers['Access-Control-Allow-Headers'] = 'Authorization, '
            response.headers['Access-Control-Allow-Credentials'] = 'false'
            response.headers['Access-Control-Expose-Headers'] = self.COMMON_REDDIT_HEADERS
        else:
            action = request.environ['pylons.routes_dict']['action_name']
            handler = self._get_action_handler(action, method)
            cors = handler and getattr(handler, 'cors_perms', None)
            if cors and cors['origin_check'](origin):
                response.headers['Access-Control-Allow-Origin'] = origin
                if cors.get('allow_credentials'):
                    response.headers['Access-Control-Allow-Credentials'] = 'true'
        c.cors_checked = True

    def OPTIONS(self):
        if False:
            i = 10
            return i + 15
        'Return empty responses for CORS preflight requests'
        self.check_cors()

    def update_qstring(self, dict):
        if False:
            return 10
        merged = copy(request.GET)
        merged.update(dict)
        return request.path + utils.query_string(merged)

    def api_wrapper(self, kw):
        if False:
            print('Hello World!')
        if request.environ.get('WANT_RAW_JSON'):
            return scriptsafe_dumps(kw)
        return filters.websafe_json(simplejson.dumps(kw))

    def should_update_last_visit(self):
        if False:
            while True:
                i = 10
        if g.disallow_db_writes:
            return False
        if not c.user_is_loggedin:
            return False
        if c.update_last_visit is not None:
            return c.update_last_visit
        return request.method.upper() != 'POST'

class OAuth2ResourceController(MinimalController):
    defer_ratelimiting = True

    def authenticate_with_token(self):
        if False:
            while True:
                i = 10
        set_extension(request.environ, 'json')
        set_content_type()
        require_https()
        require_domain(g.oauth_domain)
        try:
            access_token = OAuth2AccessToken.get_token(self._get_bearer_token())
            require(access_token)
            require(access_token.check_valid())
            c.oauth2_access_token = access_token
            if access_token.user_id:
                account = Account._byID36(access_token.user_id, data=True)
                require(account)
                require(not account._deleted)
                c.user = c.oauth_user = account
                c.user_is_loggedin = True
            else:
                c.user = UnloggedUser(get_browser_langs())
                c.user_is_loggedin = False
            c.oauth2_client = OAuth2Client._byID(access_token.client_id)
        except RequirementException:
            self._auth_error(401, 'invalid_token')
        handler = self._get_action_handler()
        if handler:
            oauth2_perms = getattr(handler, 'oauth2_perms', {})
            if oauth2_perms.get('oauth2_allowed', False):
                grant = OAuth2Scope(access_token.scope)
                required = set(oauth2_perms['required_scopes'])
                if not grant.has_access(c.site.name, required):
                    self._auth_error(403, 'insufficient_scope')
                c.oauth_scope = grant
            else:
                self._auth_error(400, 'invalid_request')

    def check_for_bearer_token(self):
        if False:
            for i in range(10):
                print('nop')
        if self._get_bearer_token(strict=False):
            self.authenticate_with_token()

    def _auth_error(self, code, error):
        if False:
            i = 10
            return i + 15
        abort(code, headers=[('WWW-Authenticate', 'Bearer realm="reddit", error="%s"' % error)])

    def _get_bearer_token(self, strict=True):
        if False:
            while True:
                i = 10
        auth = request.headers.get('Authorization')
        if not auth:
            return None
        try:
            (auth_scheme, bearer_token) = require_split(auth, 2)
            require(auth_scheme.lower() == 'bearer')
            return bearer_token
        except RequirementException:
            if strict:
                self._auth_error(400, 'invalid_request')
            else:
                return None

    def set_up_user_context(self):
        if False:
            print('Hello World!')
        if c.user.inbox_count > 0:
            c.have_messages = True
        c.have_mod_messages = bool(c.user.modmsgtime)
        c.user_special_distinguish = c.user.special_distinguish()

class OAuth2OnlyController(OAuth2ResourceController):
    """Base controller for endpoints that may only be accessed via OAuth 2"""
    handles_csrf = True

    def pre(self):
        if False:
            print('Hello World!')
        OAuth2ResourceController.pre(self)
        if request.method != 'OPTIONS':
            self.authenticate_with_token()
            self.set_up_user_context()
            self.run_sitewide_ratelimits()

    def on_validation_error(self, error):
        if False:
            i = 10
            return i + 15
        abort_with_error(error, error.code or 400)

class RedditController(OAuth2ResourceController):

    @staticmethod
    def login(user, rem=False):
        if False:
            return 10
        user.update_last_visit(c.start_time)
        force_https = feature.is_enabled('force_https', user)
        c.cookies[g.login_cookie] = Cookie(value=user.make_cookie(), expires=NEVER if rem else None, httponly=True, secure=force_https)
        change_user_cookie_security(secure=force_https, remember=rem)

    @staticmethod
    def logout():
        if False:
            i = 10
            return i + 15
        c.cookies[g.login_cookie] = Cookie(value='', expires=DELETE)
        delete_secure_session_cookie()

    @staticmethod
    def enable_admin_mode(user, first_login=None):
        if False:
            for i in range(10):
                print('nop')
        admin_cookie = user.make_admin_cookie(first_login=first_login)
        c.cookies[g.admin_cookie] = Cookie(value=admin_cookie, httponly=True, secure=feature.is_enabled('force_https'))

    @staticmethod
    def remember_otp(user):
        if False:
            print('Hello World!')
        cookie = user.make_otp_cookie()
        expiration = datetime.utcnow() + timedelta(seconds=g.OTP_COOKIE_TTL)
        set_user_cookie(g.otp_cookie, cookie, secure=True, httponly=True, expires=expiration)

    @staticmethod
    def disable_admin_mode(user):
        if False:
            while True:
                i = 10
        c.cookies[g.admin_cookie] = Cookie(value='', expires=DELETE)

    def pre(self):
        if False:
            for i in range(10):
                print('nop')
        record_timings = g.admin_cookie in request.cookies or g.debug
        admin_bar_eligible = response.content_type == 'text/html'
        if admin_bar_eligible and record_timings:
            g.stats.start_logging_timings()
        c.js_preload = JSPreload()
        MinimalController.pre(self)
        response.headers['X-UA-Compatible'] = 'IE=edge'
        if request.host != g.media_domain or g.media_domain == g.domain:
            cookie_counts = collections.Counter()
            for (k, v) in request.cookies.iteritems():
                if k not in c.cookies:
                    c.cookies[k] = Cookie(value=unquote(v), dirty=False)
                    cookie_counts[Cookie.classify(k)] += 1
            for (cookietype, count) in cookie_counts.iteritems():
                g.stats.simple_event('cookie.%s' % cookietype, count)
        delete_obsolete_cookies()
        maybe_admin = False
        is_otpcookie_valid = False
        self.check_for_bearer_token()
        if not c.user:
            if c.extension != 'rss':
                if not g.read_only_mode:
                    c.user = g.auth_provider.get_authenticated_account()
                    if c.user and c.user._deleted:
                        c.user = None
                else:
                    c.user = None
                c.user_is_loggedin = bool(c.user)
                admin_cookie = c.cookies.get(g.admin_cookie)
                if c.user_is_loggedin and admin_cookie:
                    (maybe_admin, first_login) = valid_admin_cookie(admin_cookie.value)
                    if maybe_admin:
                        self.enable_admin_mode(c.user, first_login=first_login)
                    else:
                        self.disable_admin_mode(c.user)
                otp_cookie = read_user_cookie(g.otp_cookie)
                if c.user_is_loggedin and otp_cookie:
                    is_otpcookie_valid = valid_otp_cookie(otp_cookie)
            if not c.user:
                c.user = UnloggedUser(get_browser_langs())
                if not isinstance(c.user.pref_lang, basestring):
                    c.user.pref_lang = g.lang
                    c.user._commit()
        if c.user_is_loggedin:
            self.set_up_user_context()
            c.modhash = generate_modhash()
            c.user_is_admin = maybe_admin and c.user.name in g.admins
            c.user_is_sponsor = c.user_is_admin or c.user.name in g.sponsors
            c.otp_cached = is_otpcookie_valid
        enforce_https()
        c.request_timer.intermediate('base-auth')
        self.run_sitewide_ratelimits()
        c.request_timer.intermediate('base-ratelimits')
        c.over18 = over18()
        set_obey_over18()
        set_multireddit()
        set_iface_lang()
        set_recent_clicks()
        set_colors()
        if not isinstance(c.site, FakeSubreddit):
            request.environ['REDDIT_NAME'] = c.site.name
        if c.site == Random:
            c.site = Subreddit.random_reddit(user=c.user)
            site_path = c.site.path.strip('/')
            path = '/' + site_path + request.path_qs
            abort(302, location=self.format_output_url(path))
        elif c.site == RandomSubscription:
            if not c.user.gold:
                abort(302, location=self.format_output_url('/gold/about'))
            c.site = Subreddit.random_subscription(c.user)
            site_path = c.site.path.strip('/')
            path = '/' + site_path + request.path_qs
            abort(302, location=self.format_output_url(path))
        elif c.site == RandomNSFW:
            c.site = Subreddit.random_reddit(over18=True, user=c.user)
            site_path = c.site.path.strip('/')
            path = '/' + site_path + request.path_qs
            abort(302, location=self.format_output_url(path))
        if not request.path.startswith('/api/login/'):
            if c.site.spammy() and (not c.user_is_admin) and (not c.error_page):
                ban_info = getattr(c.site, 'ban_info', {})
                if 'message' in ban_info and ban_info['message']:
                    message = ban_info['message']
                else:
                    message = None
                errpage = pages.InterstitialPage(_('banned'), content=pages.BannedInterstitial(message=message, ban_time=ban_info.get('banned_at')))
                request.environ['usable_error_content'] = errpage.render()
                self.abort404()
            if not c.site.can_view(c.user) and (not c.error_page) and (request.method != 'OPTIONS'):
                allowed_to_view = c.site.is_allowed_to_view(c.user)
                if isinstance(c.site, LabeledMulti):
                    self.abort404()
                elif not allowed_to_view and c.site.type == 'gold_only':
                    errpage = pages.InterstitialPage(_('gold members only'), content=pages.GoldOnlyInterstitial(sr_name=c.site.name, sr_description=c.site.public_description))
                    request.environ['usable_error_content'] = errpage.render()
                    self.abort403()
                elif not allowed_to_view:
                    errpage = pages.InterstitialPage(_('private'), content=pages.PrivateInterstitial(sr_name=c.site.name, sr_description=c.site.public_description))
                    request.environ['usable_error_content'] = errpage.render()
                    self.abort403()
                else:
                    if c.render_style != 'html':
                        self.abort403()
                    g.events.quarantine_event('quarantine_interstitial_view', c.site, request=request, context=c)
                    return self.intermediate_redirect('/quarantine', sr_path=False)
            if c.site.over_18 and (not c.over18) and (request.path != '/over18') and (c.render_style == 'html') and (not request.parsed_agent.bot):
                return self.intermediate_redirect('/over18', sr_path=False)
        c.allow_styles = True
        c.can_apply_styles = self.allow_stylesheets
        has_style_override = c.user.pref_default_theme_sr and feature.is_enabled('stylesheets_everywhere') and Subreddit._by_name(c.user.pref_default_theme_sr).can_view(c.user)
        sr_stylesheet_enabled = c.user.use_subreddit_style(c.site)
        if not sr_stylesheet_enabled and (not has_style_override):
            c.can_apply_styles = False
        c.bare_content = request.GET.pop('bare', False)
        c.show_admin_bar = admin_bar_eligible and (c.user_is_admin or g.debug)
        if not c.show_admin_bar:
            g.stats.end_logging_timings()
        hooks.get_hook('reddit.request.begin').call()
        c.request_timer.intermediate('base-pre')

    def post(self):
        if False:
            for i in range(10):
                print('nop')
        MinimalController.post(self)
        if response.content_type == 'text/html':
            self._embed_html_timing_data()
        if not c.cors_checked and request.method.upper() == 'GET' and (not c.user_is_loggedin) and (c.render_style == 'api'):
            response.headers['Access-Control-Allow-Origin'] = '*'
            request_origin = request.headers.get('Origin')
            if request_origin and request_origin != g.origin:
                g.stats.simple_event('cors.api_request')
                g.stats.count_string('origins', request_origin)
        if g.tracker_url and request.method.upper() == 'GET' and is_api():
            tracking_url = make_url_https(get_pageview_pixel_url())
            response.headers['X-Reddit-Tracking'] = tracking_url

    def _embed_html_timing_data(self):
        if False:
            print('Hello World!')
        timings = g.stats.end_logging_timings()
        if not timings or not c.show_admin_bar or c.is_exception_response:
            return
        timings = [{'key': timing.key, 'start': round(timing.start, 4), 'end': round(timing.end, 4)} for timing in timings]
        content = flatten_response(response.content)
        body_parts = list(content.rpartition('</body>'))
        if body_parts[1]:
            script = '<script type="text/javascript">window.r = window.r || {};r.timings = %s</script>' % simplejson.dumps(timings)
            body_parts.insert(1, script)
            response.content = ''.join(body_parts)

    def search_fail(self, exception):
        if False:
            for i in range(10):
                print('nop')
        errpage = pages.RedditError(_('search failed'), strings.search_failed)
        request.environ['usable_error_content'] = errpage.render()
        request.environ['retry_after'] = 60
        abort(503)