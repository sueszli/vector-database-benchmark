import collections
import copy
from urllib3.util import parse_url
from warehouse.config import Environment
SELF = "'self'"
NONE = "'none'"

def _serialize(policy):
    if False:
        print('Hello World!')
    return '; '.join([' '.join([k] + [v2 for v2 in v if v2 is not None]) for (k, v) in sorted(policy.items())])

def content_security_policy_tween_factory(handler, registry):
    if False:
        i = 10
        return i + 15

    def content_security_policy_tween(request):
        if False:
            while True:
                i = 10
        resp = handler(request)
        try:
            policy = request.find_service(name='csp')
        except LookupError:
            policy = collections.defaultdict(list)
        if request.path.startswith('/simple/'):
            policy = collections.defaultdict(list)
            policy['sandbox'] = ['allow-top-navigation']
            policy['default-src'] = [NONE]
        policy = _serialize(policy).format(request=request)
        if not request.path.startswith('/_debug_toolbar/') and policy:
            resp.headers['Content-Security-Policy'] = policy
        return resp
    return content_security_policy_tween

class CSPPolicy(collections.defaultdict):

    def __init__(self, policy=None):
        if False:
            print('Hello World!')
        super().__init__(list, policy or {})

    def merge(self, policy):
        if False:
            while True:
                i = 10
        for (key, attrs) in policy.items():
            self[key].extend(attrs)
            if NONE in self[key] and len(self[key]) > 1:
                self[key].remove(NONE)

def csp_factory(_, request):
    if False:
        i = 10
        return i + 15
    try:
        return CSPPolicy(copy.deepcopy(request.registry.settings['csp']))
    except KeyError:
        return CSPPolicy({})

def _connect_src_settings(config) -> list:
    if False:
        i = 10
        return i + 15
    settings = [SELF, 'https://api.github.com/repos/', 'https://api.github.com/search/issues', 'https://*.google-analytics.com', 'https://*.analytics.google.com', 'https://*.googletagmanager.com', 'fastly-insights.com', '*.fastly-insights.com', '*.ethicalads.io', 'https://api.pwnedpasswords.com', 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/sre/mathmaps/']
    settings.extend([item for item in [config.registry.settings.get('statuspage.url')] if item])
    if config.registry.settings.get('warehouse.env') == Environment.development:
        livereload_url = config.registry.settings.get('livereload.url')
        parsed_url = parse_url(livereload_url)
        scheme_replacement = 'wss' if parsed_url.scheme == 'https' else 'ws'
        replaced = parsed_url._replace(scheme=scheme_replacement)
        settings.extend([f'{replaced.url}/livereload'])
    return settings

def _script_src_settings(config) -> list:
    if False:
        while True:
            i = 10
    settings = [SELF, 'https://*.googletagmanager.com', 'https://www.google-analytics.com', 'https://ssl.google-analytics.com', '*.fastly-insights.com', '*.ethicalads.io', "'sha256-U3hKDidudIaxBDEzwGJApJgPEf2mWk6cfMWghrAa6i0='", 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/', "'sha256-1CldwzdEg2k1wTmf7s5RWVd7NMXI/7nxxjJM2C4DqII='", "'sha256-0POaN8stWYQxhzjKS+/eOfbbJ/u4YHO5ZagJvLpMypo='"]
    if config.registry.settings.get('warehouse.env') == Environment.development:
        settings.extend([f"{config.registry.settings['livereload.url']}/livereload.js"])
    return settings

def includeme(config):
    if False:
        for i in range(10):
            print('nop')
    config.register_service_factory(csp_factory, name='csp')
    config.add_settings({'csp': {'base-uri': [SELF], 'block-all-mixed-content': [], 'connect-src': _connect_src_settings(config), 'default-src': [NONE], 'font-src': [SELF, 'fonts.gstatic.com'], 'form-action': [SELF, 'https://checkout.stripe.com'], 'frame-ancestors': [NONE], 'frame-src': [NONE], 'img-src': [SELF, config.registry.settings['camo.url'], 'https://*.google-analytics.com', 'https://*.googletagmanager.com', '*.fastly-insights.com', '*.ethicalads.io'], 'script-src': _script_src_settings(config), 'style-src': [SELF, 'fonts.googleapis.com', '*.ethicalads.io', "'sha256-2YHqZokjiizkHi1Zt+6ar0XJ0OeEy/egBnlm+MDMtrM='", "'sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU='", "'sha256-JLEjeN9e5dGsz5475WyRaoA4eQOdNPxDIeUhclnJDCE='", "'sha256-mQyxHEuwZJqpxCw3SLmc4YOySNKXunyu2Oiz1r3/wAE='", "'sha256-OCf+kv5Asiwp++8PIevKBYSgnNLNUZvxAp4a7wMLuKA='", "'sha256-h5LOiLhk6wiJrGsG5ItM0KimwzWQH/yAcmoJDJL//bY='"], 'worker-src': ['*.fastly-insights.com']}})
    config.add_tween('warehouse.csp.content_security_policy_tween_factory')