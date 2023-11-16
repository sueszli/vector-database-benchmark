import re
from collections import defaultdict
from itertools import chain
import inspect
from os.path import abspath, relpath
from pylons import app_globals as g
from pylons.i18n import _
from reddit_base import RedditController
from r2.lib.utils import Storage
from r2.lib.pages import BoringPage, ApiHelp
from r2.lib.validator import validate, VOneOf
section_info = {'account': {'title': 'account'}, 'flair': {'title': 'flair'}, 'gold': {'title': 'reddit gold'}, 'links_and_comments': {'title': 'links & comments'}, 'messages': {'title': 'private messages'}, 'moderation': {'title': 'moderation'}, 'misc': {'title': 'misc'}, 'listings': {'title': 'listings'}, 'search': {'title': 'search'}, 'subreddits': {'title': 'subreddits'}, 'multis': {'title': 'multis'}, 'users': {'title': 'users'}, 'wiki': {'title': 'wiki'}, 'captcha': {'title': 'captcha'}}
api_section = Storage(((k, k) for k in section_info))

def api_doc(section, uses_site=False, **kwargs):
    if False:
        return 10
    '\n    Add documentation annotations to the decorated function.\n\n    See ApidocsController.docs_from_controller for a list of annotation fields.\n    '

    def add_metadata(api_function):
        if False:
            print('Hello World!')
        doc = api_function._api_doc = getattr(api_function, '_api_doc', {})
        if 'extends' in kwargs:
            kwargs['extends'] = kwargs['extends']._api_doc
        doc.update(kwargs)
        doc['uses_site'] = uses_site
        doc['section'] = section
        return api_function
    return add_metadata

class ApidocsController(RedditController):

    @staticmethod
    def docs_from_controller(controller, url_prefix='/api', oauth_only=False):
        if False:
            return 10
        '\n        Examines a controller for documentation.  A dictionary index of\n        sections containing dictionaries of URLs is returned.  For each URL, a\n        dictionary of HTTP methods (GET, POST, etc.) is contained.  For each\n        URL/method pair, a dictionary containing the following items is\n        available:\n\n        - `doc`: Markdown-formatted docstring.\n        - `uri`: Manually-specified URI to list the API method as\n        - `uri_variants`: Alternate URIs to access the API method from\n        - `supports_rss`: Indicates the URI also supports rss consumption\n        - `parameters`: Dictionary of possible parameter names and descriptions.\n        - `extends`: API method from which to inherit documentation\n        - `json_model`: The JSON model used instead of normal POST parameters\n        '
        api_docs = defaultdict(lambda : defaultdict(dict))
        for (name, func) in controller.__dict__.iteritems():
            (method, sep, action) = name.partition('_')
            if not action:
                continue
            valid_methods = ('GET', 'POST', 'PUT', 'DELETE', 'PATCH')
            api_doc = getattr(func, '_api_doc', None)
            if api_doc and 'section' in api_doc and (method in valid_methods):
                docs = {}
                docs['doc'] = inspect.getdoc(func)
                if 'extends' in api_doc:
                    docs.update(api_doc['extends'])
                    docs['parameters'] = {}
                docs.update(api_doc)
                if 'parameters' in api_doc:
                    docs['parameters'].pop('timeout', None)
                notes = docs.get('notes')
                if notes:
                    notes = '\n'.join(notes)
                    if docs['doc']:
                        docs['doc'] += '\n\n' + notes
                    else:
                        docs['doc'] = notes
                uri = docs.get('uri') or '/'.join((url_prefix, action))
                docs['uri'] = uri
                if 'supports_rss' not in docs:
                    docs['supports_rss'] = False
                if api_doc['uses_site']:
                    docs['in-subreddit'] = True
                oauth_perms = getattr(func, 'oauth2_perms', {})
                oauth_allowed = oauth_perms.get('oauth2_allowed', False)
                if not oauth_allowed:
                    docs['oauth_scopes'] = []
                else:
                    docs['oauth_scopes'] = oauth_perms['required_scopes'] or [None]
                if oauth_only:
                    if not oauth_allowed:
                        continue
                    for scope in docs['oauth_scopes']:
                        for variant in chain([uri], docs.get('uri_variants', [])):
                            api_docs[scope][variant][method] = docs
                else:
                    for variant in chain([uri], docs.get('uri_variants', [])):
                        api_docs[docs['section']][variant][method] = docs
        return api_docs

    @validate(mode=VOneOf('mode', options=('methods', 'oauth'), default='methods'))
    def GET_docs(self, mode):
        if False:
            return 10
        from r2.controllers.api import ApiController, ApiminimalController
        from r2.controllers.apiv1.user import APIv1UserController
        from r2.controllers.apiv1.gold import APIv1GoldController
        from r2.controllers.apiv1.scopes import APIv1ScopesController
        from r2.controllers.captcha import CaptchaController
        from r2.controllers.front import FrontController
        from r2.controllers.wiki import WikiApiController, WikiController
        from r2.controllers.multi import MultiApiController
        from r2.controllers import listingcontroller
        api_controllers = [(APIv1UserController, '/api/v1'), (APIv1GoldController, '/api/v1'), (APIv1ScopesController, '/api/v1'), (ApiController, '/api'), (ApiminimalController, '/api'), (WikiApiController, '/api/wiki'), (WikiController, '/wiki'), (MultiApiController, '/api/multi'), (CaptchaController, ''), (FrontController, '')]
        for (name, value) in vars(listingcontroller).iteritems():
            if name.endswith('Controller'):
                api_controllers.append((value, ''))
        api_controllers.extend(g.plugins.get_documented_controllers())
        api_docs = defaultdict(dict)
        oauth_index = defaultdict(set)
        for (controller, url_prefix) in api_controllers:
            controller_docs = self.docs_from_controller(controller, url_prefix, mode == 'oauth')
            for (section, contents) in controller_docs.iteritems():
                api_docs[section].update(contents)
                for (variant, method_dict) in contents.iteritems():
                    for (method, docs) in method_dict.iteritems():
                        for scope in docs['oauth_scopes']:
                            oauth_index[scope].add((section, variant, method))
        return BoringPage(_('api documentation'), content=ApiHelp(api_docs=api_docs, oauth_index=oauth_index, mode=mode), css_class='api-help', show_sidebar=False, show_infobar=False).render()