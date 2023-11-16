import requests
from json import loads
from datetime import timedelta
from datetime import datetime
from datetime import timezone
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyFormat
from ..common import NotifyType
from ..utils import parse_list
from ..utils import parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from .. import __title__, __version__
REDDIT_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token'}

class RedditMessageKind:
    """
    Define the kinds of messages supported
    """
    AUTO = 'auto'
    SELF = 'self'
    LINK = 'link'
REDDIT_MESSAGE_KINDS = (RedditMessageKind.AUTO, RedditMessageKind.SELF, RedditMessageKind.LINK)

class NotifyReddit(NotifyBase):
    """
    A wrapper for Notify Reddit Notifications
    """
    service_name = 'Reddit'
    service_url = 'https://reddit.com'
    secure_protocol = 'reddit'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_reddit'
    body_maxlen = 6000
    title_maxlen = 300
    notify_format = NotifyFormat.MARKDOWN
    auth_url = 'https://www.reddit.com/api/v1/access_token'
    submit_url = 'https://oauth.reddit.com/api/submit'
    request_rate_per_sec = 0
    clock_skew = timedelta(seconds=10)
    access_token_lifetime_sec = timedelta(seconds=3600)
    templates = ('{schema}://{user}:{password}@{app_id}/{app_secret}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'user': {'name': _('User Name'), 'type': 'string', 'required': True}, 'password': {'name': _('Password'), 'type': 'string', 'private': True, 'required': True}, 'app_id': {'name': _('Application ID'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9_-]+$', 'i')}, 'app_secret': {'name': _('Application Secret'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9_-]+$', 'i')}, 'target_subreddit': {'name': _('Target Subreddit'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'kind': {'name': _('Kind'), 'type': 'choice:string', 'values': REDDIT_MESSAGE_KINDS, 'default': RedditMessageKind.AUTO}, 'flair_id': {'name': _('Flair ID'), 'type': 'string', 'map_to': 'flair_id'}, 'flair_text': {'name': _('Flair Text'), 'type': 'string', 'map_to': 'flair_text'}, 'nsfw': {'name': _('NSFW'), 'type': 'bool', 'default': False, 'map_to': 'nsfw'}, 'ad': {'name': _('Is Ad?'), 'type': 'bool', 'default': False, 'map_to': 'advertisement'}, 'replies': {'name': _('Send Replies'), 'type': 'bool', 'default': True, 'map_to': 'sendreplies'}, 'spoiler': {'name': _('Is Spoiler'), 'type': 'bool', 'default': False, 'map_to': 'spoiler'}, 'resubmit': {'name': _('Resubmit Flag'), 'type': 'bool', 'default': False, 'map_to': 'resubmit'}})

    def __init__(self, app_id=None, app_secret=None, targets=None, kind=None, nsfw=False, sendreplies=True, resubmit=False, spoiler=False, advertisement=False, flair_id=None, flair_text=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Notify Reddit Object\n        '
        super().__init__(**kwargs)
        self.subreddits = set()
        self.nsfw = nsfw
        self.sendreplies = sendreplies
        self.spoiler = spoiler
        self.resubmit = resubmit
        self.advertisement = advertisement
        self.flair_id = flair_id
        self.flair_text = flair_text
        self.__refresh_token = None
        self.__access_token = None
        self.__access_token_expiry = datetime.now(timezone.utc)
        self.kind = kind.strip().lower() if isinstance(kind, str) else self.template_args['kind']['default']
        if self.kind not in REDDIT_MESSAGE_KINDS:
            msg = 'An invalid Reddit message kind ({}) was specified'.format(kind)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.user = validate_regex(self.user)
        if not self.user:
            msg = 'An invalid Reddit User ID ({}) was specified'.format(self.user)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.password = validate_regex(self.password)
        if not self.password:
            msg = 'An invalid Reddit Password ({}) was specified'.format(self.password)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.client_id = validate_regex(app_id, *self.template_tokens['app_id']['regex'])
        if not self.client_id:
            msg = 'An invalid Reddit App ID ({}) was specified'.format(app_id)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.client_secret = validate_regex(app_secret, *self.template_tokens['app_secret']['regex'])
        if not self.client_secret:
            msg = 'An invalid Reddit App Secret ({}) was specified'.format(app_secret)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.subreddits = [sr.lstrip('#') for sr in parse_list(targets) if sr.lstrip('#')]
        if not self.subreddits:
            self.logger.warning('No subreddits were identified to be notified')
        self.ratelimit_reset = datetime.now(timezone.utc).replace(tzinfo=None)
        self.ratelimit_remaining = 1.0
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'kind': self.kind, 'ad': 'yes' if self.advertisement else 'no', 'nsfw': 'yes' if self.nsfw else 'no', 'resubmit': 'yes' if self.resubmit else 'no', 'replies': 'yes' if self.sendreplies else 'no', 'spoiler': 'yes' if self.spoiler else 'no'}
        if self.flair_id:
            params['flair_id'] = self.flair_id
        if self.flair_text:
            params['flair_text'] = self.flair_text
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{user}:{password}@{app_id}/{app_secret}/{targets}/?{params}'.format(schema=self.secure_protocol, user=NotifyReddit.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''), app_id=self.pprint(self.client_id, privacy, mode=PrivacyMode.Secret, safe=''), app_secret=self.pprint(self.client_secret, privacy, mode=PrivacyMode.Secret, safe=''), targets='/'.join([NotifyReddit.quote(x, safe='') for x in self.subreddits]), params=NotifyReddit.urlencode(params))

    def __len__(self):
        if False:
            return 10
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.subreddits)

    def login(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A simple wrapper to authenticate with the Reddit Server\n        '
        payload = {'grant_type': 'password', 'username': self.user, 'password': self.password}
        self.__access_token = False
        (postokay, response) = self._fetch(self.auth_url, payload=payload)
        if not postokay or not response:
            self.__access_token = False
            return False
        self.__access_token = response.get('access_token')
        if 'expires_in' in response:
            delta = timedelta(seconds=int(response['expires_in']))
            self.__access_token_expiry = delta + datetime.now(timezone.utc) - self.clock_skew
        else:
            self.__access_token_expiry = self.access_token_lifetime_sec + datetime.now(timezone.utc) - self.clock_skew
        self.__refresh_token = response.get('refresh_token', self.__refresh_token)
        if self.__access_token:
            self.logger.info('Authenticated to Reddit as {}'.format(self.user))
            return True
        self.logger.warning('Failed to authenticate to Reddit as {}'.format(self.user))
        return False

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Reddit Notification\n        '
        has_error = False
        if not self.__access_token and (not self.login()):
            return False
        if not len(self.subreddits):
            self.logger.warning('There are no Reddit targets to notify')
            return False
        if self.kind == RedditMessageKind.AUTO:
            parsed = NotifyBase.parse_url(body)
            if parsed and parsed.get('schema', '').startswith('http') and parsed.get('host'):
                kind = RedditMessageKind.LINK
            else:
                kind = RedditMessageKind.SELF
        else:
            kind = self.kind
        subreddits = list(self.subreddits)
        while len(subreddits) > 0:
            subreddit = subreddits.pop()
            payload = {'ad': True if self.advertisement else False, 'api_type': 'json', 'extension': 'json', 'sr': subreddit, 'title': title if title else self.app_desc, 'kind': kind, 'nsfw': True if self.nsfw else False, 'resubmit': True if self.resubmit else False, 'sendreplies': True if self.sendreplies else False, 'spoiler': True if self.spoiler else False}
            if self.flair_id:
                payload['flair_id'] = self.flair_id
            if self.flair_text:
                payload['flair_text'] = self.flair_text
            if kind == RedditMessageKind.LINK:
                payload.update({'url': body})
            else:
                payload.update({'text': body})
            (postokay, response) = self._fetch(self.submit_url, payload=payload)
            if not postokay:
                has_error = True
                continue
            self.logger.info('Sent Reddit notification to {}'.format(subreddit))
        return not has_error

    def _fetch(self, url, payload=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrapper to Reddit API requests object\n        '
        headers = {'User-Agent': '{} v{}'.format(__title__, __version__)}
        if self.__access_token:
            headers['Authorization'] = 'Bearer {}'.format(self.__access_token)
        url = self.submit_url if self.__access_token else self.auth_url
        self.logger.debug('Reddit POST URL: {} (cert_verify={})'.format(url, self.verify_certificate))
        self.logger.debug('Reddit Payload: %s' % str(payload))
        wait = None
        if self.ratelimit_remaining <= 0.0:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if now < self.ratelimit_reset:
                wait = abs((self.ratelimit_reset - now + self.clock_skew).total_seconds())
        self.throttle(wait=wait)
        content = {}
        try:
            r = requests.post(url, data=payload, auth=None if self.__access_token else (self.client_id, self.client_secret), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok and self.__access_token and (url != self.auth_url):
                status_str = NotifyReddit.http_response_code_lookup(r.status_code, REDDIT_HTTP_ERROR_MAP)
                self.logger.debug('Taking countermeasures after failed to send to Reddit {}: {}error={}'.format(url, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                if not self.login():
                    return (False, {})
                r = requests.post(url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                content = loads(r.content)
            except (TypeError, ValueError, AttributeError):
                status_str = NotifyReddit.http_response_code_lookup(r.status_code, REDDIT_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send to Reddit after countermeasures {}: {}error={}'.format(url, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, {})
            if r.status_code != requests.codes.ok:
                status_str = NotifyReddit.http_response_code_lookup(r.status_code, REDDIT_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send to Reddit {}: {}error={}'.format(url, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, content)
            errors = [] if not content else content.get('json', {}).get('errors', [])
            if errors:
                self.logger.warning('Failed to send to Reddit {}: {}'.format(url, str(errors)))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, content)
            try:
                self.ratelimit_remaining = float(r.headers.get('X-RateLimit-Remaining'))
                self.ratelimit_reset = datetime.fromtimestamp(int(r.headers.get('X-RateLimit-Reset')), timezone.utc).replace(tzinfo=None)
            except (TypeError, ValueError):
                pass
        except requests.RequestException as e:
            self.logger.warning('Exception received when sending Reddit to {}'.format(url))
            self.logger.debug('Socket Exception: %s' % str(e))
            return (False, content)
        return (True, content)

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifyReddit.split_path(results['fullpath'])
        if 'kind' in results['qsd'] and results['qsd']['kind']:
            results['kind'] = NotifyReddit.unquote(results['qsd']['kind'].strip().lower())
        else:
            results['kind'] = RedditMessageKind.AUTO
        results['ad'] = parse_bool(results['qsd'].get('ad', False))
        results['nsfw'] = parse_bool(results['qsd'].get('nsfw', False))
        results['replies'] = parse_bool(results['qsd'].get('replies', True))
        results['resubmit'] = parse_bool(results['qsd'].get('resubmit', False))
        results['spoiler'] = parse_bool(results['qsd'].get('spoiler', False))
        if 'flair_text' in results['qsd']:
            results['flair_text'] = NotifyReddit.unquote(results['qsd']['flair_text'])
        if 'flair_id' in results['qsd']:
            results['flair_id'] = NotifyReddit.unquote(results['qsd']['flair_id'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyReddit.parse_list(results['qsd']['to'])
        if 'app_id' in results['qsd']:
            results['app_id'] = NotifyReddit.unquote(results['qsd']['app_id'])
        else:
            results['app_id'] = NotifyReddit.unquote(results['host'])
        if 'app_secret' in results['qsd']:
            results['app_secret'] = NotifyReddit.unquote(results['qsd']['app_secret'])
        else:
            results['app_secret'] = None if not results['targets'] else results['targets'].pop(0)
        return results