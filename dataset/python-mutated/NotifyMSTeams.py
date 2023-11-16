import re
import requests
import json
from json.decoder import JSONDecodeError
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyType
from ..common import NotifyFormat
from ..utils import parse_bool
from ..utils import validate_regex
from ..utils import apply_template
from ..utils import TemplateType
from ..AppriseAttachment import AppriseAttachment
from ..AppriseLocale import gettext_lazy as _

class NotifyMSTeams(NotifyBase):
    """
    A wrapper for Microsoft Teams Notifications
    """
    service_name = 'MSTeams'
    service_url = 'https://teams.micrsoft.com/'
    secure_protocol = 'msteams'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_msteams'
    notify_url_v1 = 'https://outlook.office.com/webhook/{token_a}/IncomingWebhook/{token_b}/{token_c}'
    notify_url_v2 = 'https://{team}.webhook.office.com/webhookb2/{token_a}/IncomingWebhook/{token_b}/{token_c}'
    image_size = NotifyImageSize.XY_72
    body_maxlen = 1000
    notify_format = NotifyFormat.MARKDOWN
    max_msteams_template_size = 35000
    templates = ('{schema}://{team}/{token_a}/{token_b}/{token_c}', '{schema}://{token_a}/{token_b}/{token_c}')
    template_tokens = dict(NotifyBase.template_tokens, **{'team': {'name': _('Team Name'), 'type': 'string', 'required': True, 'regex': ('^[A-Z0-9_-]+$', 'i')}, 'token_a': {'name': _('Token A'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[A-Z0-9-]+@[A-Z0-9-]+$', 'i')}, 'token_b': {'name': _('Token B'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9]+$', 'i')}, 'token_c': {'name': _('Token C'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9-]+$', 'i')}})
    template_args = dict(NotifyBase.template_args, **{'image': {'name': _('Include Image'), 'type': 'bool', 'default': False, 'map_to': 'include_image'}, 'version': {'name': _('Version'), 'type': 'choice:int', 'values': (1, 2), 'default': 2}, 'template': {'name': _('Template Path'), 'type': 'string', 'private': True}})
    template_kwargs = {'tokens': {'name': _('Template Tokens'), 'prefix': ':'}}

    def __init__(self, token_a, token_b, token_c, team=None, version=None, include_image=True, template=None, tokens=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Microsoft Teams Object\n\n        You can optional specify a template and identify arguments you\n        wish to populate your template with when posting.  Some reserved\n        template arguments that can not be over-ridden are:\n           `body`, `title`, and `type`.\n        '
        super().__init__(**kwargs)
        try:
            self.version = int(version)
        except TypeError:
            self.version = self.template_args['version']['default']
        except ValueError:
            self.version = None
        if self.version not in self.template_args['version']['values']:
            msg = 'An invalid MSTeams Version ({}) was specified.'.format(version)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.team = validate_regex(team)
        if not self.team:
            NotifyBase.logger.deprecate('Apprise requires you to identify your Microsoft Team name as part of the URL. e.g.: msteams://TEAM-NAME/{token_a}/{token_b}/{token_c}')
            self.team = 'outlook'
        self.token_a = validate_regex(token_a, *self.template_tokens['token_a']['regex'])
        if not self.token_a:
            msg = 'An invalid MSTeams (first) Token ({}) was specified.'.format(token_a)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.token_b = validate_regex(token_b, *self.template_tokens['token_b']['regex'])
        if not self.token_b:
            msg = 'An invalid MSTeams (second) Token ({}) was specified.'.format(token_b)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.token_c = validate_regex(token_c, *self.template_tokens['token_c']['regex'])
        if not self.token_c:
            msg = 'An invalid MSTeams (third) Token ({}) was specified.'.format(token_c)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.include_image = include_image
        self.template = AppriseAttachment(asset=self.asset)
        if template:
            self.template.add(template)
            self.template[0].max_file_size = self.max_msteams_template_size
        self.tokens = {}
        if isinstance(tokens, dict):
            self.tokens.update(tokens)
        elif tokens:
            msg = 'The specified MSTeams Template Tokens ({}) are not identified as a dictionary.'.format(tokens)
            self.logger.warning(msg)
            raise TypeError(msg)
        return

    def gen_payload(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        This function generates our payload whether it be the generic one\n        Apprise generates by default, or one provided by a specified\n        external template.\n        '
        image_url = None if not self.include_image else self.image_url(notify_type)
        if not self.template:
            payload = {'@type': 'MessageCard', '@context': 'https://schema.org/extensions', 'summary': self.app_desc, 'themeColor': self.color(notify_type), 'sections': [{'activityImage': None, 'activityTitle': title, 'text': body}]}
            if image_url:
                payload['sections'][0]['activityImage'] = image_url
            return payload
        template = self.template[0]
        if not template:
            self.logger.error('Could not access MSTeam template {}.'.format(template.url(privacy=True)))
            return False
        tokens = self.tokens.copy()
        tokens['app_body'] = body
        tokens['app_title'] = title
        tokens['app_type'] = notify_type
        tokens['app_id'] = self.app_id
        tokens['app_desc'] = self.app_desc
        tokens['app_color'] = self.color(notify_type)
        tokens['app_image_url'] = image_url
        tokens['app_url'] = self.app_url
        tokens['app_mode'] = TemplateType.JSON
        try:
            with open(template.path, 'r') as fp:
                content = json.loads(apply_template(fp.read(), **tokens))
        except (OSError, IOError):
            self.logger.error('MSTeam template {} could not be read.'.format(template.url(privacy=True)))
            return None
        except JSONDecodeError as e:
            self.logger.error('MSTeam template {} contains invalid JSON.'.format(template.url(privacy=True)))
            self.logger.debug('JSONDecodeError: {}'.format(e))
            return None
        has_error = False
        if '@type' not in content:
            self.logger.error('MSTeam template {} is missing @type kwarg.'.format(template.url(privacy=True)))
            has_error = True
        if '@context' not in content:
            self.logger.error('MSTeam template {} is missing @context kwarg.'.format(template.url(privacy=True)))
            has_error = True
        return content if not has_error else None

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Microsoft Teams Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        notify_url = self.notify_url_v2.format(team=self.team, token_a=self.token_a, token_b=self.token_b, token_c=self.token_c) if self.version > 1 else self.notify_url_v1.format(token_a=self.token_a, token_b=self.token_b, token_c=self.token_c)
        payload = self.gen_payload(body=body, title=title, notify_type=notify_type, **kwargs)
        if not payload:
            return False
        self.logger.debug('MSTeams POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('MSTeams Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, data=json.dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyMSTeams.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send MSTeams notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent MSTeams notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending MSTeams notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no'}
        if self.version != self.template_args['version']['default']:
            params['version'] = str(self.version)
        if self.template:
            params['template'] = NotifyMSTeams.quote(self.template[0].url(), safe='')
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        params.update({':{}'.format(k): v for (k, v) in self.tokens.items()})
        if self.version > 1:
            return '{schema}://{team}/{token_a}/{token_b}/{token_c}/?{params}'.format(schema=self.secure_protocol, team=NotifyMSTeams.quote(self.team, safe=''), token_a=self.pprint(self.token_a, privacy, safe=''), token_b=self.pprint(self.token_b, privacy, safe=''), token_c=self.pprint(self.token_c, privacy, safe=''), params=NotifyMSTeams.urlencode(params))
        else:
            return '{schema}://{token_a}/{token_b}/{token_c}/?{params}'.format(schema=self.secure_protocol, token_a=self.pprint(self.token_a, privacy, safe='@'), token_b=self.pprint(self.token_b, privacy, safe=''), token_c=self.pprint(self.token_c, privacy, safe=''), params=NotifyMSTeams.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        entries = NotifyMSTeams.split_path(results['fullpath'])
        if results.get('user'):
            results['token_a'] = '{}@{}'.format(NotifyMSTeams.unquote(results['user']), NotifyMSTeams.unquote(results['host']))
        else:
            results['team'] = NotifyMSTeams.unquote(results['host'])
            results['token_a'] = None if not entries else NotifyMSTeams.unquote(entries.pop(0))
        results['token_b'] = None if not entries else NotifyMSTeams.unquote(entries.pop(0))
        results['token_c'] = None if not entries else NotifyMSTeams.unquote(entries.pop(0))
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        if 'team' in results['qsd'] and results['qsd']['team']:
            results['team'] = NotifyMSTeams.unquote(results['qsd']['team'])
        if 'template' in results['qsd'] and results['qsd']['template']:
            results['template'] = NotifyMSTeams.unquote(results['qsd']['template'])
        if 'version' in results['qsd'] and results['qsd']['version']:
            results['version'] = NotifyMSTeams.unquote(results['qsd']['version'])
        else:
            results['version'] = 1 if not results.get('team') else 2
        results['tokens'] = results['qsd:']
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Legacy Support:\n            https://outlook.office.com/webhook/ABCD/IncomingWebhook/DEFG/HIJK\n\n        New Hook Support:\n            https://team-name.office.com/webhook/ABCD/IncomingWebhook/DEFG/HIJK\n        '
        result = re.match('^https?://(?P<team>[^.]+)(?P<v2a>\\.webhook)?\\.office\\.com/webhook(?P<v2b>b2)?/(?P<token_a>[A-Z0-9-]+@[A-Z0-9-]+)/IncomingWebhook/(?P<token_b>[A-Z0-9]+)/(?P<token_c>[A-Z0-9-]+)/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            if result.group('v2a'):
                return NotifyMSTeams.parse_url('{schema}://{team}/{token_a}/{token_b}/{token_c}/{params}'.format(schema=NotifyMSTeams.secure_protocol, team=result.group('team'), token_a=result.group('token_a'), token_b=result.group('token_b'), token_c=result.group('token_c'), params='' if not result.group('params') else result.group('params')))
            else:
                return NotifyMSTeams.parse_url('{schema}://{token_a}/{token_b}/{token_c}/{params}'.format(schema=NotifyMSTeams.secure_protocol, token_a=result.group('token_a'), token_b=result.group('token_b'), token_c=result.group('token_c'), params='' if not result.group('params') else result.group('params')))
        return None