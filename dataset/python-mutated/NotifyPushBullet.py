import requests
from json import dumps
from json import loads
from .NotifyBase import NotifyBase
from ..utils import is_email
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from ..attachment.AttachBase import AttachBase
PUSHBULLET_SEND_TO_ALL = 'ALL_DEVICES'
PUSHBULLET_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token.'}

class NotifyPushBullet(NotifyBase):
    """
    A wrapper for PushBullet Notifications
    """
    service_name = 'Pushbullet'
    service_url = 'https://www.pushbullet.com/'
    secure_protocol = 'pbul'
    request_rate_per_sec = 1.2
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_pushbullet'
    notify_url = 'https://api.pushbullet.com/v2/{}'
    attachment_support = True
    templates = ('{schema}://{accesstoken}', '{schema}://{accesstoken}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'accesstoken': {'name': _('Access Token'), 'type': 'string', 'private': True, 'required': True}, 'target_device': {'name': _('Target Device'), 'type': 'string', 'map_to': 'targets'}, 'target_channel': {'name': _('Target Channel'), 'type': 'string', 'prefix': '#', 'map_to': 'targets'}, 'target_email': {'name': _('Target Email'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}})

    def __init__(self, accesstoken, targets=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize PushBullet Object\n        '
        super().__init__(**kwargs)
        self.accesstoken = validate_regex(accesstoken)
        if not self.accesstoken:
            msg = 'An invalid PushBullet Access Token ({}) was specified.'.format(accesstoken)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = parse_list(targets)
        if len(self.targets) == 0:
            self.targets = (PUSHBULLET_SEND_TO_ALL,)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform PushBullet Notification\n        '
        has_error = False
        attachments = []
        if attach and self.attachment_support:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                self.logger.debug('Preparing PushBullet attachment {}'.format(attachment.url(privacy=True)))
                payload = {'file_name': attachment.name, 'file_type': attachment.mimetype}
                (okay, response) = self._send(self.notify_url.format('upload-request'), payload)
                if not okay:
                    return False
                try:
                    payload = {'type': 'file', 'file_name': response['file_name'], 'file_type': response['file_type'], 'file_url': response['file_url']}
                    if response['file_type'].startswith('image/'):
                        payload['image_url'] = response['file_url']
                    upload_url = response['upload_url']
                except (KeyError, TypeError):
                    return False
                (okay, response) = self._send(upload_url, attachment)
                if not okay:
                    return False
                attachments.append(payload)
        targets = list(self.targets)
        while len(targets):
            recipient = targets.pop(0)
            payload = {'type': 'note', 'title': title, 'body': body}
            match = is_email(recipient)
            if match:
                payload['email'] = match['full_email']
                self.logger.debug('PushBullet recipient {} parsed as an email address'.format(recipient))
            elif recipient is PUSHBULLET_SEND_TO_ALL:
                pass
            elif recipient[0] == '#':
                payload['channel_tag'] = recipient[1:]
                self.logger.debug('PushBullet recipient {} parsed as a channel'.format(recipient))
            else:
                payload['device_iden'] = recipient
                self.logger.debug('PushBullet recipient {} parsed as a device'.format(recipient))
            if body:
                (okay, response) = self._send(self.notify_url.format('pushes'), payload)
                if not okay:
                    has_error = True
                    continue
                self.logger.info('Sent PushBullet notification to "%s".' % recipient)
            for attach_payload in attachments:
                (okay, response) = self._send(self.notify_url.format('pushes'), attach_payload)
                if not okay:
                    has_error = True
                    continue
                self.logger.info('Sent PushBullet attachment ({}) to "{}".'.format(attach_payload['file_name'], recipient))
        return not has_error

    def _send(self, url, payload, **kwargs):
        if False:
            return 10
        '\n        Wrapper to the requests (post) object\n        '
        headers = {'User-Agent': self.app_id}
        files = None
        data = None
        if not isinstance(payload, AttachBase):
            headers['Content-Type'] = 'application/json'
            data = dumps(payload) if payload else None
        auth = (self.accesstoken, '')
        self.logger.debug('PushBullet POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('PushBullet Payload: %s' % str(payload))
        self.throttle()
        response = None
        try:
            if isinstance(payload, AttachBase):
                files = {'file': (payload.name, open(payload.path, 'rb'))}
            r = requests.post(url, data=data, headers=headers, files=files, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                response = loads(r.content)
            except (AttributeError, TypeError, ValueError):
                response = r.content
            if r.status_code not in (requests.codes.ok, requests.codes.no_content):
                status_str = NotifyPushBullet.http_response_code_lookup(r.status_code, PUSHBULLET_HTTP_ERROR_MAP)
                self.logger.warning('Failed to deliver payload to PushBullet:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, response)
            return (True, response)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred communicating with PushBullet.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return (False, response)
        except (OSError, IOError) as e:
            self.logger.warning('An I/O error occurred while handling {}.'.format(payload.name if isinstance(payload, AttachBase) else payload))
            self.logger.debug('I/O Exception: %s' % str(e))
            return (False, response)
        finally:
            if files:
                files['file'][1].close()

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        targets = '/'.join([NotifyPushBullet.quote(x) for x in self.targets])
        if targets == PUSHBULLET_SEND_TO_ALL:
            targets = ''
        return '{schema}://{accesstoken}/{targets}/?{params}'.format(schema=self.secure_protocol, accesstoken=self.pprint(self.accesstoken, privacy, safe=''), targets=targets, params=NotifyPushBullet.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.targets)

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifyPushBullet.split_path(results['fullpath'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyPushBullet.parse_list(results['qsd']['to'])
        results['accesstoken'] = NotifyPushBullet.unquote(results['host'])
        return results