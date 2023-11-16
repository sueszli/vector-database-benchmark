import re
import hmac
import requests
from hashlib import sha256
from datetime import datetime
from datetime import timezone
from collections import OrderedDict
from xml.etree import ElementTree
from itertools import chain
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
IS_TOPIC = re.compile('^#?(?P<name>[A-Za-z0-9_-]+)\\s*$')
IS_REGION = re.compile('^\\s*(?P<country>[a-z]{2})-(?P<area>[a-z-]+?)-(?P<no>[0-9]+)\\s*$', re.I)
AWS_HTTP_ERROR_MAP = {403: 'Unauthorized - Invalid Access/Secret Key Combination.'}

class NotifySNS(NotifyBase):
    """
    A wrapper for AWS SNS (Amazon Simple Notification)
    """
    service_name = 'AWS Simple Notification Service (SNS)'
    service_url = 'https://aws.amazon.com/sns/'
    secure_protocol = 'sns'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_sns'
    request_rate_per_sec = 2.5
    body_maxlen = 160
    title_maxlen = 0
    templates = ('{schema}://{access_key_id}/{secret_access_key}/{region}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'access_key_id': {'name': _('Access Key ID'), 'type': 'string', 'private': True, 'required': True}, 'secret_access_key': {'name': _('Secret Access Key'), 'type': 'string', 'private': True, 'required': True}, 'region': {'name': _('Region'), 'type': 'string', 'required': True, 'regex': ('^[a-z]{2}-[a-z-]+?-[0-9]+$', 'i'), 'required': True, 'map_to': 'region_name'}, 'target_phone_no': {'name': _('Target Phone No'), 'type': 'string', 'map_to': 'targets', 'regex': ('^[0-9\\s)(+-]+$', 'i')}, 'target_topic': {'name': _('Target Topic'), 'type': 'string', 'map_to': 'targets', 'prefix': '#', 'regex': ('^[A-Za-z0-9_-]+$', 'i')}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'access': {'alias_of': 'access_key_id'}, 'secret': {'alias_of': 'secret_access_key'}, 'region': {'alias_of': 'region'}})

    def __init__(self, access_key_id, secret_access_key, region_name, targets=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Notify AWS SNS Object\n        '
        super().__init__(**kwargs)
        self.aws_access_key_id = validate_regex(access_key_id)
        if not self.aws_access_key_id:
            msg = 'An invalid AWS Access Key ID was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.aws_secret_access_key = validate_regex(secret_access_key)
        if not self.aws_secret_access_key:
            msg = 'An invalid AWS Secret Access Key ({}) was specified.'.format(secret_access_key)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.aws_region_name = validate_regex(region_name, *self.template_tokens['region']['regex'])
        if not self.aws_region_name:
            msg = 'An invalid AWS Region ({}) was specified.'.format(region_name)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.topics = list()
        self.phone = list()
        self.notify_url = 'https://sns.{}.amazonaws.com/'.format(self.aws_region_name)
        self.aws_service_name = 'sns'
        self.aws_canonical_uri = '/'
        self.aws_auth_version = 'AWS4'
        self.aws_auth_algorithm = 'AWS4-HMAC-SHA256'
        self.aws_auth_request = 'aws4_request'
        for target in parse_list(targets):
            result = is_phone_no(target)
            if result:
                self.phone.append('+{}'.format(result['full']))
                continue
            result = IS_TOPIC.match(target)
            if result:
                self.topics.append(result.group('name'))
                continue
            self.logger.warning('Dropped invalid phone/topic (%s) specified.' % target)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        wrapper to send_notification since we can alert more then one channel\n        '
        if len(self.phone) == 0 and len(self.topics) == 0:
            self.logger.warning('No AWS targets to notify.')
            return False
        error_count = 0
        phone = list(self.phone)
        topics = list(self.topics)
        while len(phone) > 0:
            no = phone.pop(0)
            payload = {'Action': u'Publish', 'Message': body, 'Version': u'2010-03-31', 'PhoneNumber': no}
            (result, _) = self._post(payload=payload, to=no)
            if not result:
                error_count += 1
        while len(topics):
            topic = topics.pop(0)
            payload = {'Action': u'CreateTopic', 'Version': u'2010-03-31', 'Name': topic}
            (result, response) = self._post(payload=payload, to=topic)
            if not result:
                error_count += 1
                continue
            topic_arn = response.get('topic_arn')
            if not topic_arn:
                error_count += 1
                continue
            payload = {'Action': u'Publish', 'Version': u'2010-03-31', 'TopicArn': topic_arn, 'Message': body}
            (result, _) = self._post(payload=payload, to=topic)
            if not result:
                error_count += 1
        return error_count == 0

    def _post(self, payload, to):
        if False:
            i = 10
            return i + 15
        "\n        Wrapper to request.post() to manage it's response better and make\n        the send() function cleaner and easier to maintain.\n\n        This function returns True if the _post was successful and False\n        if it wasn't.\n        "
        self.throttle()
        payload = NotifySNS.urlencode(payload)
        headers = self.aws_prepare_request(payload)
        self.logger.debug('AWS POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('AWS Payload: %s' % str(payload))
        try:
            r = requests.post(self.notify_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifySNS.http_response_code_lookup(r.status_code, AWS_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send AWS notification to {}: {}{}error={}.'.format(to, status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, NotifySNS.aws_response_to_dict(r.text))
            else:
                self.logger.info('Sent AWS notification to "%s".' % to)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending AWS notification to "%s".' % to)
            self.logger.debug('Socket Exception: %s' % str(e))
            return (False, NotifySNS.aws_response_to_dict(None))
        return (True, NotifySNS.aws_response_to_dict(r.text))

    def aws_prepare_request(self, payload, reference=None):
        if False:
            i = 10
            return i + 15
        '\n        Takes the intended payload and returns the headers for it.\n\n        The payload is presumed to have been already urlencoded()\n\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8', 'Content-Length': 0, 'Authorization': None, 'X-Amz-Date': None}
        reference = datetime.now(timezone.utc)
        headers['Content-Length'] = str(len(payload))
        amzdate = reference.strftime('%Y%m%dT%H%M%SZ')
        headers['X-Amz-Date'] = amzdate
        scope = '{date}/{region}/{service}/{request}'.format(date=reference.strftime('%Y%m%d'), region=self.aws_region_name, service=self.aws_service_name, request=self.aws_auth_request)
        signed_headers = OrderedDict([('content-type', headers['Content-Type']), ('host', '{service}.{region}.amazonaws.com'.format(service=self.aws_service_name, region=self.aws_region_name)), ('x-amz-date', headers['X-Amz-Date'])])
        canonical_request = '\n'.join([u'POST', self.aws_canonical_uri, '', '\n'.join(['%s:%s' % (k, v) for (k, v) in signed_headers.items()]) + '\n', ';'.join(signed_headers.keys()), sha256(payload.encode('utf-8')).hexdigest()])
        to_sign = '\n'.join([self.aws_auth_algorithm, amzdate, scope, sha256(canonical_request.encode('utf-8')).hexdigest()])
        headers['Authorization'] = ', '.join(['{algorithm} Credential={key}/{scope}'.format(algorithm=self.aws_auth_algorithm, key=self.aws_access_key_id, scope=scope), 'SignedHeaders={signed_headers}'.format(signed_headers=';'.join(signed_headers.keys())), 'Signature={signature}'.format(signature=self.aws_auth_signature(to_sign, reference))])
        return headers

    def aws_auth_signature(self, to_sign, reference):
        if False:
            print('Hello World!')
        '\n        Generates a AWS v4 signature based on provided payload\n        which should be in the form of a string.\n        '

        def _sign(key, msg, to_hex=False):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Perform AWS Signing\n            '
            if to_hex:
                return hmac.new(key, msg.encode('utf-8'), sha256).hexdigest()
            return hmac.new(key, msg.encode('utf-8'), sha256).digest()
        _date = _sign((self.aws_auth_version + self.aws_secret_access_key).encode('utf-8'), reference.strftime('%Y%m%d'))
        _region = _sign(_date, self.aws_region_name)
        _service = _sign(_region, self.aws_service_name)
        _signed = _sign(_service, self.aws_auth_request)
        return _sign(_signed, to_sign, to_hex=True)

    @staticmethod
    def aws_response_to_dict(aws_response):
        if False:
            return 10
        '\n        Takes an AWS Response object as input and returns it as a dictionary\n        but not befor extracting out what is useful to us first.\n\n        eg:\n          IN:\n            <CreateTopicResponse\n                  xmlns="http://sns.amazonaws.com/doc/2010-03-31/">\n              <CreateTopicResult>\n                <TopicArn>arn:aws:sns:us-east-1:000000000000:abcd</TopicArn>\n                   </CreateTopicResult>\n               <ResponseMetadata>\n               <RequestId>604bef0f-369c-50c5-a7a4-bbd474c83d6a</RequestId>\n               </ResponseMetadata>\n           </CreateTopicResponse>\n\n          OUT:\n           {\n              type: \'CreateTopicResponse\',\n              request_id: \'604bef0f-369c-50c5-a7a4-bbd474c83d6a\',\n              topic_arn: \'arn:aws:sns:us-east-1:000000000000:abcd\',\n           }\n        '
        aws_keep_map = {'RequestId': 'request_id', 'TopicArn': 'topic_arn', 'MessageId': 'message_id', 'Type': 'error_type', 'Code': 'error_code', 'Message': 'error_message'}
        response = {'type': None, 'request_id': None}
        try:
            root = ElementTree.fromstring(re.sub(' xmlns="[^"]+"', '', aws_response, count=1))
            response['type'] = str(root.tag)

            def _xml_iter(root, response):
                if False:
                    i = 10
                    return i + 15
                if len(root) > 0:
                    for child in root:
                        _xml_iter(child, response)
                elif root.tag in aws_keep_map.keys():
                    response[aws_keep_map[root.tag]] = root.text.strip()
            _xml_iter(root, response)
        except (ElementTree.ParseError, TypeError):
            pass
        return response

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        return '{schema}://{key_id}/{key_secret}/{region}/{targets}/?{params}'.format(schema=self.secure_protocol, key_id=self.pprint(self.aws_access_key_id, privacy, safe=''), key_secret=self.pprint(self.aws_secret_access_key, privacy, mode=PrivacyMode.Secret, safe=''), region=NotifySNS.quote(self.aws_region_name, safe=''), targets='/'.join([NotifySNS.quote(x) for x in chain(self.phone, ['#{}'.format(x) for x in self.topics])]), params=NotifySNS.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.phone) + len(self.topics)

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        access_key_id = NotifySNS.unquote(results['host'])
        secret_access_key = None
        region_name = None
        secret_access_key_parts = list()
        entries = NotifySNS.split_path(results['fullpath'])
        index = 0
        for (i, entry) in enumerate(entries):
            result = IS_REGION.match(entry)
            if result:
                secret_access_key = '/'.join(secret_access_key_parts)
                region_name = '{country}-{area}-{no}'.format(country=result.group('country').lower(), area=result.group('area').lower(), no=result.group('no'))
                index = i + 1
                break
            secret_access_key_parts.append(entry)
        results['targets'] = entries[index:]
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifySNS.parse_list(results['qsd']['to'])
        if 'secret' in results['qsd'] and len(results['qsd']['secret']):
            results['secret_access_key'] = NotifySNS.unquote(results['qsd']['secret'])
        else:
            results['secret_access_key'] = secret_access_key
        if 'access' in results['qsd'] and len(results['qsd']['access']):
            results['access_key_id'] = NotifySNS.unquote(results['qsd']['access'])
        else:
            results['access_key_id'] = access_key_id
        if 'region' in results['qsd'] and len(results['qsd']['region']):
            results['region_name'] = NotifySNS.unquote(results['qsd']['region'])
        else:
            results['region_name'] = region_name
        return results