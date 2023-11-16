import asyncio
import re
from functools import partial
from ..URLBase import URLBase
from ..common import NotifyType
from ..common import NOTIFY_TYPES
from ..common import NotifyFormat
from ..common import NOTIFY_FORMATS
from ..common import OverflowMode
from ..common import OVERFLOW_MODES
from ..AppriseLocale import gettext_lazy as _
from ..AppriseAttachment import AppriseAttachment

class NotifyBase(URLBase):
    """
    This is the base class for all notification services
    """
    enabled = True
    category = 'native'
    requirements = {'details': None, 'packages_required': [], 'packages_recommended': []}
    service_url = None
    setup_url = None
    request_rate_per_sec = 5.5
    image_size = None
    body_maxlen = 32768
    title_maxlen = 250
    body_max_line_count = 0
    notify_format = NotifyFormat.TEXT
    overflow_mode = OverflowMode.UPSTREAM
    attachment_support = False
    default_html_tag_id = 'b'
    template_args = dict(URLBase.template_args, **{'overflow': {'name': _('Overflow Mode'), 'type': 'choice:string', 'values': OVERFLOW_MODES, 'default': overflow_mode, '_lookup_default': 'overflow_mode'}, 'format': {'name': _('Notify Format'), 'type': 'choice:string', 'values': NOTIFY_FORMATS, 'default': notify_format, '_lookup_default': 'notify_format'}})

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize some general configuration that will keep things consistent\n        when working with the notifiers that will inherit this class.\n\n        '
        super().__init__(**kwargs)
        if 'format' in kwargs:
            notify_format = kwargs.get('format', '')
            if notify_format.lower() not in NOTIFY_FORMATS:
                msg = 'Invalid notification format {}'.format(notify_format)
                self.logger.error(msg)
                raise TypeError(msg)
            self.notify_format = notify_format
        if 'overflow' in kwargs:
            overflow = kwargs.get('overflow', '')
            if overflow.lower() not in OVERFLOW_MODES:
                msg = 'Invalid overflow method {}'.format(overflow)
                self.logger.error(msg)
                raise TypeError(msg)
            self.overflow_mode = overflow

    def image_url(self, notify_type, logo=False, extension=None, image_size=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns Image URL if possible\n        '
        if not self.image_size:
            return None
        if notify_type not in NOTIFY_TYPES:
            return None
        return self.asset.image_url(notify_type=notify_type, image_size=self.image_size if image_size is None else image_size, logo=logo, extension=extension)

    def image_path(self, notify_type, extension=None):
        if False:
            return 10
        '\n        Returns the path of the image if it can\n        '
        if not self.image_size:
            return None
        if notify_type not in NOTIFY_TYPES:
            return None
        return self.asset.image_path(notify_type=notify_type, image_size=self.image_size, extension=extension)

    def image_raw(self, notify_type, extension=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the raw image if it can\n        '
        if not self.image_size:
            return None
        if notify_type not in NOTIFY_TYPES:
            return None
        return self.asset.image_raw(notify_type=notify_type, image_size=self.image_size, extension=extension)

    def color(self, notify_type, color_type=None):
        if False:
            while True:
                i = 10
        '\n        Returns the html color (hex code) associated with the notify_type\n        '
        if notify_type not in NOTIFY_TYPES:
            return None
        return self.asset.color(notify_type=notify_type, color_type=color_type)

    def notify(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Performs notification\n        '
        try:
            send_calls = list(self._build_send_calls(*args, **kwargs))
        except TypeError:
            return False
        else:
            the_calls = [self.send(**kwargs2) for kwargs2 in send_calls]
            return all(the_calls)

    async def async_notify(self, *args, **kwargs):
        """
        Performs notification for asynchronous callers
        """
        try:
            send_calls = list(self._build_send_calls(*args, **kwargs))
        except TypeError:
            return False
        else:
            loop = asyncio.get_event_loop()

            async def do_send(**kwargs2):
                send = partial(self.send, **kwargs2)
                result = await loop.run_in_executor(None, send)
                return result
            the_cors = (do_send(**kwargs2) for kwargs2 in send_calls)
            return all(await asyncio.gather(*the_cors))

    def _build_send_calls(self, body=None, title=None, notify_type=NotifyType.INFO, overflow=None, attach=None, body_format=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Get a list of dictionaries that can be used to call send() or\n        (in the future) async_send().\n        '
        if not self.enabled:
            msg = f'{self.service_name} is currently disabled on this system.'
            self.logger.warning(msg)
            raise TypeError(msg)
        if attach is not None and (not isinstance(attach, AppriseAttachment)):
            try:
                attach = AppriseAttachment(attach, asset=self.asset)
            except TypeError:
                raise
            body = '' if not body else body
        elif not (body or attach):
            msg = 'No message body or attachment was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        if not body and (not self.attachment_support):
            msg = f'{self.service_name} does not support attachments;  service skipped'
            self.logger.warning(msg)
            raise TypeError(msg)
        title = '' if not title else title
        for chunk in self._apply_overflow(body=body, title=title, overflow=overflow, body_format=body_format):
            yield dict(body=chunk['body'], title=chunk['title'], notify_type=notify_type, attach=attach, body_format=body_format)

    def _apply_overflow(self, body, title=None, overflow=None, body_format=None):
        if False:
            print('Hello World!')
        "\n        Takes the message body and title as input.  This function then\n        applies any defined overflow restrictions associated with the\n        notification service and may alter the message if/as required.\n\n        The function will always return a list object in the following\n        structure:\n            [\n                {\n                    title: 'the title goes here',\n                    body: 'the message body goes here',\n                },\n                {\n                    title: 'the title goes here',\n                    body: 'the message body goes here',\n                },\n\n            ]\n        "
        response = list()
        title = '' if not title else title.strip()
        body = '' if not body else body.rstrip()
        if overflow is None:
            overflow = self.overflow_mode
        if self.title_maxlen <= 0 and len(title) > 0:
            if self.notify_format == NotifyFormat.HTML:
                body = '<{open_tag}>{title}</{close_tag}><br />\r\n{body}'.format(open_tag=self.default_html_tag_id, title=title, close_tag=self.default_html_tag_id, body=body)
            elif self.notify_format == NotifyFormat.MARKDOWN and body_format == NotifyFormat.TEXT:
                title = title.lstrip('\r\n \t\x0b\x0c#-')
                if title:
                    body = '# {}\r\n{}'.format(title, body)
            else:
                body = '{}\r\n{}'.format(title, body)
            title = ''
        if self.body_max_line_count > 0:
            body = re.split('\\r*\\n', body)
            body = '\r\n'.join(body[0:self.body_max_line_count])
        if overflow == OverflowMode.UPSTREAM:
            response.append({'body': body, 'title': title})
            return response
        elif len(title) > self.title_maxlen:
            title = title[:self.title_maxlen]
        if self.body_maxlen > 0 and len(body) <= self.body_maxlen:
            response.append({'body': body, 'title': title})
            return response
        if overflow == OverflowMode.TRUNCATE:
            response.append({'body': body[:self.body_maxlen], 'title': title})
            return response
        response = [{'body': body[i:i + self.body_maxlen], 'title': title} for i in range(0, len(body), self.body_maxlen)]
        return response

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Should preform the actual notification itself.\n\n        '
        raise NotImplementedError('send() is not implimented by the child class.')

    def url_parameters(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Provides a default set of parameters to work with. This can greatly\n        simplify URL construction in the acommpanied url() function in all\n        defined plugin services.\n        '
        params = {'format': self.notify_format, 'overflow': self.overflow_mode}
        params.update(super().url_parameters(*args, **kwargs))
        return params

    @staticmethod
    def parse_url(url, verify_host=True, plus_to_space=False):
        if False:
            i = 10
            return i + 15
        'Parses the URL and returns it broken apart into a dictionary.\n\n        This is very specific and customized for Apprise.\n\n\n        Args:\n            url (str): The URL you want to fully parse.\n            verify_host (:obj:`bool`, optional): a flag kept with the parsed\n                 URL which some child classes will later use to verify SSL\n                 keys (if SSL transactions take place).  Unless under very\n                 specific circumstances, it is strongly recomended that\n                 you leave this default value set to True.\n\n        Returns:\n            A dictionary is returned containing the URL fully parsed if\n            successful, otherwise None is returned.\n        '
        results = URLBase.parse_url(url, verify_host=verify_host, plus_to_space=plus_to_space)
        if not results:
            return results
        if 'format' in results['qsd']:
            results['format'] = results['qsd'].get('format')
            if results['format'] not in NOTIFY_FORMATS:
                URLBase.logger.warning('Unsupported format specified {}'.format(results['format']))
                del results['format']
        if 'overflow' in results['qsd']:
            results['overflow'] = results['qsd'].get('overflow')
            if results['overflow'] not in OVERFLOW_MODES:
                URLBase.logger.warning('Unsupported overflow specified {}'.format(results['overflow']))
                del results['overflow']
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            while True:
                i = 10
        "\n        This is a base class that can be optionally over-ridden by child\n        classes who can build their Apprise URL based on the one provided\n        by the notification service they choose to use.\n\n        The intent of this is to make Apprise a little more userfriendly\n        to people who aren't familiar with constructing URLs and wish to\n        use the ones that were just provied by their notification serivice\n        that they're using.\n\n        This function will return None if the passed in URL can't be matched\n        as belonging to the notification service. Otherwise this function\n        should return the same set of results that parse_url() does.\n        "
        return None