import re
from .NotifyDiscord import NotifyDiscord

class NotifyGuilded(NotifyDiscord):
    """
    A wrapper to Guilded Notifications

    """
    service_name = 'Guilded'
    service_url = 'https://guilded.gg/'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_guilded'
    secure_protocol = 'guilded'
    notify_url = 'https://media.guilded.gg/webhooks'

    @staticmethod
    def parse_native_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Support https://media.guilded.gg/webhooks/WEBHOOK_ID/WEBHOOK_TOKEN\n        '
        result = re.match('^https?://(media\\.)?guilded\\.gg/webhooks/(?P<webhook_id>[-0-9a-f]+)/(?P<webhook_token>[A-Z0-9_-]+)/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            return NotifyGuilded.parse_url('{schema}://{webhook_id}/{webhook_token}/{params}'.format(schema=NotifyGuilded.secure_protocol, webhook_id=result.group('webhook_id'), webhook_token=result.group('webhook_token'), params='' if not result.group('params') else result.group('params')))
        return None