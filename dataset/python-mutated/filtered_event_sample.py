"""Sample plugin which responds to events."""
import logging
from django.conf import settings
from plugin import InvenTreePlugin
from plugin.mixins import EventMixin
logger = logging.getLogger('inventree')

class FilteredEventPluginSample(EventMixin, InvenTreePlugin):
    """A sample plugin which provides supports for triggered events."""
    NAME = 'FilteredEventPlugin'
    SLUG = 'filteredsampleevent'
    TITLE = 'Triggered by test.event only'

    def wants_process_event(self, event):
        if False:
            print('Hello World!')
        'Return whether given event should be processed or not.'
        return event == 'test.event'

    def process_event(self, event, *args, **kwargs):
        if False:
            print('Hello World!')
        'Custom event processing.'
        print(f"Processing triggered event: '{event}'")
        print('args:', str(args))
        print('kwargs:', str(kwargs))
        if settings.PLUGIN_TESTING:
            logger.debug('Event `%s` triggered in sample plugin', event)