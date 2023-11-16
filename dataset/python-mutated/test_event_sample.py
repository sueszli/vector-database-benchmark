"""Unit tests for event_sample sample plugins."""
from django.conf import settings
from django.test import TestCase
from common.models import InvenTreeSetting
from plugin import InvenTreePlugin, registry
from plugin.base.event.events import trigger_event
from plugin.helpers import MixinNotImplementedError
from plugin.mixins import EventMixin
from .event_sample import logger

class EventPluginSampleTests(TestCase):
    """Tests for EventPluginSample."""

    def test_run_event(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if the event is issued.'
        config = registry.get_plugin('sampleevent').plugin_config()
        config.active = True
        config.save()
        InvenTreeSetting.set_setting('ENABLE_PLUGINS_EVENTS', True, change_user=None)
        settings.PLUGIN_TESTING_EVENTS = True
        with self.assertLogs(logger=logger, level='DEBUG') as cm:
            trigger_event('test.event')
        self.assertIn('DEBUG:inventree:Event `test.event` triggered in sample plugin', cm[1])
        settings.PLUGIN_TESTING_EVENTS = False

    def test_mixin(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that MixinNotImplementedError is raised.'
        with self.assertRaises(MixinNotImplementedError):

            class Wrong(EventMixin, InvenTreePlugin):
                pass
            plugin = Wrong()
            plugin.process_event('abc')