"""Unit tests for locate_sample sample plugins."""
from django.urls import reverse
from InvenTree.unit_test import InvenTreeAPITestCase
from plugin import InvenTreePlugin, registry
from plugin.helpers import MixinNotImplementedError
from plugin.mixins import LocateMixin

class SampleLocatePlugintests(InvenTreeAPITestCase):
    """Tests for SampleLocatePlugin."""
    fixtures = ['location', 'category', 'part', 'stock']

    def test_run_locator(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if the event is issued.'
        config = registry.get_plugin('samplelocate').plugin_config()
        config.active = True
        config.save()
        url = reverse('api-locate-plugin')
        self.post(url, {}, expected_code=400)
        self.post(url, {'plugin': 'sampleevent'}, expected_code=400)
        self.post(url, {'plugin': 'samplelocate'}, expected_code=400)
        self.post(url, {'plugin': 'samplelocate', 'item': 999}, expected_code=404)
        self.post(url, {'plugin': 'samplelocate', 'item': 1}, expected_code=200)
        self.post(url, {'plugin': 'samplelocate', 'location': 999}, expected_code=404)
        self.post(url, {'plugin': 'samplelocate', 'location': 1}, expected_code=200)

    def test_mixin(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that MixinNotImplementedError is raised.'
        with self.assertRaises(MixinNotImplementedError):

            class Wrong(LocateMixin, InvenTreePlugin):
                pass
            plugin = Wrong()
            plugin.locate_stock_location(1)
        with self.assertRaises(MixinNotImplementedError):
            plugin.locate_stock_item(1)