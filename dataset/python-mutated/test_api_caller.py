"""Unit tests for action caller sample."""
from django.test import TestCase
from plugin import registry

class SampleApiCallerPluginTests(TestCase):
    """Tests for SampleApiCallerPluginTests."""

    def test_return(self):
        if False:
            print('Hello World!')
        'Check if the external api call works.'
        self.assertIn('sample-api-caller', registry.plugins)
        plg = registry.plugins['sample-api-caller']
        self.assertTrue(plg)
        result = plg.get_external_url()
        self.assertTrue(result)
        self.assertIn('data', result)