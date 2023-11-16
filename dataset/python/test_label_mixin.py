"""Unit tests for the label printing mixin."""

import json
import os
from unittest import mock

from django.apps import apps
from django.urls import reverse

from PIL import Image

from InvenTree.unit_test import InvenTreeAPITestCase
from label.models import PartLabel, StockItemLabel, StockLocationLabel
from part.models import Part
from plugin.base.label.mixins import LabelPrintingMixin
from plugin.helpers import MixinNotImplementedError
from plugin.plugin import InvenTreePlugin
from plugin.registry import registry
from stock.models import StockItem, StockLocation


class LabelMixinTests(InvenTreeAPITestCase):
    """Test that the Label mixin operates correctly."""

    fixtures = [
        'category',
        'part',
        'location',
        'stock',
    ]

    roles = 'all'

    def do_activate_plugin(self):
        """Activate the 'samplelabel' plugin."""
        config = registry.get_plugin('samplelabelprinter').plugin_config()
        config.active = True
        config.save()

    def do_url(self, parts, plugin_ref, label, url_name: str = 'api-part-label-print', url_single: str = 'part', invalid: bool = False):
        """Generate an URL to print a label."""
        # Construct URL
        kwargs = {}
        if label:
            kwargs["pk"] = label.pk

        url = reverse(url_name, kwargs=kwargs)

        # Append part filters
        if not parts:
            pass
        elif len(parts) == 1:
            url += f'?{url_single}={parts[0].pk}'
        elif len(parts) > 1:
            url += '?' + '&'.join([f'{url_single}s={item.pk}' for item in parts])

        # Append an invalid item
        if invalid:
            url += f'&{url_single}{"s" if len(parts) > 1 else ""}=abc'

        # Append plugin reference
        if plugin_ref:
            url += f'&plugin={plugin_ref}'

        return url

    def test_wrong_implementation(self):
        """Test that a wrong implementation raises an error."""

        class WrongPlugin(LabelPrintingMixin, InvenTreePlugin):
            pass

        with self.assertRaises(MixinNotImplementedError):
            plugin = WrongPlugin()
            plugin.print_label(filename='test')

    def test_installed(self):
        """Test that the sample printing plugin is installed."""
        # Get all label plugins
        plugins = registry.with_mixin('labels')
        self.assertEqual(len(plugins), 3)

        # But, it is not 'active'
        plugins = registry.with_mixin('labels', active=True)
        self.assertEqual(len(plugins), 2)

    def test_api(self):
        """Test that we can filter the API endpoint by mixin."""
        url = reverse('api-plugin-list')

        # Try POST (disallowed)
        response = self.client.post(url, {})
        self.assertEqual(response.status_code, 405)

        response = self.client.get(
            url,
            {
                'mixin': 'labels',
                'active': True,
            }
        )

        # No results matching this query!
        self.assertEqual(len(response.data), 0)

        # What about inactive?
        response = self.client.get(
            url,
            {
                'mixin': 'labels',
                'active': False,
            }
        )

        self.assertEqual(len(response.data), 0)

        self.do_activate_plugin()
        # Should be available via the API now
        response = self.client.get(
            url,
            {
                'mixin': 'labels',
                'active': True,
            }
        )

        self.assertEqual(len(response.data), 3)

        labels = [item['key'] for item in response.data]

        self.assertIn('samplelabelprinter', labels)
        self.assertIn('inventreelabelsheet', labels)

    def test_printing_process(self):
        """Test that a label can be printed."""
        # Ensure the labels were created
        apps.get_app_config('label').create_labels()

        # Lookup references
        part = Part.objects.first()
        plugin_ref = 'samplelabelprinter'
        label = PartLabel.objects.first()

        url = self.do_url([part], plugin_ref, label)

        # Non-exsisting plugin
        response = self.get(f'{url}123', expected_code=404)
        self.assertIn(f'Plugin \'{plugin_ref}123\' not found', str(response.content, 'utf8'))

        # Inactive plugin
        response = self.get(url, expected_code=400)
        self.assertIn(f'Plugin \'{plugin_ref}\' is not enabled', str(response.content, 'utf8'))

        # Active plugin
        self.do_activate_plugin()

        # Print one part
        self.get(url, expected_code=200)

        # Print multiple parts
        self.get(self.do_url(Part.objects.all()[:2], plugin_ref, label), expected_code=200)

        # Print multiple parts without a plugin
        self.get(self.do_url(Part.objects.all()[:2], None, label), expected_code=200)

        # Print multiple parts without a plugin in debug mode
        response = self.get(self.do_url(Part.objects.all()[:2], None, label), expected_code=200)

        data = json.loads(response.content)
        self.assertIn('file', data)

        # Print no part
        self.get(self.do_url(None, plugin_ref, label), expected_code=400)

        # Test that the labels have been printed
        # The sample labelling plugin simply prints to file
        self.assertTrue(os.path.exists('label.pdf'))

        # Read the raw .pdf data - ensure it contains some sensible information
        with open('label.pdf', 'rb') as f:
            pdf_data = str(f.read())
            self.assertIn('WeasyPrint', pdf_data)

        # Check that the .png file has already been created
        self.assertTrue(os.path.exists('label.png'))

        # And that it is a valid image file
        Image.open('label.png')

    def test_printing_options(self):
        """Test printing options."""
        # Ensure the labels were created
        apps.get_app_config('label').create_labels()

        # Lookup references
        plugin_ref = 'samplelabelprinter'
        label = PartLabel.objects.first()

        self.do_activate_plugin()

        # test options response
        options = self.options(self.do_url(Part.objects.all()[:2], plugin_ref, label), expected_code=200).json()
        self.assertTrue("amount" in options["actions"]["POST"])

        plg = registry.get_plugin(plugin_ref)
        with mock.patch.object(plg, "print_label") as print_label:
            # wrong value type
            res = self.post(self.do_url(Part.objects.all()[:2], plugin_ref, label), data={"amount": "-no-valid-int-"}, expected_code=400).json()
            self.assertTrue("amount" in res)
            print_label.assert_not_called()

            # correct value type
            self.post(self.do_url(Part.objects.all()[:2], plugin_ref, label), data={"amount": 13}, expected_code=200).json()
            self.assertEqual(print_label.call_args.kwargs["printing_options"], {"amount": 13})

    def test_printing_endpoints(self):
        """Cover the endpoints not covered by `test_printing_process`."""
        plugin_ref = 'samplelabelprinter'

        # Activate the label components
        apps.get_app_config('label').create_labels()
        self.do_activate_plugin()

        def run_print_test(label, qs, url_name, url_single):
            """Run tests on single and multiple page printing.

            Args:
                label: class of the label
                qs: class of the base queryset
                url_name: url for endpoints
                url_single: item lookup reference
            """
            label = label.objects.first()
            qs = qs.objects.all()

            # List endpoint
            self.get(self.do_url(None, None, None, f'{url_name}-list', url_single), expected_code=200)

            # List endpoint with filter
            self.get(self.do_url(qs[:2], None, None, f'{url_name}-list', url_single, invalid=True), expected_code=200)

            # Single page printing
            self.get(self.do_url(qs[:1], plugin_ref, label, f'{url_name}-print', url_single), expected_code=200)

            # Multi page printing
            self.get(self.do_url(qs[:2], plugin_ref, label, f'{url_name}-print', url_single), expected_code=200)

        # Test StockItemLabels
        run_print_test(StockItemLabel, StockItem, 'api-stockitem-label', 'item')

        # Test StockLocationLabels
        run_print_test(StockLocationLabel, StockLocation, 'api-stocklocation-label', 'location')

        # Test PartLabels
        run_print_test(PartLabel, Part, 'api-part-label', 'part')
