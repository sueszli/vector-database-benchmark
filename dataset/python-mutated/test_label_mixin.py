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
    fixtures = ['category', 'part', 'location', 'stock']
    roles = 'all'

    def do_activate_plugin(self):
        if False:
            print('Hello World!')
        "Activate the 'samplelabel' plugin."
        config = registry.get_plugin('samplelabelprinter').plugin_config()
        config.active = True
        config.save()

    def do_url(self, parts, plugin_ref, label, url_name: str='api-part-label-print', url_single: str='part', invalid: bool=False):
        if False:
            i = 10
            return i + 15
        'Generate an URL to print a label.'
        kwargs = {}
        if label:
            kwargs['pk'] = label.pk
        url = reverse(url_name, kwargs=kwargs)
        if not parts:
            pass
        elif len(parts) == 1:
            url += f'?{url_single}={parts[0].pk}'
        elif len(parts) > 1:
            url += '?' + '&'.join([f'{url_single}s={item.pk}' for item in parts])
        if invalid:
            url += f"&{url_single}{('s' if len(parts) > 1 else '')}=abc"
        if plugin_ref:
            url += f'&plugin={plugin_ref}'
        return url

    def test_wrong_implementation(self):
        if False:
            i = 10
            return i + 15
        'Test that a wrong implementation raises an error.'

        class WrongPlugin(LabelPrintingMixin, InvenTreePlugin):
            pass
        with self.assertRaises(MixinNotImplementedError):
            plugin = WrongPlugin()
            plugin.print_label(filename='test')

    def test_installed(self):
        if False:
            print('Hello World!')
        'Test that the sample printing plugin is installed.'
        plugins = registry.with_mixin('labels')
        self.assertEqual(len(plugins), 3)
        plugins = registry.with_mixin('labels', active=True)
        self.assertEqual(len(plugins), 2)

    def test_api(self):
        if False:
            while True:
                i = 10
        'Test that we can filter the API endpoint by mixin.'
        url = reverse('api-plugin-list')
        response = self.client.post(url, {})
        self.assertEqual(response.status_code, 405)
        response = self.client.get(url, {'mixin': 'labels', 'active': True})
        self.assertEqual(len(response.data), 0)
        response = self.client.get(url, {'mixin': 'labels', 'active': False})
        self.assertEqual(len(response.data), 0)
        self.do_activate_plugin()
        response = self.client.get(url, {'mixin': 'labels', 'active': True})
        self.assertEqual(len(response.data), 3)
        labels = [item['key'] for item in response.data]
        self.assertIn('samplelabelprinter', labels)
        self.assertIn('inventreelabelsheet', labels)

    def test_printing_process(self):
        if False:
            return 10
        'Test that a label can be printed.'
        apps.get_app_config('label').create_labels()
        part = Part.objects.first()
        plugin_ref = 'samplelabelprinter'
        label = PartLabel.objects.first()
        url = self.do_url([part], plugin_ref, label)
        response = self.get(f'{url}123', expected_code=404)
        self.assertIn(f"Plugin '{plugin_ref}123' not found", str(response.content, 'utf8'))
        response = self.get(url, expected_code=400)
        self.assertIn(f"Plugin '{plugin_ref}' is not enabled", str(response.content, 'utf8'))
        self.do_activate_plugin()
        self.get(url, expected_code=200)
        self.get(self.do_url(Part.objects.all()[:2], plugin_ref, label), expected_code=200)
        self.get(self.do_url(Part.objects.all()[:2], None, label), expected_code=200)
        response = self.get(self.do_url(Part.objects.all()[:2], None, label), expected_code=200)
        data = json.loads(response.content)
        self.assertIn('file', data)
        self.get(self.do_url(None, plugin_ref, label), expected_code=400)
        self.assertTrue(os.path.exists('label.pdf'))
        with open('label.pdf', 'rb') as f:
            pdf_data = str(f.read())
            self.assertIn('WeasyPrint', pdf_data)
        self.assertTrue(os.path.exists('label.png'))
        Image.open('label.png')

    def test_printing_options(self):
        if False:
            i = 10
            return i + 15
        'Test printing options.'
        apps.get_app_config('label').create_labels()
        plugin_ref = 'samplelabelprinter'
        label = PartLabel.objects.first()
        self.do_activate_plugin()
        options = self.options(self.do_url(Part.objects.all()[:2], plugin_ref, label), expected_code=200).json()
        self.assertTrue('amount' in options['actions']['POST'])
        plg = registry.get_plugin(plugin_ref)
        with mock.patch.object(plg, 'print_label') as print_label:
            res = self.post(self.do_url(Part.objects.all()[:2], plugin_ref, label), data={'amount': '-no-valid-int-'}, expected_code=400).json()
            self.assertTrue('amount' in res)
            print_label.assert_not_called()
            self.post(self.do_url(Part.objects.all()[:2], plugin_ref, label), data={'amount': 13}, expected_code=200).json()
            self.assertEqual(print_label.call_args.kwargs['printing_options'], {'amount': 13})

    def test_printing_endpoints(self):
        if False:
            print('Hello World!')
        'Cover the endpoints not covered by `test_printing_process`.'
        plugin_ref = 'samplelabelprinter'
        apps.get_app_config('label').create_labels()
        self.do_activate_plugin()

        def run_print_test(label, qs, url_name, url_single):
            if False:
                for i in range(10):
                    print('nop')
            'Run tests on single and multiple page printing.\n\n            Args:\n                label: class of the label\n                qs: class of the base queryset\n                url_name: url for endpoints\n                url_single: item lookup reference\n            '
            label = label.objects.first()
            qs = qs.objects.all()
            self.get(self.do_url(None, None, None, f'{url_name}-list', url_single), expected_code=200)
            self.get(self.do_url(qs[:2], None, None, f'{url_name}-list', url_single, invalid=True), expected_code=200)
            self.get(self.do_url(qs[:1], plugin_ref, label, f'{url_name}-print', url_single), expected_code=200)
            self.get(self.do_url(qs[:2], plugin_ref, label, f'{url_name}-print', url_single), expected_code=200)
        run_print_test(StockItemLabel, StockItem, 'api-stockitem-label', 'item')
        run_print_test(StockLocationLabel, StockLocation, 'api-stocklocation-label', 'location')
        run_print_test(PartLabel, Part, 'api-part-label', 'part')