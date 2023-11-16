from django.core.exceptions import ValidationError
from django.db import models
from django.test import TestCase
from .models import GenericIPAddress

class GenericIPAddressFieldTests(TestCase):

    def test_genericipaddressfield_formfield_protocol(self):
        if False:
            return 10
        '\n        GenericIPAddressField with a specified protocol does not generate a\n        formfield without a protocol.\n        '
        model_field = models.GenericIPAddressField(protocol='IPv4')
        form_field = model_field.formfield()
        with self.assertRaises(ValidationError):
            form_field.clean('::1')
        model_field = models.GenericIPAddressField(protocol='IPv6')
        form_field = model_field.formfield()
        with self.assertRaises(ValidationError):
            form_field.clean('127.0.0.1')

    def test_null_value(self):
        if False:
            return 10
        '\n        Null values should be resolved to None.\n        '
        GenericIPAddress.objects.create()
        o = GenericIPAddress.objects.get()
        self.assertIsNone(o.ip)

    def test_blank_string_saved_as_null(self):
        if False:
            print('Hello World!')
        o = GenericIPAddress.objects.create(ip='')
        o.refresh_from_db()
        self.assertIsNone(o.ip)
        GenericIPAddress.objects.update(ip='')
        o.refresh_from_db()
        self.assertIsNone(o.ip)

    def test_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        instance = GenericIPAddress.objects.create(ip='::1')
        loaded = GenericIPAddress.objects.get()
        self.assertEqual(loaded.ip, instance.ip)