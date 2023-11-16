import re
from django.test import TestCase
from django.urls import reverse
from wagtail.admin.urls import get_sprite_hash, sprite_hash

class TestIconSprite(TestCase):

    def test_get_sprite_hash(self):
        if False:
            for i in range(10):
                print('nop')
        result = get_sprite_hash()
        self.assertTrue(bool(re.match('^[a-z0-9]{8}$', result)))

    def test_hash_var(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(sprite_hash, str)
        self.assertEqual(len(sprite_hash), 8)

    def test_url(self):
        if False:
            while True:
                i = 10
        url = reverse('wagtailadmin_sprite')
        self.assertEqual(url[:14], '/admin/sprite-')

    def test_view(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('wagtailadmin_sprite'))
        self.assertIn('Content-Type: text/html; charset=utf-8', str(response.serialize_headers()))