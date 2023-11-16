from rest_framework.test import APITestCase, APIClient
from django.urls import reverse
from rest_framework.authtoken.models import Token

class MetadataTest(APITestCase):
    """
    Test the metadata APIv2 endpoint.
    """
    fixtures = ['dojo_testdata.json']

    def setUp(self):
        if False:
            print('Hello World!')
        token = Token.objects.get(user__username='admin')
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)
        r = self.create(product=1, name='foo', value='bar')
        self.assertEqual(r.status_code, 201)
        self.mid = r.json()['id']

    def create(self, **kwargs):
        if False:
            return 10
        return self.client.post(reverse('metadata-list'), kwargs, format='json')

    def test_docs(self):
        if False:
            i = 10
            return i + 15
        r = self.client.get(reverse('api_v2_schema'))
        self.assertEqual(r.status_code, 200)

    def test_query_metadata(self):
        if False:
            i = 10
            return i + 15
        r = self.client.get(reverse('metadata-detail', args=(self.mid,)))
        self.assertEqual(r.status_code, 200)

    def test_query_product_endpoint(self):
        if False:
            return 10
        r = self.client.get(reverse('product-detail', args=(1,)))
        self.assertTrue(dict(name='foo', value='bar') in r.json()['product_meta'])

    def test_delete(self):
        if False:
            print('Hello World!')
        r = self.client.delete(reverse('metadata-detail', args=(self.mid,)))
        self.assertEqual(r.status_code, 204)
        r = self.client.get(reverse('metadata-detail', args=(self.mid,)))
        self.assertEqual(r.status_code, 404)
        r = self.client.get(reverse('product-detail', args=(1,)))
        self.assertTrue(dict(name='foo', value='bar') not in r.json()['product_meta'])

    def test_no_product_or_endpoint_as_parameter(self):
        if False:
            i = 10
            return i + 15
        r = self.create(name='foo', value='bar')
        self.assertEqual(r.status_code, 400)

    def test_product_and_endpoint_as_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        r = self.create(product=1, endpoint=1, name='foo', value='bar')
        self.assertEqual(r.status_code, 400)

    def test_invalid_product(self):
        if False:
            i = 10
            return i + 15
        r = self.create(product=99999, name='quux', value='bar')
        self.assertEqual(r.status_code, 404)
        r = self.client.get(reverse('metadata-list'))
        for x in r.json()['results']:
            self.assertFalse(x['name'] == 'quux' and x['value'] == 'bar', x)

    def test_missing_name(self):
        if False:
            print('Hello World!')
        r = self.create(product=1, value='bar')
        self.assertEqual(r.status_code, 400)

    def test_none_name(self):
        if False:
            return 10
        r = self.create(product=1, name=None, value='bar')
        self.assertEqual(r.status_code, 400)

    def test_empty_name(self):
        if False:
            return 10
        r = self.create(product=1, name='', value='bar')
        self.assertEqual(r.status_code, 400)

    def test_missing_value(self):
        if False:
            while True:
                i = 10
        r = self.create(product=1, name='foo')
        self.assertEqual(r.status_code, 400)

    def test_none_value(self):
        if False:
            while True:
                i = 10
        r = self.create(product=1, name='foo', value=None)
        self.assertEqual(r.status_code, 400)

    def test_empty_value(self):
        if False:
            return 10
        r = self.create(product=1, name='foo', value='')
        self.assertEqual(r.status_code, 400)

    def test_unique_constraint(self):
        if False:
            print('Hello World!')
        r = self.create(product=1, name='foo', value='bar')
        self.assertEqual(r.status_code, 400)
        r = self.create(product=1, name='quux', value='bar')
        self.assertEqual(r.status_code, 201)
        r = self.create(product=2, name='foo', value='bar')
        self.assertEqual(r.status_code, 201)
        r = self.create(endpoint=1, name='foo', value='bar')
        self.assertEqual(r.status_code, 201)