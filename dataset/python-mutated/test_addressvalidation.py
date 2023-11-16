"""Tests for the addressvalidation module."""
import responses
import googlemaps
from . import TestCase

class AddressValidationTest(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.key = 'AIzaSyD_sJl0qMA65CYHMBokVfMNA7AKyt5ERYs'
        self.client = googlemaps.Client(self.key)

    @responses.activate
    def test_simple_addressvalidation(self):
        if False:
            while True:
                i = 10
        responses.add(responses.POST, 'https://addressvalidation.googleapis.com/v1:validateAddress', body='{"address": {"regionCode": "US","locality": "Mountain View","addressLines": "1600 Amphitheatre Pkwy"},"enableUspsCass":true}', status=200, content_type='application/json')
        results = self.client.addressvalidation('1600 Amphitheatre Pk', regionCode='US', locality='Mountain View', enableUspsCass=True)
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://addressvalidation.googleapis.com/v1:validateAddress?key=%s' % self.key, responses.calls[0].request.url)