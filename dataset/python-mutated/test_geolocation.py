"""Tests for the geocolocation module."""
import responses
import googlemaps
from . import TestCase

class GeolocationTest(TestCase):

    def setUp(self):
        if False:
            return 10
        self.key = 'AIzaasdf'
        self.client = googlemaps.Client(self.key)

    @responses.activate
    def test_simple_geolocate(self):
        if False:
            print('Hello World!')
        responses.add(responses.POST, 'https://www.googleapis.com/geolocation/v1/geolocate', body='{"location": {"lat": 51.0,"lng": -0.1},"accuracy": 1200.4}', status=200, content_type='application/json')
        results = self.client.geolocate()
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://www.googleapis.com/geolocation/v1/geolocate?key=%s' % self.key, responses.calls[0].request.url)