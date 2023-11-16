"""Tests for the geocoding module."""
import datetime
import responses
import googlemaps
from . import TestCase

class GeocodingTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.key = 'AIzaasdf'
        self.client = googlemaps.Client(self.key)

    @responses.activate
    def test_simple_geocode(self):
        if False:
            while True:
                i = 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode('Sydney')
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?key=%s&address=Sydney' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_reverse_geocode(self):
        if False:
            i = 10
            return i + 15
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.reverse_geocode((-33.8674869, 151.2069902))
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?latlng=-33.8674869,151.2069902&key=%s' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_geocoding_the_googleplex(self):
        if False:
            print('Hello World!')
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode('1600 Amphitheatre Parkway, Mountain View, CA')
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?key=%s&address=1600+Amphitheatre+Parkway%%2C+Mountain+View%%2C+CA' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_geocode_with_bounds(self):
        if False:
            while True:
                i = 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode('Winnetka', bounds={'southwest': (34.172684, -118.604794), 'northeast': (34.236144, -118.500938)})
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?bounds=34.172684%%2C-118.604794%%7C34.236144%%2C-118.500938&key=%s&address=Winnetka' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_geocode_with_region_biasing(self):
        if False:
            return 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode('Toledo', region='es')
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?region=es&key=%s&address=Toledo' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_geocode_with_component_filter(self):
        if False:
            return 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode('santa cruz', components={'country': 'ES'})
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?key=%s&components=country%%3AES&address=santa+cruz' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_geocode_with_multiple_component_filters(self):
        if False:
            while True:
                i = 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode('Torun', components={'administrative_area': 'TX', 'country': 'US'})
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?key=%s&components=administrative_area%%3ATX%%7Ccountry%%3AUS&address=Torun' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_geocode_with_just_components(self):
        if False:
            while True:
                i = 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode(components={'route': 'Annegatan', 'administrative_area': 'Helsinki', 'country': 'Finland'})
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?key=%s&components=administrative_area%%3AHelsinki%%7Ccountry%%3AFinland%%7Croute%%3AAnnegatan' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_geocode_place_id(self):
        if False:
            i = 10
            return i + 15
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode(place_id='ChIJeRpOeF67j4AR9ydy_PIzPuM')
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?key=%s&place_id=ChIJeRpOeF67j4AR9ydy_PIzPuM' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_simple_reverse_geocode(self):
        if False:
            for i in range(10):
                print('nop')
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.reverse_geocode((40.714224, -73.961452))
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?latlng=40.714224%%2C-73.961452&key=%s' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_reverse_geocode_restricted_by_type(self):
        if False:
            i = 10
            return i + 15
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.reverse_geocode((40.714224, -73.961452), location_type='ROOFTOP', result_type='street_address')
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?latlng=40.714224%%2C-73.961452&result_type=street_address&key=%s&location_type=ROOFTOP' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_reverse_geocode_multiple_location_types(self):
        if False:
            while True:
                i = 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.reverse_geocode((40.714224, -73.961452), location_type=['ROOFTOP', 'RANGE_INTERPOLATED'], result_type='street_address')
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?latlng=40.714224%%2C-73.961452&result_type=street_address&key=%s&location_type=ROOFTOP%%7CRANGE_INTERPOLATED' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_reverse_geocode_multiple_result_types(self):
        if False:
            i = 10
            return i + 15
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.reverse_geocode((40.714224, -73.961452), location_type='ROOFTOP', result_type=['street_address', 'route'])
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?latlng=40.714224%%2C-73.961452&result_type=street_address%%7Croute&key=%s&location_type=ROOFTOP' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_partial_match(self):
        if False:
            while True:
                i = 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode('Pirrama Pyrmont')
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?key=%s&address=Pirrama+Pyrmont' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_utf_results(self):
        if False:
            while True:
                i = 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.geocode(components={'postal_code': '96766'})
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?key=%s&components=postal_code%%3A96766' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_utf8_request(self):
        if False:
            return 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/geocode/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        self.client.geocode(self.u('\\u4e2d\\u56fd'))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/geocode/json?key=%s&address=%s' % (self.key, '%E4%B8%AD%E5%9B%BD'), responses.calls[0].request.url)