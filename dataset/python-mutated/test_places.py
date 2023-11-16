"""Tests for the places module."""
import uuid
from types import GeneratorType
import responses
import googlemaps
from . import TestCase

class PlacesTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.key = 'AIzaasdf'
        self.client = googlemaps.Client(self.key)
        self.location = (-33.86746, 151.20709)
        self.type = 'liquor_store'
        self.language = 'en-AU'
        self.region = 'AU'
        self.radius = 100

    @responses.activate
    def test_places_find(self):
        if False:
            for i in range(10):
                print('nop')
        url = 'https://maps.googleapis.com/maps/api/place/findplacefromtext/json'
        responses.add(responses.GET, url, body='{"status": "OK", "candidates": []}', status=200, content_type='application/json')
        self.client.find_place('restaurant', 'textquery', fields=['business_status', 'geometry/location', 'place_id'], location_bias='point:90,90', language=self.language)
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('%s?language=en-AU&inputtype=textquery&locationbias=point:90,90&input=restaurant&fields=business_status,geometry/location,place_id&key=%s' % (url, self.key), responses.calls[0].request.url)
        with self.assertRaises(ValueError):
            self.client.find_place('restaurant', 'invalid')
        with self.assertRaises(ValueError):
            self.client.find_place('restaurant', 'textquery', fields=['geometry', 'invalid'])
        with self.assertRaises(ValueError):
            self.client.find_place('restaurant', 'textquery', location_bias='invalid')

    @responses.activate
    def test_places_text_search(self):
        if False:
            i = 10
            return i + 15
        url = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
        responses.add(responses.GET, url, body='{"status": "OK", "results": [], "html_attributions": []}', status=200, content_type='application/json')
        self.client.places('restaurant', location=self.location, radius=self.radius, region=self.region, language=self.language, min_price=1, max_price=4, open_now=True, type=self.type)
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('%s?language=en-AU&location=-33.86746%%2C151.20709&maxprice=4&minprice=1&opennow=true&query=restaurant&radius=100&region=AU&type=liquor_store&key=%s' % (url, self.key), responses.calls[0].request.url)

    @responses.activate
    def test_places_nearby_search(self):
        if False:
            return 10
        url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
        responses.add(responses.GET, url, body='{"status": "OK", "results": [], "html_attributions": []}', status=200, content_type='application/json')
        self.client.places_nearby(location=self.location, keyword='foo', language=self.language, min_price=1, max_price=4, name='bar', open_now=True, rank_by='distance', type=self.type)
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('%s?keyword=foo&language=en-AU&location=-33.86746%%2C151.20709&maxprice=4&minprice=1&name=bar&opennow=true&rankby=distance&type=liquor_store&key=%s' % (url, self.key), responses.calls[0].request.url)
        with self.assertRaises(ValueError):
            self.client.places_nearby(radius=self.radius)
        with self.assertRaises(ValueError):
            self.client.places_nearby(self.location, rank_by='distance')
        with self.assertRaises(ValueError):
            self.client.places_nearby(location=self.location, rank_by='distance', keyword='foo', radius=self.radius)

    @responses.activate
    def test_place_detail(self):
        if False:
            return 10
        url = 'https://maps.googleapis.com/maps/api/place/details/json'
        responses.add(responses.GET, url, body='{"status": "OK", "result": {}, "html_attributions": []}', status=200, content_type='application/json')
        self.client.place('ChIJN1t_tDeuEmsRUsoyG83frY4', fields=['business_status', 'geometry/location', 'place_id', 'reviews'], language=self.language, reviews_no_translations=True, reviews_sort='newest')
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('%s?language=en-AU&placeid=ChIJN1t_tDeuEmsRUsoyG83frY4&reviews_no_translations=true&reviews_sort=newest&key=%s&fields=business_status,geometry/location,place_id,reviews' % (url, self.key), responses.calls[0].request.url)
        with self.assertRaises(ValueError):
            self.client.place('ChIJN1t_tDeuEmsRUsoyG83frY4', fields=['geometry', 'invalid'])

    @responses.activate
    def test_photo(self):
        if False:
            i = 10
            return i + 15
        url = 'https://maps.googleapis.com/maps/api/place/photo'
        responses.add(responses.GET, url, status=200)
        ref = 'CnRvAAAAwMpdHeWlXl-lH0vp7lez4znKPIWSWvgvZFISdKx45AwJVP1Qp37YOrH7sqHMJ8C-vBDC546decipPHchJhHZL94RcTUfPa1jWzo-rSHaTlbNtjh-N68RkcToUCuY9v2HNpo5mziqkir37WU8FJEqVBIQ4k938TI3e7bf8xq-uwDZcxoUbO_ZJzPxremiQurAYzCTwRhE_V0'
        response = self.client.places_photo(ref, max_width=100)
        self.assertTrue(isinstance(response, GeneratorType))
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('%s?maxwidth=100&photoreference=%s&key=%s' % (url, ref, self.key), responses.calls[0].request.url)

    @responses.activate
    def test_autocomplete(self):
        if False:
            i = 10
            return i + 15
        url = 'https://maps.googleapis.com/maps/api/place/autocomplete/json'
        responses.add(responses.GET, url, body='{"status": "OK", "predictions": []}', status=200, content_type='application/json')
        session_token = uuid.uuid4().hex
        self.client.places_autocomplete('Google', session_token=session_token, offset=3, origin=self.location, location=self.location, radius=self.radius, language=self.language, types='geocode', components={'country': 'au'}, strict_bounds=True)
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('%s?components=country%%3Aau&input=Google&language=en-AU&origin=-33.86746%%2C151.20709&location=-33.86746%%2C151.20709&offset=3&radius=100&strictbounds=true&types=geocode&key=%s&sessiontoken=%s' % (url, self.key, session_token), responses.calls[0].request.url)

    @responses.activate
    def test_autocomplete_query(self):
        if False:
            while True:
                i = 10
        url = 'https://maps.googleapis.com/maps/api/place/queryautocomplete/json'
        responses.add(responses.GET, url, body='{"status": "OK", "predictions": []}', status=200, content_type='application/json')
        self.client.places_autocomplete_query('pizza near New York')
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('%s?input=pizza+near+New+York&key=%s' % (url, self.key), responses.calls[0].request.url)