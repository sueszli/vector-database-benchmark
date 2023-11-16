"""Tests for the elevation module."""
import datetime
import responses
import googlemaps
from . import TestCase

class ElevationTest(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.key = 'AIzaasdf'
        self.client = googlemaps.Client(self.key)

    @responses.activate
    def test_elevation_single(self):
        if False:
            print('Hello World!')
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/elevation/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.elevation((40.714728, -73.998672))
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/elevation/json?locations=enc:abowFtzsbM&key=%s' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_elevation_single_list(self):
        if False:
            print('Hello World!')
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/elevation/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.elevation([(40.714728, -73.998672)])
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/elevation/json?locations=enc:abowFtzsbM&key=%s' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_elevation_multiple(self):
        if False:
            return 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/elevation/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        locations = [(40.714728, -73.998672), (-34.397, 150.644)]
        results = self.client.elevation(locations)
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/elevation/json?locations=enc:abowFtzsbMhgmiMuobzi@&key=%s' % self.key, responses.calls[0].request.url)

    def test_elevation_along_path_single(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(googlemaps.exceptions.ApiError):
            results = self.client.elevation_along_path([(40.714728, -73.998672)], 5)

    @responses.activate
    def test_elevation_along_path(self):
        if False:
            return 10
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/elevation/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        path = [(40.714728, -73.998672), (-34.397, 150.644)]
        results = self.client.elevation_along_path(path, 5)
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/elevation/json?path=enc:abowFtzsbMhgmiMuobzi@&key=%s&samples=5' % self.key, responses.calls[0].request.url)

    @responses.activate
    def test_short_latlng(self):
        if False:
            i = 10
            return i + 15
        responses.add(responses.GET, 'https://maps.googleapis.com/maps/api/elevation/json', body='{"status":"OK","results":[]}', status=200, content_type='application/json')
        results = self.client.elevation((40, -73))
        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual('https://maps.googleapis.com/maps/api/elevation/json?locations=40,-73&key=%s' % self.key, responses.calls[0].request.url)