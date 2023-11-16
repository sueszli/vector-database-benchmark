import json
import unittest
from unittest.mock import Mock, patch
from tap_facebook import FacebookRequestError
from tap_facebook import facebook_business
from facebook_business.exceptions import FacebookBadObjectError
from facebook_business import FacebookAdsApi
from facebook_business.api import FacebookResponse
from tap_facebook import AdCreative, AdsInsights
from facebook_business.adobjects.adaccount import AdAccount
import requests
from requests.models import Response

class TestAdCreative(unittest.TestCase):
    """A set of unit tests to ensure that requests to get AdCreatives behave
    as expected"""

    def test_retries_on_500(self):
        if False:
            return 10
        '`AdCreative.sync.do_request()` calls a `facebook_business` method,\n        `get_ad_creatives()`, to make a request to the API. We mock this\n        method to raise a `FacebookRequestError` with an `http_status` of\n        `500`.\n\n        We expect the tap to retry this request up to 5 times, which is\n        the current hard coded `max_tries` value.\n        '
        mocked_account = Mock()
        mocked_account.get_ad_creatives = Mock()
        mocked_account.get_ad_creatives.side_effect = FacebookRequestError(message='', request_context={'': Mock()}, http_status=500, http_headers=Mock(), body={})
        ad_creative_object = AdCreative('', mocked_account, '', '')
        with self.assertRaises(FacebookRequestError):
            ad_creative_object.sync()
        self.assertEquals(5, mocked_account.get_ad_creatives.call_count)

    def test_catch_a_type_error(self):
        if False:
            i = 10
            return i + 15
        '`AdCreative.sync.do_request()` calls a `facebook_business` method `get_ad_creatives()`.\n        We want to mock this to throw a `TypeError("string indices must be integers")` and assert\n        that we retry this specific error.\n        '
        mocked_account = Mock()
        mocked_account.get_ad_creatives = Mock()
        mocked_account.get_ad_creatives.side_effect = TypeError('string indices must be integers')
        ad_creative_object = AdCreative('', mocked_account, '', '')
        with self.assertRaises(TypeError):
            ad_creative_object.sync()
        self.assertEquals(5, mocked_account.get_ad_creatives.call_count)

    def test_retries_and_good_response(self):
        if False:
            for i in range(10):
                print('nop')
        'Facebook has a class called `FacebookResponse` and it is created from a `requests.Response`. Some\n        `facebook_business` functions depend on calling `FacebookResponse.json()`, which sometimes returns a\n        string instead of a dictionary. This leads to a `TypeError("string indices must be integers")` and\n        we want to retry these.\n\n        This test will return a "bad" API response the first time the function is called, then a\n        "good" response that can be `json.loads()`. We check that the resulting object has our\n        expected value in it.\n\n        '
        FacebookAdsApi.init(access_token='access_token')
        expected_value = {'foo': 'bar'}
        account = AdAccount('abc_123')
        patcher = patch('requests.Session.request')
        mocked_request = patcher.start()
        mocked_bad_response = Response()
        mocked_bad_response._content = b'images'
        mocked_good_response = Response()
        byte_string = json.dumps(expected_value).encode()
        mocked_good_response._content = byte_string
        mocked_request.side_effect = [mocked_bad_response, mocked_good_response]
        ad_creative_object = AdCreative('', account, '', '')
        with self.assertRaises(TypeError):
            ad_creative_object.account.get_ad_creatives(params={})
        list_response = ad_creative_object.account.get_ad_creatives(params={})
        actual_response = list_response.get_one()
        self.assertDictEqual(expected_value, actual_response._json)
        patcher.stop()

class TestInsightJobs(unittest.TestCase):
    """A set of unit tests to ensure that requests to get AdsInsights behave
    as expected"""

    def test_retries_on_bad_data(self):
        if False:
            while True:
                i = 10
        '`AdInsights.run_job()` calls a `facebook_business` method,\n        `get_insights()`, to make a request to the API. We mock this\n        method to raise a `FacebookBadObjectError`\n\n        We expect the tap to retry this request up to 5 times, which is\n        the current hard coded `max_tries` value.\n        '
        mocked_account = Mock()
        mocked_account.get_insights = Mock()
        mocked_account.get_insights.side_effect = FacebookBadObjectError('Bad data to set object data')
        ad_creative_object = AdsInsights('', mocked_account, '', '', {}, {})
        with self.assertRaises(FacebookBadObjectError):
            ad_creative_object.run_job({})
        self.assertEquals(5, mocked_account.get_insights.call_count)

    def test_retries_on_type_error(self):
        if False:
            return 10
        '`AdInsights.run_job()` calls a `facebook_business` method, `get_insights()`, to make a request to\n        the API. We want to mock this to throw a `TypeError("string indices must be integers")` and\n        assert that we retry this specific error.\n        '
        mocked_account = Mock()
        mocked_account.get_insights = Mock()
        mocked_account.get_insights.side_effect = TypeError('string indices must be integers')
        ad_creative_object = AdsInsights('', mocked_account, '', '', {}, {})
        with self.assertRaises(TypeError):
            ad_creative_object.run_job({})
        self.assertEquals(5, mocked_account.get_insights.call_count)

    def test_retries_and_good_response(self):
        if False:
            while True:
                i = 10
        'Facebook has a class called `FacebookResponse` and it is created from a `requests.Response`. Some\n        `facebook_business` functions depend on calling `FacebookResponse.json()`, which sometimes returns a\n        string instead of a dictionary. This leads to a `TypeError("string indices must be integers")` and\n        we want to retry these.\n\n        This test will return a "bad" API response the first time the function is called, then a\n        "good" response that can be `json.loads()`. We check that the resulting object has our\n        expected value in it.\n        '
        FacebookAdsApi.init(access_token='access_token')
        expected_value = {'foo': 'bar'}
        account = AdAccount('abc_123')
        patcher = patch('requests.Session.request')
        mocked_request = patcher.start()
        mocked_bad_response = Response()
        mocked_bad_response._content = b'images'
        mocked_good_response = Response()
        byte_string = json.dumps(expected_value).encode()
        mocked_good_response._content = byte_string
        mocked_request.side_effect = [mocked_bad_response, mocked_good_response]
        ad_creative_object = AdsInsights('', account, '', '', {}, {})
        with self.assertRaises(TypeError):
            ad_creative_object.account.get_insights(params={}, is_async=True)
        actual_response = ad_creative_object.account.get_insights(params={}, is_async=True)
        self.assertDictEqual(expected_value, actual_response._json)
        patcher.stop()

    def test_job_polling_retry(self):
        if False:
            while True:
                i = 10
        "AdInsights.api_get() polls the job status of an insights job we've requested\n        that Facebook generate. This test makes a request with a mock response to\n        raise a 400 status error that should be retried.\n\n        We expect the tap to retry this request up to 5 times for each insights job attempted.\n        "
        mocked_api_get = Mock()
        mocked_api_get.side_effect = FacebookRequestError(message='Unsupported get request; Object does not exist', request_context={'': Mock()}, http_status=400, http_headers=Mock(), body={'error': {'error_subcode': 33}})
        mocked_account = Mock()
        mocked_account.get_insights = Mock()
        mocked_account.get_insights.return_value.api_get = mocked_api_get
        ad_insights_object = AdsInsights('', mocked_account, '', '', {}, {})
        with self.assertRaises(FacebookRequestError):
            ad_insights_object.run_job({})
        self.assertEquals(25, mocked_account.get_insights.return_value.api_get.call_count)
        self.assertEquals(5, mocked_account.get_insights.call_count)

    def test_job_polling_retry_succeeds_eventually(self):
        if False:
            print('Hello World!')
        "AdInsights.api_get() polls the job status of an insights job we've requested\n        that Facebook generate. This test makes a request with a mock response to\n        raise a 400 status error that should be retried.\n\n        We expect the tap to retry this request up to 5 times for each insights job attempted.\n        "
        mocked_bad_response = FacebookRequestError(message='Unsupported get request; Object does not exist', request_context={'': Mock()}, http_status=400, http_headers=Mock(), body={'error': {'error_subcode': 33}})
        mocked_good_response = {'async_status': 'Job Completed', 'async_percent_completion': 100, 'id': '2134'}
        mocked_api_get = Mock()
        mocked_api_get.side_effect = [mocked_bad_response, mocked_bad_response, mocked_good_response]
        mocked_account = Mock()
        mocked_account.get_insights = Mock()
        mocked_account.get_insights.return_value.api_get = mocked_api_get
        ad_insights_object = AdsInsights('', mocked_account, '', '', {}, {})
        ad_insights_object.run_job({})
        self.assertEquals(3, mocked_account.get_insights.return_value.api_get.call_count)
        self.assertEquals(1, mocked_account.get_insights.call_count)