"""This module is designed to interact with the innertube API.

This module is NOT intended to be used directly by end users, as each of the
interfaces returns raw results. These should instead be parsed to extract
the useful information for the end user.
"""
import json
import os
import pathlib
import time
from urllib import parse
from pytube import request
_client_id = '861556708454-d6dlm3lh05idd8npek18k6be8ba3oc68.apps.googleusercontent.com'
_client_secret = 'SboVhoG9s0rNafixCSGGKXAT'
_api_keys = ['AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8', 'AIzaSyCtkvNIR1HCEwzsqK6JuE6KqpyjusIRI30', 'AIzaSyA8eiZmM1FaDVjRy-df2KTyQ_vz_yYM39w', 'AIzaSyC8UYZpvA2eknNex0Pjid0_eTLJoDu6los', 'AIzaSyCjc_pVEDi4qsv5MtC2dMXzpIaDoRFLsxw', 'AIzaSyDHQ9ipnphqTzDqZsbtd8_Ru4_kiKVQe2k']
_default_clients = {'WEB': {'context': {'client': {'clientName': 'WEB', 'clientVersion': '2.20200720.00.02'}}, 'header': {'User-Agent': 'Mozilla/5.0'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'ANDROID': {'context': {'client': {'clientName': 'ANDROID', 'clientVersion': '17.31.35', 'androidSdkVersion': 30}}, 'header': {'User-Agent': 'com.google.android.youtube/'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'IOS': {'context': {'client': {'clientName': 'IOS', 'clientVersion': '17.33.2', 'deviceModel': 'iPhone14,3'}}, 'header': {'User-Agent': 'com.google.ios.youtube/'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'WEB_EMBED': {'context': {'client': {'clientName': 'WEB_EMBEDDED_PLAYER', 'clientVersion': '2.20210721.00.00', 'clientScreen': 'EMBED'}}, 'header': {'User-Agent': 'Mozilla/5.0'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'ANDROID_EMBED': {'context': {'client': {'clientName': 'ANDROID_EMBEDDED_PLAYER', 'clientVersion': '17.31.35', 'clientScreen': 'EMBED', 'androidSdkVersion': 30}}, 'header': {'User-Agent': 'com.google.android.youtube/'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'IOS_EMBED': {'context': {'client': {'clientName': 'IOS_MESSAGES_EXTENSION', 'clientVersion': '17.33.2', 'deviceModel': 'iPhone14,3'}}, 'header': {'User-Agent': 'com.google.ios.youtube/'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'WEB_MUSIC': {'context': {'client': {'clientName': 'WEB_REMIX', 'clientVersion': '1.20220727.01.00'}}, 'header': {'User-Agent': 'Mozilla/5.0'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'ANDROID_MUSIC': {'context': {'client': {'clientName': 'ANDROID_MUSIC', 'clientVersion': '5.16.51', 'androidSdkVersion': 30}}, 'header': {'User-Agent': 'com.google.android.apps.youtube.music/'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'IOS_MUSIC': {'context': {'client': {'clientName': 'IOS_MUSIC', 'clientVersion': '5.21', 'deviceModel': 'iPhone14,3'}}, 'header': {'User-Agent': 'com.google.ios.youtubemusic/'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'WEB_CREATOR': {'context': {'client': {'clientName': 'WEB_CREATOR', 'clientVersion': '1.20220726.00.00'}}, 'header': {'User-Agent': 'Mozilla/5.0'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'ANDROID_CREATOR': {'context': {'client': {'clientName': 'ANDROID_CREATOR', 'clientVersion': '22.30.100', 'androidSdkVersion': 30}}, 'header': {'User-Agent': 'com.google.android.apps.youtube.creator/'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'IOS_CREATOR': {'context': {'client': {'clientName': 'IOS_CREATOR', 'clientVersion': '22.33.101', 'deviceModel': 'iPhone14,3'}}, 'header': {'User-Agent': 'com.google.ios.ytcreator/'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'MWEB': {'context': {'client': {'clientName': 'MWEB', 'clientVersion': '2.20220801.00.00'}}, 'header': {'User-Agent': 'Mozilla/5.0'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}, 'TV_EMBED': {'context': {'client': {'clientName': 'TVHTML5_SIMPLY_EMBEDDED_PLAYER', 'clientVersion': '2.0'}}, 'header': {'User-Agent': 'Mozilla/5.0'}, 'api_key': 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'}}
_token_timeout = 1800
_cache_dir = pathlib.Path(__file__).parent.resolve() / '__cache__'
_token_file = os.path.join(_cache_dir, 'tokens.json')

class InnerTube:
    """Object for interacting with the innertube API."""

    def __init__(self, client='ANDROID_MUSIC', use_oauth=False, allow_cache=True):
        if False:
            i = 10
            return i + 15
        'Initialize an InnerTube object.\n\n        :param str client:\n            Client to use for the object.\n            Default to web because it returns the most playback types.\n        :param bool use_oauth:\n            Whether or not to authenticate to YouTube.\n        :param bool allow_cache:\n            Allows caching of oauth tokens on the machine.\n        '
        self.context = _default_clients[client]['context']
        self.header = _default_clients[client]['header']
        self.api_key = _default_clients[client]['api_key']
        self.access_token = None
        self.refresh_token = None
        self.use_oauth = use_oauth
        self.allow_cache = allow_cache
        self.expires = None
        if self.use_oauth and self.allow_cache:
            if os.path.exists(_token_file):
                with open(_token_file) as f:
                    data = json.load(f)
                    self.access_token = data['access_token']
                    self.refresh_token = data['refresh_token']
                    self.expires = data['expires']
                    self.refresh_bearer_token()

    def cache_tokens(self):
        if False:
            print('Hello World!')
        'Cache tokens to file if allowed.'
        if not self.allow_cache:
            return
        data = {'access_token': self.access_token, 'refresh_token': self.refresh_token, 'expires': self.expires}
        if not os.path.exists(_cache_dir):
            os.mkdir(_cache_dir)
        with open(_token_file, 'w') as f:
            json.dump(data, f)

    def refresh_bearer_token(self, force=False):
        if False:
            return 10
        'Refreshes the OAuth token if necessary.\n\n        :param bool force:\n            Force-refresh the bearer token.\n        '
        if not self.use_oauth:
            return
        if self.expires > time.time() and (not force):
            return
        start_time = int(time.time() - 30)
        data = {'client_id': _client_id, 'client_secret': _client_secret, 'grant_type': 'refresh_token', 'refresh_token': self.refresh_token}
        response = request._execute_request('https://oauth2.googleapis.com/token', 'POST', headers={'Content-Type': 'application/json'}, data=data)
        response_data = json.loads(response.read())
        self.access_token = response_data['access_token']
        self.expires = start_time + response_data['expires_in']
        self.cache_tokens()

    def fetch_bearer_token(self):
        if False:
            while True:
                i = 10
        'Fetch an OAuth token.'
        start_time = int(time.time() - 30)
        data = {'client_id': _client_id, 'scope': 'https://www.googleapis.com/auth/youtube'}
        response = request._execute_request('https://oauth2.googleapis.com/device/code', 'POST', headers={'Content-Type': 'application/json'}, data=data)
        response_data = json.loads(response.read())
        verification_url = response_data['verification_url']
        user_code = response_data['user_code']
        print(f'Please open {verification_url} and input code {user_code}')
        input('Press enter when you have completed this step.')
        data = {'client_id': _client_id, 'client_secret': _client_secret, 'device_code': response_data['device_code'], 'grant_type': 'urn:ietf:params:oauth:grant-type:device_code'}
        response = request._execute_request('https://oauth2.googleapis.com/token', 'POST', headers={'Content-Type': 'application/json'}, data=data)
        response_data = json.loads(response.read())
        self.access_token = response_data['access_token']
        self.refresh_token = response_data['refresh_token']
        self.expires = start_time + response_data['expires_in']
        self.cache_tokens()

    @property
    def base_url(self):
        if False:
            print('Hello World!')
        'Return the base url endpoint for the innertube API.'
        return 'https://www.youtube.com/youtubei/v1'

    @property
    def base_data(self):
        if False:
            while True:
                i = 10
        'Return the base json data to transmit to the innertube API.'
        return {'context': self.context}

    @property
    def base_params(self):
        if False:
            while True:
                i = 10
        'Return the base query parameters to transmit to the innertube API.'
        return {'key': self.api_key, 'contentCheckOk': True, 'racyCheckOk': True}

    def _call_api(self, endpoint, query, data):
        if False:
            print('Hello World!')
        'Make a request to a given endpoint with the provided query parameters and data.'
        if self.use_oauth:
            del query['key']
        endpoint_url = f'{endpoint}?{parse.urlencode(query)}'
        headers = {'Content-Type': 'application/json'}
        if self.use_oauth:
            if self.access_token:
                self.refresh_bearer_token()
                headers['Authorization'] = f'Bearer {self.access_token}'
            else:
                self.fetch_bearer_token()
                headers['Authorization'] = f'Bearer {self.access_token}'
        headers.update(self.header)
        response = request._execute_request(endpoint_url, 'POST', headers=headers, data=data)
        return json.loads(response.read())

    def browse(self):
        if False:
            i = 10
            return i + 15
        'Make a request to the browse endpoint.\n\n        TODO: Figure out how we can use this\n        '
        ...

    def config(self):
        if False:
            i = 10
            return i + 15
        'Make a request to the config endpoint.\n\n        TODO: Figure out how we can use this\n        '
        ...

    def guide(self):
        if False:
            i = 10
            return i + 15
        'Make a request to the guide endpoint.\n\n        TODO: Figure out how we can use this\n        '
        ...

    def next(self):
        if False:
            print('Hello World!')
        'Make a request to the next endpoint.\n\n        TODO: Figure out how we can use this\n        '
        ...

    def player(self, video_id):
        if False:
            while True:
                i = 10
        'Make a request to the player endpoint.\n\n        :param str video_id:\n            The video id to get player info for.\n        :rtype: dict\n        :returns:\n            Raw player info results.\n        '
        endpoint = f'{self.base_url}/player'
        query = {'videoId': video_id}
        query.update(self.base_params)
        return self._call_api(endpoint, query, self.base_data)

    def search(self, search_query, continuation=None):
        if False:
            i = 10
            return i + 15
        'Make a request to the search endpoint.\n\n        :param str search_query:\n            The query to search.\n        :rtype: dict\n        :returns:\n            Raw search query results.\n        '
        endpoint = f'{self.base_url}/search'
        query = {'query': search_query}
        query.update(self.base_params)
        data = {}
        if continuation:
            data['continuation'] = continuation
        data.update(self.base_data)
        return self._call_api(endpoint, query, data)

    def verify_age(self, video_id):
        if False:
            print('Hello World!')
        'Make a request to the age_verify endpoint.\n\n        Notable examples of the types of video this verification step is for:\n        * https://www.youtube.com/watch?v=QLdAhwSBZ3w\n        * https://www.youtube.com/watch?v=hc0ZDaAZQT0\n\n        :param str video_id:\n            The video id to get player info for.\n        :rtype: dict\n        :returns:\n            Returns information that includes a URL for bypassing certain restrictions.\n        '
        endpoint = f'{self.base_url}/verify_age'
        data = {'nextEndpoint': {'urlEndpoint': {'url': f'/watch?v={video_id}'}}, 'setControvercy': True}
        data.update(self.base_data)
        result = self._call_api(endpoint, self.base_params, data)
        return result

    def get_transcript(self, video_id):
        if False:
            print('Hello World!')
        'Make a request to the get_transcript endpoint.\n\n        This is likely related to captioning for videos, but is currently untested.\n        '
        endpoint = f'{self.base_url}/get_transcript'
        query = {'videoId': video_id}
        query.update(self.base_params)
        result = self._call_api(endpoint, query, self.base_data)
        return result