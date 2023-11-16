import os
import requests
from models.api_based_extension import APIBasedExtensionPoint

class APIBasedExtensionRequestor:
    timeout: (int, int) = (5, 60)
    'timeout for request connect and read'

    def __init__(self, api_endpoint: str, api_key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    def request(self, point: APIBasedExtensionPoint, params: dict) -> dict:
        if False:
            return 10
        '\n        Request the api.\n\n        :param point: the api point\n        :param params: the request params\n        :return: the response json\n        '
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(self.api_key)}
        url = self.api_endpoint
        try:
            proxies = None
            if os.environ.get('API_BASED_EXTENSION_HTTP_PROXY') and os.environ.get('API_BASED_EXTENSION_HTTPS_PROXY'):
                proxies = {'http': os.environ.get('API_BASED_EXTENSION_HTTP_PROXY'), 'https': os.environ.get('API_BASED_EXTENSION_HTTPS_PROXY')}
            response = requests.request(method='POST', url=url, json={'point': point.value, 'params': params}, headers=headers, timeout=self.timeout, proxies=proxies)
        except requests.exceptions.Timeout:
            raise ValueError('request timeout')
        except requests.exceptions.ConnectionError:
            raise ValueError('request connection error')
        if response.status_code != 200:
            raise ValueError('request error, status_code: {}, content: {}'.format(response.status_code, response.text[:100]))
        return response.json()