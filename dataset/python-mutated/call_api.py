import json
import cloudinary
from cloudinary.api_client.execute_request import execute_request
from cloudinary.utils import get_http_connector
logger = cloudinary.logger
_http = get_http_connector(cloudinary.config(), cloudinary.CERT_KWARGS)

def call_metadata_api(method, uri, params, **options):
    if False:
        i = 10
        return i + 15
    "Private function that assists with performing an API call to the\n    metadata_fields part of the Admin API\n    :param method: The HTTP method. Valid methods: get, post, put, delete\n    :param uri: REST endpoint of the API (without 'metadata_fields')\n    :param params: Query/body parameters passed to the method\n    :param options: Additional options\n    :rtype: Response\n    "
    uri = ['metadata_fields'] + (uri or [])
    return call_json_api(method, uri, params, **options)

def call_json_api(method, uri, json_body, **options):
    if False:
        print('Hello World!')
    data = json.dumps(json_body).encode('utf-8')
    return _call_api(method, uri, body=data, headers={'Content-Type': 'application/json'}, **options)

def call_api(method, uri, params, **options):
    if False:
        print('Hello World!')
    return _call_api(method, uri, params=params, **options)

def _call_api(method, uri, params=None, body=None, headers=None, extra_headers=None, **options):
    if False:
        while True:
            i = 10
    prefix = options.pop('upload_prefix', cloudinary.config().upload_prefix) or 'https://api.cloudinary.com'
    cloud_name = options.pop('cloud_name', cloudinary.config().cloud_name)
    if not cloud_name:
        raise Exception('Must supply cloud_name')
    api_key = options.pop('api_key', cloudinary.config().api_key)
    api_secret = options.pop('api_secret', cloudinary.config().api_secret)
    oauth_token = options.pop('oauth_token', cloudinary.config().oauth_token)
    _validate_authorization(api_key, api_secret, oauth_token)
    api_url = '/'.join([prefix, cloudinary.API_VERSION, cloud_name] + uri)
    auth = {'key': api_key, 'secret': api_secret, 'oauth_token': oauth_token}
    if body is not None:
        options['body'] = body
    if extra_headers is not None:
        headers.update(extra_headers)
    return execute_request(http_connector=_http, method=method, params=params, headers=headers, auth=auth, api_url=api_url, **options)

def _validate_authorization(api_key, api_secret, oauth_token):
    if False:
        while True:
            i = 10
    if oauth_token:
        return
    if not api_key:
        raise Exception('Must supply api_key')
    if not api_secret:
        raise Exception('Must supply api_secret')