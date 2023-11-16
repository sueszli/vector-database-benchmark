"""Example use of a service account to authenticate to Identity-Aware Proxy."""
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import requests

def make_iap_request(url, client_id, method='GET', **kwargs):
    if False:
        while True:
            i = 10
    "Makes a request to an application protected by Identity-Aware Proxy.\n\n    Args:\n      url: The Identity-Aware Proxy-protected URL to fetch.\n      client_id: The client ID used by Identity-Aware Proxy.\n      method: The request method to use\n              ('GET', 'OPTIONS', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE')\n      **kwargs: Any of the parameters defined for the request function:\n                https://github.com/requests/requests/blob/master/requests/api.py\n                If no timeout is provided, it is set to 90 by default.\n\n    Returns:\n      The page body, or raises an exception if the page couldn't be retrieved.\n    "
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 90
    open_id_connect_token = id_token.fetch_id_token(Request(), client_id)
    resp = requests.request(method, url, headers={'Authorization': 'Bearer {}'.format(open_id_connect_token)}, **kwargs)
    if resp.status_code == 403:
        raise Exception('Service account does not have permission to access the IAP-protected application.')
    elif resp.status_code != 200:
        raise Exception('Bad response from application: {!r} / {!r} / {!r}'.format(resp.status_code, resp.headers, resp.text))
    else:
        return resp.text