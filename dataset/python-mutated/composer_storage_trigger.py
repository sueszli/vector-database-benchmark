from google.auth.transport.requests import Request
from google.oauth2 import id_token
import requests
IAM_SCOPE = 'https://www.googleapis.com/auth/iam'
OAUTH_TOKEN_URI = 'https://www.googleapis.com/oauth2/v4/token'
USE_EXPERIMENTAL_API = True

def trigger_dag(data, context=None):
    if False:
        for i in range(10):
            print('nop')
    'Makes a POST request to the Composer DAG Trigger API\n\n    When called via Google Cloud Functions (GCF),\n    data and context are Background function parameters.\n\n    For more info, refer to\n    https://cloud.google.com/functions/docs/writing/background#functions_background_parameters-python\n\n    To call this function from a Python script, omit the ``context`` argument\n    and pass in a non-null value for the ``data`` argument.\n\n    This function is currently only compatible with Composer v1 environments.\n    '
    client_id = 'YOUR-CLIENT-ID'
    webserver_id = 'YOUR-TENANT-PROJECT'
    dag_name = 'composer_sample_trigger_response_dag'
    if USE_EXPERIMENTAL_API:
        endpoint = f'api/experimental/dags/{dag_name}/dag_runs'
        json_data = {'conf': data, 'replace_microseconds': 'false'}
    else:
        endpoint = f'api/v1/dags/{dag_name}/dagRuns'
        json_data = {'conf': data}
    webserver_url = 'https://' + webserver_id + '.appspot.com/' + endpoint
    make_iap_request(webserver_url, client_id, method='POST', json=json_data)

def make_iap_request(url, client_id, method='GET', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Makes a request to an application protected by Identity-Aware Proxy.\n    Args:\n      url: The Identity-Aware Proxy-protected URL to fetch.\n      client_id: The client ID used by Identity-Aware Proxy.\n      method: The request method to use\n              ('GET', 'OPTIONS', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE')\n      **kwargs: Any of the parameters defined for the request function:\n                https://github.com/requests/requests/blob/master/requests/api.py\n                If no timeout is provided, it is set to 90 by default.\n    Returns:\n      The page body, or raises an exception if the page couldn't be retrieved.\n    "
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 90
    google_open_id_connect_token = id_token.fetch_id_token(Request(), client_id)
    resp = requests.request(method, url, headers={'Authorization': 'Bearer {}'.format(google_open_id_connect_token)}, **kwargs)
    if resp.status_code == 403:
        raise Exception('Service account does not have permission to access the IAP-protected application.')
    elif resp.status_code != 200:
        raise Exception('Bad response from application: {!r} / {!r} / {!r}'.format(resp.status_code, resp.headers, resp.text))
    else:
        return resp.text