"""
Trigger a DAG in Cloud Composer 2 environment using the Airflow 2 stable REST API.
"""
from __future__ import annotations
from typing import Any
import google.auth
from google.auth.transport.requests import AuthorizedSession
import requests
AUTH_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
(CREDENTIALS, _) = google.auth.default(scopes=[AUTH_SCOPE])

def make_composer2_web_server_request(url: str, method: str='GET', **kwargs: Any) -> google.auth.transport.Response:
    if False:
        print('Hello World!')
    "\n    Make a request to Cloud Composer 2 environment's web server.\n    Args:\n      url: The URL to fetch.\n      method: The request method to use ('GET', 'OPTIONS', 'HEAD', 'POST', 'PUT',\n        'PATCH', 'DELETE')\n      **kwargs: Any of the parameters defined for the request function:\n                https://github.com/requests/requests/blob/master/requests/api.py\n                  If no timeout is provided, it is set to 90 by default.\n    "
    authed_session = AuthorizedSession(CREDENTIALS)
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 90
    return authed_session.request(method, url, **kwargs)

def trigger_dag(web_server_url: str, dag_id: str, data: dict) -> str:
    if False:
        print('Hello World!')
    '\n    Make a request to trigger a dag using the stable Airflow 2 REST API.\n    https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html\n\n    Args:\n      web_server_url: The URL of the Airflow 2 web server.\n      dag_id: The DAG ID.\n      data: Additional configuration parameters for the DAG run (json).\n    '
    endpoint = f'api/v1/dags/{dag_id}/dagRuns'
    request_url = f'{web_server_url}/{endpoint}'
    json_data = {'conf': data}
    response = make_composer2_web_server_request(request_url, method='POST', json=json_data)
    if response.status_code == 403:
        raise requests.HTTPError(f'You do not have a permission to perform this operation. Check Airflow RBAC roles for your account.{response.headers} / {response.text}')
    elif response.status_code != 200:
        response.raise_for_status()
    else:
        return response.text
if __name__ == '__main__':
    dag_id = 'your-dag-id'
    dag_config = {'your-key': 'your-value'}
    web_server_url = 'https://example-airflow-ui-url-dot-us-central1.composer.googleusercontent.com'
    response_text = trigger_dag(web_server_url=web_server_url, dag_id=dag_id, data=dag_config)
    print(response_text)