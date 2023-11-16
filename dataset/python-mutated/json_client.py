"""JSON API Client."""
from __future__ import annotations
from urllib.parse import urljoin
from airflow.api.client import api_client

class Client(api_client.Client):
    """Json API client implementation.

    This client is used to interact with a Json API server and perform various actions
    such as triggering DAG runs,deleting DAGs, interacting with pools, and getting lineage information.
    """

    def _request(self, url: str, json=None, method: str='GET') -> dict:
        if False:
            i = 10
            return i + 15
        'Make a request to the Json API server.\n\n        :param url: The URL to send the request to.\n        :param method: The HTTP method to use (e.g. "GET", "POST", "DELETE").\n        :param json: A dictionary containing JSON data to send in the request body.\n        :return: A dictionary containing the JSON response from the server.\n        :raises OSError: If the server returns an error status.\n        '
        params = {'url': url}
        if json is not None:
            params['json'] = json
        resp = getattr(self._session, method.lower())(**params)
        if resp.is_error:
            try:
                data = resp.json()
            except Exception:
                data = {}
            raise OSError(data.get('error', 'Server error'))
        return resp.json()

    def trigger_dag(self, dag_id, run_id=None, conf=None, execution_date=None, replace_microseconds=True):
        if False:
            for i in range(10):
                print('nop')
        'Trigger a DAG run.\n\n        :param dag_id: The ID of the DAG to trigger.\n        :param run_id: The ID of the DAG run to create. If not provided, a default ID will be generated.\n        :param conf: A dictionary containing configuration data to pass to the DAG run.\n        :param execution_date: The execution date for the DAG run, in the format "YYYY-MM-DDTHH:MM:SS".\n        :param replace_microseconds: Whether to replace microseconds in the execution date with zeros.\n        :return: A message indicating the status of the DAG run trigger.\n        '
        endpoint = f'/api/experimental/dags/{dag_id}/dag_runs'
        url = urljoin(self._api_base_url, endpoint)
        data = {'run_id': run_id, 'conf': conf, 'execution_date': execution_date, 'replace_microseconds': replace_microseconds}
        return self._request(url, method='POST', json=data)['message']

    def delete_dag(self, dag_id: str):
        if False:
            print('Hello World!')
        'Delete a DAG.\n\n        :param dag_id: The ID of the DAG to delete.\n        :return: A message indicating the status of the DAG delete operation.\n        '
        endpoint = f'/api/experimental/dags/{dag_id}/delete_dag'
        url = urljoin(self._api_base_url, endpoint)
        data = self._request(url, method='DELETE')
        return data['message']

    def get_pool(self, name: str):
        if False:
            print('Hello World!')
        'Get information about a specific pool.\n\n        :param name: The name of the pool to retrieve information for.\n        :return: A tuple containing the name of the pool, the number of\n            slots in the pool, and a description of the pool.\n        '
        endpoint = f'/api/experimental/pools/{name}'
        url = urljoin(self._api_base_url, endpoint)
        pool = self._request(url)
        return (pool['pool'], pool['slots'], pool['description'])

    def get_pools(self):
        if False:
            while True:
                i = 10
        'Get a list of all pools.\n\n        :return: A list of tuples, each containing the name of a pool,\n            the number of slots in the pool, and a description of the pool.\n        '
        endpoint = '/api/experimental/pools'
        url = urljoin(self._api_base_url, endpoint)
        pools = self._request(url)
        return [(p['pool'], p['slots'], p['description']) for p in pools]

    def create_pool(self, name: str, slots: int, description: str, include_deferred: bool):
        if False:
            i = 10
            return i + 15
        'Create a new pool.\n\n        :param name: The name of the pool to create.\n        :param slots: The number of slots in the pool.\n        :param description: A description of the pool.\n        :param include_deferred: include deferred tasks in pool calculations\n\n        :return: A tuple containing the name of the pool, the number of slots in the pool,\n            a description of the pool and the include_deferred flag.\n        '
        endpoint = '/api/experimental/pools'
        data = {'name': name, 'slots': slots, 'description': description, 'include_deferred': include_deferred}
        response = self._request(urljoin(self._api_base_url, endpoint), method='POST', json=data)
        return (response['pool'], response['slots'], response['description'], response['include_deferred'])

    def delete_pool(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        'Delete a pool.\n\n        :param name: The name of the pool to delete.\n        :return: A tuple containing the name of the pool, the number\n            of slots in the pool, and a description of the pool.\n        '
        endpoint = f'/api/experimental/pools/{name}'
        url = urljoin(self._api_base_url, endpoint)
        pool = self._request(url, method='DELETE')
        return (pool['pool'], pool['slots'], pool['description'])

    def get_lineage(self, dag_id: str, execution_date: str):
        if False:
            while True:
                i = 10
        'Get the lineage of a DAG run.\n\n        :param dag_id: The ID of the DAG.\n        :param execution_date: The execution date of the DAG run, in the format "YYYY-MM-DDTHH:MM:SS".\n        :return: A message indicating the status of the lineage request.\n        '
        endpoint = f'/api/experimental/lineage/{dag_id}/{execution_date}'
        url = urljoin(self._api_base_url, endpoint)
        data = self._request(url, method='GET')
        return data['message']