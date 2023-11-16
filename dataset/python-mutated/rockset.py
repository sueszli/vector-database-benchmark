import requests
from redash.query_runner import TYPE_BOOLEAN, TYPE_FLOAT, TYPE_INTEGER, TYPE_STRING, BaseSQLQueryRunner, register
from redash.utils import json_dumps

def _get_type(value):
    if False:
        return 10
    if isinstance(value, int):
        return TYPE_INTEGER
    elif isinstance(value, float):
        return TYPE_FLOAT
    elif isinstance(value, bool):
        return TYPE_BOOLEAN
    elif isinstance(value, str):
        return TYPE_STRING
    return TYPE_STRING

class RocksetAPI(object):

    def __init__(self, api_key, api_server, vi_id):
        if False:
            return 10
        self.api_key = api_key
        self.api_server = api_server
        self.vi_id = vi_id

    def _request(self, endpoint, method='GET', body=None):
        if False:
            for i in range(10):
                print('nop')
        headers = {'Authorization': 'ApiKey {}'.format(self.api_key), 'User-Agent': 'rest:redash/1.0'}
        url = '{}/v1/orgs/self/{}'.format(self.api_server, endpoint)
        if method == 'GET':
            r = requests.get(url, headers=headers)
            return r.json()
        elif method == 'POST':
            r = requests.post(url, headers=headers, json=body)
            return r.json()
        else:
            raise 'Unknown method: {}'.format(method)

    def list_workspaces(self):
        if False:
            i = 10
            return i + 15
        response = self._request('ws')
        return [x['name'] for x in response['data'] if x['collection_count'] > 0]

    def list_collections(self, workspace='commons'):
        if False:
            print('Hello World!')
        response = self._request('ws/{}/collections'.format(workspace))
        return [x['name'] for x in response['data']]

    def collection_columns(self, workspace, collection):
        if False:
            while True:
                i = 10
        response = self.query('DESCRIBE "{}"."{}" OPTION(max_field_depth=1)'.format(workspace, collection))
        return sorted(set([x['field'][0] for x in response['results']]))

    def query(self, sql):
        if False:
            while True:
                i = 10
        query_path = 'queries'
        if self.vi_id is not None and self.vi_id != '':
            query_path = f'virtualinstances/{self.vi_id}/queries'
        return self._request(query_path, 'POST', {'sql': {'query': sql}})

class Rockset(BaseSQLQueryRunner):
    noop_query = 'SELECT 1'

    @classmethod
    def configuration_schema(cls):
        if False:
            print('Hello World!')
        return {'type': 'object', 'properties': {'api_server': {'type': 'string', 'title': 'API Server', 'default': 'https://api.rs2.usw2.rockset.com'}, 'api_key': {'title': 'API Key', 'type': 'string'}, 'vi_id': {'title': 'Virtual Instance ID', 'type': 'string'}}, 'order': ['api_key', 'api_server', 'vi_id'], 'required': ['api_server', 'api_key'], 'secret': ['api_key']}

    @classmethod
    def type(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'rockset'

    def __init__(self, configuration):
        if False:
            for i in range(10):
                print('nop')
        super(Rockset, self).__init__(configuration)
        self.api = RocksetAPI(self.configuration.get('api_key'), self.configuration.get('api_server', 'https://api.usw2a1.rockset.com'), self.configuration.get('vi_id'))

    def _get_tables(self, schema):
        if False:
            i = 10
            return i + 15
        for workspace in self.api.list_workspaces():
            for collection in self.api.list_collections(workspace):
                table_name = collection if workspace == 'commons' else '{}.{}'.format(workspace, collection)
                schema[table_name] = {'name': table_name, 'columns': self.api.collection_columns(workspace, collection)}
        return sorted(schema.values(), key=lambda x: x['name'])

    def run_query(self, query, user):
        if False:
            print('Hello World!')
        results = self.api.query(query)
        if 'code' in results and results['code'] != 200:
            return (None, '{}: {}'.format(results['type'], results['message']))
        if 'results' not in results:
            message = results.get('message', 'Unknown response from Rockset.')
            return (None, message)
        rows = results['results']
        columns = []
        if len(rows) > 0:
            columns = []
            for k in rows[0]:
                columns.append({'name': k, 'friendly_name': k, 'type': _get_type(rows[0][k])})
        data = json_dumps({'columns': columns, 'rows': rows})
        return (data, None)
register(Rockset)