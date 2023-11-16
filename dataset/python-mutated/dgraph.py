import json
try:
    import pydgraph
    enabled = True
except ImportError:
    enabled = False
from redash.query_runner import BaseQueryRunner, register
from redash.utils import json_dumps

def reduce_item(reduced_item, key, value):
    if False:
        while True:
            i = 10
    'From https://github.com/vinay20045/json-to-csv'
    if isinstance(value, list):
        for (i, sub_item) in enumerate(value):
            reduce_item(reduced_item, '{}.{}'.format(key, i), sub_item)
    elif isinstance(value, dict):
        sub_keys = value.keys()
        for sub_key in sub_keys:
            reduce_item(reduced_item, '{}.{}'.format(key, sub_key), value[sub_key])
    else:
        reduced_item[key] = value

class Dgraph(BaseQueryRunner):
    should_annotate_query = False
    noop_query = '\n    {\n      test() {\n      }\n    }\n    '

    @classmethod
    def configuration_schema(cls):
        if False:
            while True:
                i = 10
        return {'type': 'object', 'properties': {'user': {'type': 'string'}, 'password': {'type': 'string'}, 'servers': {'type': 'string'}}, 'order': ['servers', 'user', 'password'], 'required': ['servers'], 'secret': ['password']}

    @classmethod
    def type(cls):
        if False:
            print('Hello World!')
        return 'dgraph'

    @classmethod
    def enabled(cls):
        if False:
            i = 10
            return i + 15
        return enabled

    def run_dgraph_query_raw(self, query):
        if False:
            while True:
                i = 10
        servers = self.configuration.get('servers')
        client_stub = pydgraph.DgraphClientStub(servers)
        client = pydgraph.DgraphClient(client_stub)
        txn = client.txn(read_only=True)
        try:
            response_raw = txn.query(query)
            data = json.loads(response_raw.json)
            return data
        except Exception as e:
            raise e
        finally:
            txn.discard()
            client_stub.close()

    def run_query(self, query, user):
        if False:
            i = 10
            return i + 15
        json_data = None
        error = None
        try:
            data = self.run_dgraph_query_raw(query)
            first_key = next(iter(list(data.keys())))
            first_node = data[first_key]
            data_to_be_processed = first_node
            processed_data = []
            header = []
            for item in data_to_be_processed:
                reduced_item = {}
                reduce_item(reduced_item, first_key, item)
                header += reduced_item.keys()
                processed_data.append(reduced_item)
            header = list(set(header))
            columns = [{'name': c, 'friendly_name': c, 'type': 'string'} for c in header]
            data = {'columns': columns, 'rows': processed_data}
            json_data = json_dumps(data)
        except Exception as e:
            error = e
        return (json_data, error)

    def get_schema(self, get_stats=False):
        if False:
            print('Hello World!')
        "Queries Dgraph for all the predicates, their types, their tokenizers, etc.\n\n        Dgraph only has one schema, and there's no such things as columns"
        query = 'schema {}'
        results = self.run_dgraph_query_raw(query)
        schema = {}
        for row in results['schema']:
            table_name = row['predicate']
            if table_name not in schema:
                schema[table_name] = {'name': table_name, 'columns': []}
        return list(schema.values())
register(Dgraph)