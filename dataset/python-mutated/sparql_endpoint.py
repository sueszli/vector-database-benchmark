"""Provide the query runner for SPARQL Endpoints.

seeAlso: https://www.w3.org/TR/rdf-sparql-query/
"""
import json
import logging
from os import environ
from redash.query_runner import BaseQueryRunner
from redash.utils import json_dumps, json_loads
from . import register
try:
    import requests
    from cmem.cmempy.queries import SparqlQuery
    from rdflib.plugins.sparql import prepareQuery
    enabled = True
except ImportError:
    enabled = False
logger = logging.getLogger(__name__)

class SPARQLEndpointQueryRunner(BaseQueryRunner):
    """Use SPARQL Endpoint as redash data source"""
    KNOWN_CONFIG_KEYS = ('SPARQL_BASE_URI', 'SSL_VERIFY')
    KNOWN_SECRET_KEYS = ()
    noop_query = "SELECT ?noop WHERE {BIND('noop' as ?noop)}"

    def __init__(self, configuration):
        if False:
            while True:
                i = 10
        'init the class and configuration'
        super(SPARQLEndpointQueryRunner, self).__init__(configuration)
        self.configuration = configuration

    def _setup_environment(self):
        if False:
            for i in range(10):
                print('nop')
        'provide environment for rdflib\n\n        rdflib environment variables need to match key in the properties\n        object of the configuration_schema\n        '
        for key in self.KNOWN_CONFIG_KEYS:
            if key in environ:
                environ.pop(key)
            value = self.configuration.get(key, None)
            if value is not None:
                environ[key] = str(value)
                if key in self.KNOWN_SECRET_KEYS:
                    logger.info('{} set by config'.format(key))
                else:
                    logger.info('{} set by config to {}'.format(key, environ[key]))

    @staticmethod
    def _transform_sparql_results(results):
        if False:
            while True:
                i = 10
        'transforms a SPARQL query result to a redash query result\n\n        source structure: SPARQL 1.1 Query Results JSON Format\n            - seeAlso: https://www.w3.org/TR/sparql11-results-json/\n\n        target structure: redash result set\n            there is no good documentation available\n            so here an example result set as needed for redash:\n            data = {\n                "columns": [ {"name": "name", "type": "string", "friendly_name": "friendly name"}],\n                "rows": [\n                    {"name": "value 1"},\n                    {"name": "value 2"}\n                ]}\n\n        FEATURE?: During the sparql_row loop, we could check the data types of the\n            values and, in case they are all the same, choose something better than\n            just string.\n        '
        logger.info('results are: {}'.format(results))
        sparql_results = json_loads(results)
        rows = []
        for sparql_row in sparql_results['results']['bindings']:
            row = {}
            for var in sparql_results['head']['vars']:
                try:
                    row[var] = sparql_row[var]['value']
                except KeyError:
                    row[var] = ''
            rows.append(row)
        columns = []
        for var in sparql_results['head']['vars']:
            columns.append({'name': var, 'friendly_name': var, 'type': 'string'})
        return json_dumps({'columns': columns, 'rows': rows})

    @classmethod
    def name(cls):
        if False:
            i = 10
            return i + 15
        return 'SPARQL Endpoint'

    @classmethod
    def enabled(cls):
        if False:
            return 10
        return enabled

    @classmethod
    def type(cls):
        if False:
            print('Hello World!')
        return 'sparql_endpoint'

    def remove_comments(self, string):
        if False:
            while True:
                i = 10
        return string[string.index('*/') + 2:].strip()

    def run_query(self, query, user):
        if False:
            return 10
        'send a query to a sparql endpoint'
        logger.info("about to execute query (user='{}'): {}".format(user, query))
        query_text = self.remove_comments(query)
        query = SparqlQuery(query_text)
        query_type = query.get_query_type()
        if query_type not in ['SELECT', None]:
            raise ValueError('Queries of type {} can not be processed by redash.'.format(query_type))
        self._setup_environment()
        try:
            endpoint = self.configuration.get('SPARQL_BASE_URI')
            r = requests.get(endpoint, params=dict(query=query_text), headers=dict(Accept='application/json'))
            data = self._transform_sparql_results(r.text)
        except Exception as error:
            logger.info('Error: {}'.format(error))
            try:
                details = json.loads(error.response.text)
                error = ''
                if 'title' in details:
                    error += details['title'] + ': '
                if 'detail' in details:
                    error += details['detail']
                    return (None, error)
            except Exception:
                pass
            return (None, error)
        error = None
        return (data, error)

    @classmethod
    def configuration_schema(cls):
        if False:
            while True:
                i = 10
        'provide the configuration of the data source as json schema'
        return {'type': 'object', 'properties': {'SPARQL_BASE_URI': {'type': 'string', 'title': 'Base URL'}, 'SSL_VERIFY': {'type': 'boolean', 'title': 'Verify SSL certificates for API requests', 'default': True}}, 'required': ['SPARQL_BASE_URI'], 'secret': [], 'extra_options': ['SSL_VERIFY']}

    def get_schema(self, get_stats=False):
        if False:
            print('Hello World!')
        'Get the schema structure (prefixes, graphs).'
        schema = dict()
        schema['1'] = {'name': '-> Common Prefixes <-', 'columns': self._get_common_prefixes_schema()}
        schema['2'] = {'name': '-> Graphs <-', 'columns': self._get_graphs_schema()}
        logger.info(f'Getting Schema Values: {schema.values()}')
        return schema.values()

    def _get_graphs_schema(self):
        if False:
            i = 10
            return i + 15
        'Get a list of readable graph FROM clause strings.'
        self._setup_environment()
        endpoint = self.configuration.get('SPARQL_BASE_URI')
        query_text = 'SELECT DISTINCT ?g WHERE {GRAPH ?g {?s ?p ?o}}'
        r = requests.get(endpoint, params=dict(query=query_text), headers=dict(Accept='application/json')).json()
        graph_iris = [g.get('g').get('value') for g in r.get('results').get('bindings')]
        graphs = []
        for graph in graph_iris:
            graphs.append('FROM <{}>'.format(graph))
        return graphs

    @staticmethod
    def _get_common_prefixes_schema():
        if False:
            print('Hello World!')
        'Get a list of SPARQL prefix declarations.'
        common_prefixes = ['PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>', 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>', 'PREFIX owl: <http://www.w3.org/2002/07/owl#>', 'PREFIX schema: <http://schema.org/>', 'PREFIX dct: <http://purl.org/dc/terms/>', 'PREFIX skos: <http://www.w3.org/2004/02/skos/core#>']
        return common_prefixes
register(SPARQLEndpointQueryRunner)