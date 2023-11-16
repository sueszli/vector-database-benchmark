"""Provide the query runner for eccenca Corporate Memory.

seeAlso: https://documentation.eccenca.com/
seeAlso: https://eccenca.com/
"""
import json
import logging
from os import environ
from redash.query_runner import BaseQueryRunner
from redash.utils import json_dumps, json_loads
from . import register
try:
    from cmem.cmempy.dp.proxy.graph import get_graphs_list
    from cmem.cmempy.queries import QUERY_STRING, QueryCatalog, SparqlQuery
    enabled = True
except ImportError:
    enabled = False
logger = logging.getLogger(__name__)

class CorporateMemoryQueryRunner(BaseQueryRunner):
    """Use eccenca Corporate Memory as redash data source"""
    KNOWN_CONFIG_KEYS = ('CMEM_BASE_PROTOCOL', 'CMEM_BASE_DOMAIN', 'CMEM_BASE_URI', 'SSL_VERIFY', 'REQUESTS_CA_BUNDLE', 'DP_API_ENDPOINT', 'DI_API_ENDPOINT', 'OAUTH_TOKEN_URI', 'OAUTH_GRANT_TYPE', 'OAUTH_USER', 'OAUTH_PASSWORD', 'OAUTH_CLIENT_ID', 'OAUTH_CLIENT_SECRET')
    KNOWN_SECRET_KEYS = ('OAUTH_PASSWORD', 'OAUTH_CLIENT_SECRET')
    noop_query = "SELECT ?noop WHERE {BIND('noop' as ?noop)}"
    should_annotate_query = False

    def __init__(self, configuration):
        if False:
            print('Hello World!')
        'init the class and configuration'
        super(CorporateMemoryQueryRunner, self).__init__(configuration)
        '\n        FEATURE?: activate SPARQL support in the redash query editor\n            Currently SPARQL syntax seems not to be available for react-ace\n            component. However, the ace editor itself supports sparql mode:\n            https://github.com/ajaxorg/ace/blob/master/lib/ace/mode/sparql.js\n            then we can hopefully do: self.syntax = "sparql"\n        FEATURE?: implement the retrieve Query catalog URIs in order to use them in queries\n        FEATURE?: implement a way to use queries from the query catalog\n        FEATURE?: allow a checkbox to NOT use owl:imports imported graphs\n        FEATURE?: allow to use a context graph per data source\n        '
        self.configuration = configuration

    def _setup_environment(self):
        if False:
            print('Hello World!')
        'provide environment for cmempy\n\n        cmempy environment variables need to match key in the properties\n        object of the configuration_schema\n        '
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
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
        return 'eccenca Corporate Memory (with SPARQL)'

    @classmethod
    def enabled(cls):
        if False:
            i = 10
            return i + 15
        return enabled

    @classmethod
    def type(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'corporate_memory'

    def run_query(self, query, user):
        if False:
            for i in range(10):
                print('nop')
        'send a sparql query to corporate memory'
        query_text = query
        logger.info("about to execute query (user='{}'): {}".format(user, query_text))
        query = SparqlQuery(query_text)
        query_type = query.get_query_type()
        if query_type not in ['SELECT', None]:
            raise ValueError('Queries of type {} can not be processed by redash.'.format(query_type))
        self._setup_environment()
        try:
            data = self._transform_sparql_results(query.get_results())
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
            i = 10
            return i + 15
        'provide the configuration of the data source as json schema'
        return {'type': 'object', 'properties': {'CMEM_BASE_URI': {'type': 'string', 'title': 'Base URL'}, 'OAUTH_GRANT_TYPE': {'type': 'string', 'title': 'Grant Type', 'default': 'client_credentials', 'extendedEnum': [{'value': 'client_credentials', 'name': 'client_credentials'}, {'value': 'password', 'name': 'password'}]}, 'OAUTH_CLIENT_ID': {'type': 'string', 'title': 'Client ID (e.g. cmem-service-account)', 'default': 'cmem-service-account'}, 'OAUTH_CLIENT_SECRET': {'type': 'string', 'title': "Client Secret - only needed for grant type 'client_credentials'"}, 'OAUTH_USER': {'type': 'string', 'title': "User account - only needed for grant type 'password'"}, 'OAUTH_PASSWORD': {'type': 'string', 'title': "User Password - only needed for grant type 'password'"}, 'SSL_VERIFY': {'type': 'boolean', 'title': 'Verify SSL certificates for API requests', 'default': True}, 'REQUESTS_CA_BUNDLE': {'type': 'string', 'title': 'Path to the CA Bundle file (.pem)'}}, 'required': ['CMEM_BASE_URI', 'OAUTH_GRANT_TYPE', 'OAUTH_CLIENT_ID'], 'secret': ['OAUTH_CLIENT_SECRET', 'OAUTH_PASSWORD'], 'extra_options': ['OAUTH_GRANT_TYPE', 'OAUTH_USER', 'OAUTH_PASSWORD', 'SSL_VERIFY', 'REQUESTS_CA_BUNDLE']}

    def get_schema(self, get_stats=False):
        if False:
            i = 10
            return i + 15
        'Get the schema structure (prefixes, graphs).'
        schema = dict()
        schema['1'] = {'name': '-> Common Prefixes <-', 'columns': self._get_common_prefixes_schema()}
        schema['2'] = {'name': '-> Graphs <-', 'columns': self._get_graphs_schema()}
        logger.info(schema.values())
        return schema.values()

    def _get_graphs_schema(self):
        if False:
            while True:
                i = 10
        'Get a list of readable graph FROM clause strings.'
        self._setup_environment()
        graphs = []
        for graph in get_graphs_list():
            graphs.append('FROM <{}>'.format(graph['iri']))
        return graphs

    @staticmethod
    def _get_common_prefixes_schema():
        if False:
            return 10
        'Get a list of SPARQL prefix declarations.'
        common_prefixes = ['PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>', 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>', 'PREFIX owl: <http://www.w3.org/2002/07/owl#>', 'PREFIX schema: <http://schema.org/>', 'PREFIX dct: <http://purl.org/dc/terms/>', 'PREFIX skos: <http://www.w3.org/2004/02/skos/core#>']
        return common_prefixes
register(CorporateMemoryQueryRunner)