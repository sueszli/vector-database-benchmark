from django.core.exceptions import ImproperlyConfigured
from elasticsearch import NotFoundError
from wagtail.search.backends.elasticsearch7 import Elasticsearch7AutocompleteQueryCompiler, Elasticsearch7Index, Elasticsearch7Mapping, Elasticsearch7SearchBackend, Elasticsearch7SearchQueryCompiler, Elasticsearch7SearchResults
from wagtail.search.index import class_is_indexed

class Elasticsearch8Mapping(Elasticsearch7Mapping):
    pass

class Elasticsearch8Index(Elasticsearch7Index):

    def put(self):
        if False:
            while True:
                i = 10
        self.es.indices.create(index=self.name, **self.backend.settings)

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.es.indices.delete(index=self.name)
        except NotFoundError:
            pass

    def refresh(self):
        if False:
            return 10
        self.es.indices.refresh(index=self.name)

    def add_model(self, model):
        if False:
            for i in range(10):
                print('nop')
        mapping = self.mapping_class(model)
        self.es.indices.put_mapping(index=self.name, **mapping.get_mapping())

    def add_item(self, item):
        if False:
            i = 10
            return i + 15
        if not class_is_indexed(item.__class__):
            return
        mapping = self.mapping_class(item.__class__)
        self.es.index(index=self.name, document=mapping.get_document(item), id=mapping.get_document_id(item))

class Elasticsearch8SearchQueryCompiler(Elasticsearch7SearchQueryCompiler):
    mapping_class = Elasticsearch8Mapping

class Elasticsearch8SearchResults(Elasticsearch7SearchResults):

    def _backend_do_search(self, body, **kwargs):
        if False:
            return 10
        return self.backend.es.search(**body, **kwargs)

class Elasticsearch8AutocompleteQueryCompiler(Elasticsearch7AutocompleteQueryCompiler):
    mapping_class = Elasticsearch8Mapping

class Elasticsearch8SearchBackend(Elasticsearch7SearchBackend):
    mapping_class = Elasticsearch8Mapping
    index_class = Elasticsearch8Index
    query_compiler_class = Elasticsearch8SearchQueryCompiler
    autocomplete_query_compiler_class = Elasticsearch8AutocompleteQueryCompiler
    results_class = Elasticsearch8SearchResults
    timeout_kwarg_name = 'request_timeout'

    def _get_host_config_from_url(self, url):
        if False:
            i = 10
            return i + 15
        'Given a parsed URL, return the host configuration to be added to self.hosts'
        use_ssl = url.scheme == 'https'
        port = url.port or (443 if use_ssl else 80)
        return {'host': url.hostname, 'port': port, 'path_prefix': url.path, 'scheme': url.scheme}

    def _get_options_from_host_urls(self, urls):
        if False:
            print('Hello World!')
        "Given a list of parsed URLs, return a dict of additional options to be passed into the\n        Elasticsearch constructor; necessary for options that aren't valid as part of the 'hosts' config"
        opts = super()._get_options_from_host_urls(urls)
        basic_auth = (urls[0].username, urls[0].password)
        if any(((url.username, url.password) != basic_auth for url in urls)):
            raise ImproperlyConfigured('Elasticsearch host configuration is invalid. Elasticsearch 8 does not support multiple hosts with differing authentication credentials.')
        if basic_auth != (None, None):
            opts['basic_auth'] = basic_auth
        return opts
SearchBackend = Elasticsearch8SearchBackend