import re
import structlog
from django.conf import settings
from elasticsearch import Elasticsearch
from elasticsearch_dsl import FacetedSearch, TermsFacet
from elasticsearch_dsl.query import Bool, FunctionScore, MultiMatch, Nested, SimpleQueryString, Term, Terms, Wildcard
from readthedocs.search.documents import PageDocument, ProjectDocument
log = structlog.get_logger(__name__)

class RTDFacetedSearch(FacetedSearch):
    """Custom wrapper around FacetedSearch."""
    operators = ['and', 'or']
    excludes = []
    _highlight_options = {'encoder': 'html', 'number_of_fragments': 1, 'pre_tags': ['<span>'], 'post_tags': ['</span>']}

    def __init__(self, query=None, filters=None, projects=None, aggregate_results=True, use_advanced_query=True, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Custom wrapper around FacetedSearch.\n\n        :param string query: Query to search for\n        :param dict filters: Filters to be used with the query.\n        :param projects: A dictionary of project slugs mapped to a `VersionData` object.\n         Or a list of project slugs.\n         Results are filter with these values.\n        :param use_advanced_query: If `True` forces to always use\n         `SimpleQueryString` for the text query object.\n        :param bool aggregate_results: If results should be aggregated,\n         this is returning the number of results within other facets.\n        :param bool use_advanced_query: Always use SimpleQueryString.\n         Set this to `False` to use the experimental fuzzy search.\n        '
        self.use_advanced_query = use_advanced_query
        self.aggregate_results = aggregate_results
        self.projects = projects or {}
        log.debug('Hacking Elastic to fix search connection pooling')
        self.using = Elasticsearch(**settings.ELASTICSEARCH_DSL['default'])
        filters = filters or {}
        valid_filters = {k: v for (k, v) in filters.items() if k in self.facets}
        super().__init__(query=query, filters=valid_filters, **kwargs)

    def _get_queries(self, *, query, fields):
        if False:
            i = 10
            return i + 15
        '\n        Get a list of query objects according to the query.\n\n        If the query is a single term we try to match partial words and substrings\n        (available only with the DEFAULT_TO_FUZZY_SEARCH feature flag),\n        otherwise we use the SimpleQueryString query.\n        '
        get_queries_function = self._get_single_term_queries if self._is_single_term(query) else self._get_text_queries
        return get_queries_function(query=query, fields=fields)

    def _get_text_queries(self, *, query, fields):
        if False:
            return 10
        '\n        Returns a list of query objects according to the query.\n\n        SimpleQueryString provides a syntax to let advanced users manipulate\n        the results explicitly.\n\n        We need to search for both "and" and "or" operators.\n        The score of "and" should be higher as it satisfies both "or" and "and".\n\n        For valid options, see:\n\n        - https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-simple-query-string-query.html  # noqa\n        '
        queries = []
        is_advanced_query = self.use_advanced_query or self._is_advanced_query(query)
        for operator in self.operators:
            if is_advanced_query:
                query_string = SimpleQueryString(query=query, fields=fields, default_operator=operator)
            else:
                query_string = self._get_fuzzy_query(query=query, fields=fields, operator=operator)
            queries.append(query_string)
        return queries

    def _get_single_term_queries(self, query, fields):
        if False:
            while True:
                i = 10
        '\n        Returns a list of query objects for fuzzy and partial results.\n\n        We need to search for both "and" and "or" operators.\n        The score of "and" should be higher as it satisfies both "or" and "and".\n\n        We use the Wildcard query with the query suffixed by ``*`` to match substrings.\n\n        For valid options, see:\n\n        - https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-wildcard-query.html  # noqa\n\n        .. note::\n\n           Doing a prefix **and** suffix search is slow on big indexes like ours.\n        '
        query_string = self._get_fuzzy_query(query=query, fields=fields)
        queries = [query_string]
        for field in fields:
            field = re.sub('\\^.*$', '', field)
            kwargs = {field: {'value': f'{query}*'}}
            queries.append(Wildcard(**kwargs))
        return queries

    def _get_fuzzy_query(self, *, query, fields, operator='or'):
        if False:
            while True:
                i = 10
        '\n        Returns a query object used for fuzzy results.\n\n        For valid options, see:\n\n        - https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-match-query.html\n        '
        return MultiMatch(query=query, fields=fields, operator=operator, fuzziness='AUTO:4,6', prefix_length=1)

    def _is_single_term(self, query):
        if False:
            return 10
        "\n        Check if the query is a single term.\n\n        A query is a single term if it is a single word,\n        if it doesn't contain the syntax from a simple query string,\n        and if `self.use_advanced_query` is False.\n        "
        is_single_term = not self.use_advanced_query and query and (len(query.split()) <= 1) and (not self._is_advanced_query(query))
        return is_single_term

    def _is_advanced_query(self, query):
        if False:
            for i in range(10):
                print('nop')
        "\n        Check if query looks like to be using the syntax from a simple query string.\n\n        .. note::\n\n           We don't check if the syntax is valid.\n           The tokens used aren't very common in a normal query, so checking if\n           the query contains any of them should be enough to determinate if\n           it's an advanced query.\n\n        Simple query syntax:\n\n        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-simple-query-string-query.html#simple-query-string-syntax\n        "
        tokens = {'+', '|', '-', '"', '*', '(', ')', '~'}
        query_tokens = set(query)
        return not tokens.isdisjoint(query_tokens)

    def aggregate(self, search):
        if False:
            i = 10
            return i + 15
        'Overridden to decide if we should aggregate or not.'
        if self.aggregate_results:
            super().aggregate(search)

class ProjectSearch(RTDFacetedSearch):
    facets = {'language': TermsFacet(field='language')}
    doc_types = [ProjectDocument]
    index = ProjectDocument._index._name
    fields = ('name^10', 'slug^5', 'description')
    excludes = ['users', 'language']

    def query(self, search, query):
        if False:
            print('Hello World!')
        '\n        Customize search results to support extra functionality.\n\n        If `self.projects` was given, we use it to filter the documents.\n        Only filtering by a list of slugs is supported.\n\n        Also:\n\n        * Adds SimpleQueryString with `self.operators` instead of default query.\n        * Adds HTML encoding of results to avoid XSS issues.\n        '
        search = search.highlight_options(**self._highlight_options)
        search = search.source(excludes=self.excludes)
        queries = self._get_queries(query=query, fields=self.fields)
        bool_query = Bool(should=queries)
        if self.projects:
            if isinstance(self.projects, list):
                projects_query = Bool(filter=Terms(slug=self.projects))
                bool_query = Bool(must=[bool_query, projects_query])
            else:
                raise ValueError('projects must be a list!')
        search = search.query(bool_query)
        return search

class PageSearch(RTDFacetedSearch):
    facets = {'project': TermsFacet(field='project')}
    doc_types = [PageDocument]
    index = PageDocument._index._name
    _outer_fields = ['title^1.5']
    _section_fields = ['sections.title^2', 'sections.content']
    fields = _outer_fields
    excludes = ['rank', 'sections', 'commit', 'build']

    def _get_projects_query(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get filter by projects query.\n\n        If it's a dict, filter by project and version,\n        if it's a list filter by project.\n        "
        if not self.projects:
            return None
        if isinstance(self.projects, dict):
            versions_query = [Bool(filter=[Term(project=project), Term(version=version)]) for (project, version) in self.projects.items()]
            return Bool(should=versions_query)
        if isinstance(self.projects, list):
            return Bool(filter=Terms(project=self.projects))
        raise ValueError('projects must be a list or a dict!')

    def query(self, search, query):
        if False:
            i = 10
            return i + 15
        '\n        Manipulates the query to support nested queries and a custom rank for pages.\n\n        If `self.projects` was given, we use it to filter the documents that\n        match the same project and version.\n        '
        search = search.highlight_options(**self._highlight_options)
        search = search.source(excludes=self.excludes)
        queries = self._get_queries(query=query, fields=self.fields)
        sections_nested_query = self._get_nested_query(query=query, path='sections', fields=self._section_fields)
        queries.append(sections_nested_query)
        bool_query = Bool(should=queries)
        projects_query = self._get_projects_query()
        if projects_query:
            bool_query = Bool(must=[bool_query, projects_query])
        final_query = FunctionScore(query=bool_query, script_score=self._get_script_score())
        search = search.query(final_query)
        return search

    def _get_nested_query(self, *, query, path, fields):
        if False:
            i = 10
            return i + 15
        'Generate a nested query with passed parameters.'
        queries = self._get_queries(query=query, fields=fields)
        bool_query = Bool(should=queries)
        raw_fields = [re.sub('\\^.*$', '', field) for field in fields]
        highlight = dict(self._highlight_options, fields={field: {} for field in raw_fields})
        return Nested(path=path, inner_hits={'highlight': highlight}, query=bool_query)

    def _get_script_score(self):
        if False:
            while True:
                i = 10
        '\n        Gets an ES script to map the page rank to a valid score weight.\n\n        ES expects the rank to be a number greater than 0,\n        but users can set this between [-10, +10].\n        We map that range to [0.01, 2] (21 possible values).\n\n        The first lower rank (0.8) needs to bring the score from the highest boost (sections.title^2)\n        close to the lowest boost (title^1.5), that way exact results take priority:\n\n        - 2.0 * 0.8 = 1.6 (score close to 1.5, but not lower than it)\n        - 1.5 * 0.8 = 1.2 (score lower than 1.5)\n\n        The first higher rank (1.2) needs to bring the score from the lowest boost (title^1.5)\n        close to the highest boost (sections.title^2), that way exact results take priority:\n\n        - 2.0 * 1.3 = 2.6 (score higher thank 2.0)\n        - 1.5 * 1.3 = 1.95 (score close to 2.0, but not higher than it)\n\n        The next lower and higher ranks need to decrease/increase both scores.\n\n        See https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-script-score-query.html#field-value-factor  # noqa\n        '
        ranking = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.93, 1.96, 2]
        source = "\n            int rank = doc['rank'].size() == 0 ? 0 : (int) doc['rank'].value;\n            return params.ranking[rank + 10] * _score;\n        "
        return {'script': {'source': source, 'params': {'ranking': ranking}}}