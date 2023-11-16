import re
from elasticsearch_dsl import Q
SEARCH_FIELDS = ['author', 'author_email', 'description', 'download_url', 'home_page', 'keywords', 'license', 'maintainer', 'maintainer_email', 'normalized_name', 'platform', 'summary']
SEARCH_BOOSTS = {'name': 10, 'normalized_name': 10, 'description': 5, 'keywords': 5, 'summary': 5}
SEARCH_FILTER_ORDER = ('Framework', 'Topic', 'Development Status', 'License', 'Programming Language', 'Operating System', 'Environment', 'Intended Audience', 'Natural Language')

def get_es_query(es, terms, order, classifiers):
    if False:
        i = 10
        return i + 15
    '\n    Returns an Elasticsearch query from data from the request.\n    '
    classifier_q = Q('bool', must=[Q('bool', should=[Q('term', classifiers=classifier), Q('prefix', classifiers=classifier + ' :: ')]) for classifier in classifiers])
    if not terms:
        query = es.query(classifier_q) if classifiers else es.query()
    else:
        (quoted_string, unquoted_string) = filter_query(terms)
        bool_query = Q('bool', must=[form_query('phrase', i) for i in quoted_string] + [form_query('best_fields', i) for i in unquoted_string] + ([classifier_q] if classifiers else []))
        if len(terms) > 1:
            bool_query = bool_query | Q('prefix', normalized_name=terms)
        query = es.query(bool_query)
        query = query.suggest('name_suggestion', terms, term={'field': 'name'})
    query = query_for_order(query, order)
    return query

def filter_query(s):
    if False:
        i = 10
        return i + 15
    '\n    Filters given query with the below regex\n    and returns lists of quoted and unquoted strings\n    '
    matches = re.findall('(?:"([^"]*)")|([^"]*)', s)
    result_quoted = [t[0].strip() for t in matches if t[0]]
    result_unquoted = [t[1].strip() for t in matches if t[1]]
    return (result_quoted, result_unquoted)

def form_query(query_type, query):
    if False:
        i = 10
        return i + 15
    '\n    Returns a multi match query\n    '
    fields = [field + '^' + str(SEARCH_BOOSTS[field]) if field in SEARCH_BOOSTS else field for field in SEARCH_FIELDS]
    return Q('multi_match', fields=fields, query=query, type=query_type)

def query_for_order(query, order):
    if False:
        while True:
            i = 10
    '\n    Applies transformations on the ES query based on the search order.\n\n    Order is assumed to be a string with the name of a field with an optional\n    hyphen to indicate descending sort order.\n    '
    if order == '':
        return query
    field = order[order.find('-') + 1:]
    sort_info = {field: {'order': 'desc' if order.startswith('-') else 'asc', 'unmapped_type': 'long'}}
    query = query.sort(sort_info)
    return query