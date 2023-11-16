import json
from collections import OrderedDict
from copy import deepcopy
from urllib.parse import urlparse
from django.db import DEFAULT_DB_ALIAS, models
from django.db.models.sql import Query
from django.db.models.sql.constants import MULTI
from django.utils.crypto import get_random_string
from elasticsearch import VERSION as ELASTICSEARCH_VERSION
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk
from wagtail.search.backends.base import BaseSearchBackend, BaseSearchQueryCompiler, BaseSearchResults, FilterFieldError, get_model_root
from wagtail.search.index import AutocompleteField, FilterField, Indexed, RelatedFields, SearchField, class_is_indexed, get_indexed_models
from wagtail.search.query import And, Boost, Fuzzy, MatchAll, Not, Or, Phrase, PlainText
from wagtail.utils.utils import deep_update
use_new_elasticsearch_api = ELASTICSEARCH_VERSION >= (7, 15)

class Field:

    def __init__(self, field_name, boost=1):
        if False:
            i = 10
            return i + 15
        self.field_name = field_name
        self.boost = boost

    @property
    def field_name_with_boost(self):
        if False:
            for i in range(10):
                print('nop')
        if self.boost == 1:
            return self.field_name
        else:
            return f'{self.field_name}^{self.boost}'

class Elasticsearch7Mapping:
    all_field_name = '_all_text'
    edgengrams_field_name = '_edgengrams'
    type_map = {'AutoField': 'integer', 'BinaryField': 'binary', 'BooleanField': 'boolean', 'CharField': 'string', 'CommaSeparatedIntegerField': 'string', 'DateField': 'date', 'DateTimeField': 'date', 'DecimalField': 'double', 'FileField': 'string', 'FilePathField': 'string', 'FloatField': 'double', 'IntegerField': 'integer', 'BigIntegerField': 'long', 'IPAddressField': 'string', 'GenericIPAddressField': 'string', 'NullBooleanField': 'boolean', 'PositiveIntegerField': 'integer', 'PositiveSmallIntegerField': 'integer', 'SlugField': 'string', 'SmallIntegerField': 'integer', 'TextField': 'string', 'TimeField': 'date'}
    keyword_type = 'keyword'
    text_type = 'text'
    edgengram_analyzer_config = {'analyzer': 'edgengram_analyzer', 'search_analyzer': 'standard'}

    def __init__(self, model):
        if False:
            return 10
        self.model = model

    def get_parent(self):
        if False:
            return 10
        for base in self.model.__bases__:
            if issubclass(base, Indexed) and issubclass(base, models.Model):
                return type(self)(base)

    def get_document_type(self):
        if False:
            return 10
        return 'doc'

    def get_field_column_name(self, field):
        if False:
            while True:
                i = 10
        root_model = get_model_root(self.model)
        definition_model = field.get_definition_model(self.model)
        if definition_model != root_model:
            prefix = definition_model._meta.app_label.lower() + '_' + definition_model.__name__.lower() + '__'
        else:
            prefix = ''
        if isinstance(field, FilterField):
            return prefix + field.get_attname(self.model) + '_filter'
        elif isinstance(field, AutocompleteField):
            return prefix + field.get_attname(self.model) + '_edgengrams'
        elif isinstance(field, SearchField):
            return prefix + field.get_attname(self.model)
        elif isinstance(field, RelatedFields):
            return prefix + field.field_name

    def get_boost_field_name(self, boost):
        if False:
            print('Hello World!')
        boost = str(float(boost)).replace('.', '_')
        return f'{self.all_field_name}_boost_{boost}'

    def get_content_type(self):
        if False:
            print('Hello World!')
        '\n        Returns the content type as a string for the model.\n\n        For example: "wagtailcore.Page"\n                     "myapp.MyModel"\n        '
        return self.model._meta.app_label + '.' + self.model.__name__

    def get_all_content_types(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns all the content type strings that apply to this model.\n        This includes the models\' content type and all concrete ancestor\n        models that inherit from Indexed.\n\n        For example: ["myapp.MyPageModel", "wagtailcore.Page"]\n                     ["myapp.MyModel"]\n        '
        content_types = [self.get_content_type()]
        ancestor = self.get_parent()
        while ancestor:
            content_types.append(ancestor.get_content_type())
            ancestor = ancestor.get_parent()
        return content_types

    def get_field_mapping(self, field):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(field, RelatedFields):
            mapping = {'type': 'nested', 'properties': {}}
            nested_model = field.get_field(self.model).related_model
            nested_mapping = type(self)(nested_model)
            for sub_field in field.fields:
                (sub_field_name, sub_field_mapping) = nested_mapping.get_field_mapping(sub_field)
                mapping['properties'][sub_field_name] = sub_field_mapping
            return (self.get_field_column_name(field), mapping)
        else:
            mapping = {'type': self.type_map.get(field.get_type(self.model), 'string')}
            if isinstance(field, SearchField):
                if mapping['type'] == 'string':
                    mapping['type'] = self.text_type
                if field.boost:
                    mapping['boost'] = field.boost
                mapping['include_in_all'] = True
            if isinstance(field, AutocompleteField):
                mapping['type'] = self.text_type
                mapping.update(self.edgengram_analyzer_config)
            elif isinstance(field, FilterField):
                if mapping['type'] == 'string':
                    mapping['type'] = self.keyword_type
            if 'es_extra' in field.kwargs:
                for (key, value) in field.kwargs['es_extra'].items():
                    mapping[key] = value
            return (self.get_field_column_name(field), mapping)

    def get_mapping(self):
        if False:
            i = 10
            return i + 15
        fields = {'pk': {'type': self.keyword_type, 'store': True}, 'content_type': {'type': self.keyword_type}, self.edgengrams_field_name: {'type': self.text_type}}
        fields[self.edgengrams_field_name].update(self.edgengram_analyzer_config)
        for field in self.model.get_search_fields():
            (key, val) = self.get_field_mapping(field)
            fields[key] = val
        fields[self.all_field_name] = {'type': 'text'}
        unique_boosts = set()

        def replace_include_in_all(properties):
            if False:
                i = 10
                return i + 15
            for field_mapping in properties.values():
                if 'include_in_all' in field_mapping:
                    if field_mapping['include_in_all']:
                        field_mapping['copy_to'] = self.all_field_name
                        if 'boost' in field_mapping:
                            unique_boosts.add(field_mapping['boost'])
                            field_mapping['copy_to'] = [field_mapping['copy_to'], self.get_boost_field_name(field_mapping['boost'])]
                            del field_mapping['boost']
                    del field_mapping['include_in_all']
                if field_mapping['type'] == 'nested':
                    replace_include_in_all(field_mapping['properties'])
        replace_include_in_all(fields)
        for boost in unique_boosts:
            fields[self.get_boost_field_name(boost)] = {'type': 'text'}
        return {'properties': fields}

    def get_document_id(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return str(obj.pk)

    def _get_nested_document(self, fields, obj):
        if False:
            for i in range(10):
                print('nop')
        doc = {}
        edgengrams = []
        model = type(obj)
        mapping = type(self)(model)
        for field in fields:
            value = field.get_value(obj)
            doc[mapping.get_field_column_name(field)] = value
            if isinstance(field, AutocompleteField):
                edgengrams.append(value)
        return (doc, edgengrams)

    def get_document(self, obj):
        if False:
            return 10
        doc = {'pk': str(obj.pk), 'content_type': self.get_all_content_types()}
        edgengrams = []
        for field in self.model.get_search_fields():
            value = field.get_value(obj)
            if isinstance(field, RelatedFields):
                if isinstance(value, (models.Manager, models.QuerySet)):
                    nested_docs = []
                    for nested_obj in value.all():
                        (nested_doc, extra_edgengrams) = self._get_nested_document(field.fields, nested_obj)
                        nested_docs.append(nested_doc)
                        edgengrams.extend(extra_edgengrams)
                    value = nested_docs
                elif isinstance(value, models.Model):
                    (value, extra_edgengrams) = self._get_nested_document(field.fields, value)
                    edgengrams.extend(extra_edgengrams)
            elif isinstance(field, FilterField):
                if isinstance(value, (models.Manager, models.QuerySet)):
                    value = list(value.values_list('pk', flat=True))
                elif isinstance(value, models.Model):
                    value = value.pk
                elif isinstance(value, (list, tuple)):
                    value = [item.pk if isinstance(item, models.Model) else item for item in value]
            doc[self.get_field_column_name(field)] = value
            if isinstance(field, AutocompleteField):
                edgengrams.append(value)
        doc[self.edgengrams_field_name] = edgengrams
        return doc

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'<ElasticsearchMapping: {self.model.__name__}>'

class Elasticsearch7Index:

    def __init__(self, backend, name):
        if False:
            while True:
                i = 10
        self.backend = backend
        self.es = backend.es
        self.mapping_class = backend.mapping_class
        self.name = name
    if use_new_elasticsearch_api:

        def put(self):
            if False:
                print('Hello World!')
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
                print('Hello World!')
            self.es.indices.refresh(index=self.name)
    else:

        def put(self):
            if False:
                return 10
            self.es.indices.create(self.name, self.backend.settings)

        def delete(self):
            if False:
                print('Hello World!')
            try:
                self.es.indices.delete(self.name)
            except NotFoundError:
                pass

        def refresh(self):
            if False:
                return 10
            self.es.indices.refresh(self.name)

    def exists(self):
        if False:
            return 10
        return self.es.indices.exists(self.name)

    def is_alias(self):
        if False:
            while True:
                i = 10
        return self.es.indices.exists_alias(name=self.name)

    def aliased_indices(self):
        if False:
            while True:
                i = 10
        '\n        If this index object represents an alias (which appear the same in the\n        Elasticsearch API), this method can be used to fetch the list of indices\n        the alias points to.\n\n        Use the is_alias method if you need to find out if this an alias. This\n        returns an empty list if called on an index.\n        '
        return [self.backend.index_class(self.backend, index_name) for index_name in self.es.indices.get_alias(name=self.name).keys()]

    def put_alias(self, name):
        if False:
            while True:
                i = 10
        '\n        Creates a new alias to this index. If the alias already exists it will\n        be repointed to this index.\n        '
        self.es.indices.put_alias(name=name, index=self.name)

    def add_model(self, model):
        if False:
            while True:
                i = 10
        mapping = self.mapping_class(model)
        self.es.indices.put_mapping(index=self.name, body=mapping.get_mapping())
    if use_new_elasticsearch_api:

        def add_item(self, item):
            if False:
                while True:
                    i = 10
            if not class_is_indexed(item.__class__):
                return
            mapping = self.mapping_class(item.__class__)
            self.es.index(index=self.name, document=mapping.get_document(item), id=mapping.get_document_id(item))
    else:

        def add_item(self, item):
            if False:
                for i in range(10):
                    print('nop')
            if not class_is_indexed(item.__class__):
                return
            mapping = self.mapping_class(item.__class__)
            self.es.index(self.name, mapping.get_document(item), id=mapping.get_document_id(item))

    def add_items(self, model, items):
        if False:
            return 10
        if not class_is_indexed(model):
            return
        mapping = self.mapping_class(model)
        actions = []
        for item in items:
            action = {'_id': mapping.get_document_id(item)}
            action.update(mapping.get_document(item))
            actions.append(action)
        bulk(self.es, actions, index=self.name)

    def delete_item(self, item):
        if False:
            while True:
                i = 10
        if not class_is_indexed(item.__class__):
            return
        mapping = self.mapping_class(item.__class__)
        try:
            self.es.delete(index=self.name, id=mapping.get_document_id(item))
        except NotFoundError:
            pass

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.delete()
        self.put()

class Elasticsearch7SearchQueryCompiler(BaseSearchQueryCompiler):
    mapping_class = Elasticsearch7Mapping
    DEFAULT_OPERATOR = 'or'

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.mapping = self.mapping_class(self.queryset.model)
        self.remapped_fields = self._remap_fields(self.fields)

    def _remap_fields(self, fields):
        if False:
            for i in range(10):
                print('nop')
        'Convert field names into index column names and add boosts.'
        remapped_fields = []
        if fields:
            searchable_fields = {f.field_name: f for f in self.get_searchable_fields()}
            for field_name in fields:
                field = searchable_fields.get(field_name)
                if field:
                    field_name = self.mapping.get_field_column_name(field)
                    remapped_fields.append(Field(field_name, field.boost or 1))
        else:
            remapped_fields.append(Field(self.mapping.all_field_name))
            models = get_indexed_models()
            unique_boosts = set()
            for model in models:
                if not issubclass(model, self.queryset.model):
                    continue
                for field in model.get_searchable_search_fields():
                    if field.boost:
                        unique_boosts.add(float(field.boost))
            remapped_fields.extend([Field(self.mapping.get_boost_field_name(boost), boost) for boost in unique_boosts])
        return remapped_fields

    def _process_lookup(self, field, lookup, value):
        if False:
            while True:
                i = 10
        column_name = self.mapping.get_field_column_name(field)
        if lookup == 'exact':
            if value is None:
                return {'missing': {'field': column_name}}
            else:
                return {'term': {column_name: value}}
        if lookup == 'isnull':
            query = {'exists': {'field': column_name}}
            if value:
                query = {'bool': {'mustNot': query}}
            return query
        if lookup in ['startswith', 'prefix']:
            return {'prefix': {column_name: value}}
        if lookup in ['gt', 'gte', 'lt', 'lte']:
            return {'range': {column_name: {lookup: value}}}
        if lookup == 'range':
            (lower, upper) = value
            return {'range': {column_name: {'gte': lower, 'lte': upper}}}
        if lookup == 'in':
            if isinstance(value, Query):
                db_alias = self.queryset._db or DEFAULT_DB_ALIAS
                resultset = value.get_compiler(db_alias).execute_sql(result_type=MULTI)
                value = [row[0] for chunk in resultset for row in chunk]
            elif not isinstance(value, list):
                value = list(value)
            return {'terms': {column_name: value}}

    def _connect_filters(self, filters, connector, negated):
        if False:
            for i in range(10):
                print('nop')
        if filters:
            if len(filters) == 1:
                filter_out = filters[0]
            elif connector == 'AND':
                filter_out = {'bool': {'must': [fil for fil in filters if fil is not None]}}
            elif connector == 'OR':
                filter_out = {'bool': {'should': [fil for fil in filters if fil is not None]}}
            if negated:
                filter_out = {'bool': {'mustNot': filter_out}}
            return filter_out

    def _compile_plaintext_query(self, query, fields, boost=1.0):
        if False:
            i = 10
            return i + 15
        match_query = {'query': query.query_string}
        if query.operator != 'or':
            match_query['operator'] = query.operator
        if len(fields) == 1:
            if boost != 1.0 or fields[0].boost != 1.0:
                match_query['boost'] = boost * fields[0].boost
            return {'match': {fields[0].field_name: match_query}}
        else:
            if boost != 1.0:
                match_query['boost'] = boost
            match_query['fields'] = [field.field_name_with_boost for field in fields]
            return {'multi_match': match_query}

    def _compile_fuzzy_query(self, query, fields):
        if False:
            print('Hello World!')
        match_query = {'query': query.query_string, 'fuzziness': 'AUTO'}
        if len(fields) == 1:
            if fields[0].boost != 1.0:
                match_query['boost'] = fields[0].boost
            return {'match': {fields[0].field_name: match_query}}
        else:
            match_query['fields'] = [field.field_name_with_boost for field in fields]
            return {'multi_match': match_query}

    def _compile_phrase_query(self, query, fields):
        if False:
            while True:
                i = 10
        if len(fields) == 1:
            if fields[0].boost != 1.0:
                return {'match_phrase': {fields[0].field_name: {'query': query.query_string, 'boost': fields[0].boost}}}
            else:
                return {'match_phrase': {fields[0].field_name: query.query_string}}
        else:
            return {'multi_match': {'query': query.query_string, 'fields': [field.field_name_with_boost for field in fields], 'type': 'phrase'}}

    def _compile_query(self, query, field, boost=1.0):
        if False:
            print('Hello World!')
        if isinstance(query, MatchAll):
            match_all_query = {}
            if boost != 1.0:
                match_all_query['boost'] = boost
            return {'match_all': match_all_query}
        elif isinstance(query, And):
            return {'bool': {'must': [self._compile_query(child_query, field, boost) for child_query in query.subqueries]}}
        elif isinstance(query, Or):
            return {'bool': {'should': [self._compile_query(child_query, field, boost) for child_query in query.subqueries]}}
        elif isinstance(query, Not):
            return {'bool': {'mustNot': self._compile_query(query.subquery, field, boost)}}
        elif isinstance(query, PlainText):
            return self._compile_plaintext_query(query, [field], boost)
        elif isinstance(query, Fuzzy):
            return self._compile_fuzzy_query(query, [field])
        elif isinstance(query, Phrase):
            return self._compile_phrase_query(query, [field])
        elif isinstance(query, Boost):
            return self._compile_query(query.subquery, field, boost * query.boost)
        else:
            raise NotImplementedError('`%s` is not supported by the Elasticsearch search backend.' % query.__class__.__name__)

    def get_inner_query(self):
        if False:
            return 10
        if self.remapped_fields:
            fields = self.remapped_fields
        else:
            fields = [self.mapping.all_field_name]
        if len(fields) == 0:
            return {'bool': {'mustNot': {'match_all': {}}}}
        if isinstance(self.query, MatchAll):
            return {'match_all': {}}
        elif isinstance(self.query, PlainText):
            return self._compile_plaintext_query(self.query, fields)
        elif isinstance(self.query, Phrase):
            return self._compile_phrase_query(self.query, fields)
        elif isinstance(self.query, Fuzzy):
            return self._compile_fuzzy_query(self.query, fields)
        elif isinstance(self.query, Not):
            return {'bool': {'mustNot': [self._compile_query(self.query.subquery, field) for field in fields]}}
        else:
            return self._join_and_compile_queries(self.query, fields)

    def _join_and_compile_queries(self, query, fields, boost=1.0):
        if False:
            i = 10
            return i + 15
        if len(fields) == 1:
            return self._compile_query(query, fields[0], boost)
        else:
            field_queries = []
            for field in fields:
                field_queries.append(self._compile_query(query, field, boost))
            return {'dis_max': {'queries': field_queries}}

    def get_content_type_filter(self):
        if False:
            return 10
        content_type = self.mapping_class(self.queryset.model).get_content_type()
        return {'match': {'content_type': content_type}}

    def get_filters(self):
        if False:
            i = 10
            return i + 15
        filters = [self.get_content_type_filter()]
        queryset_filters = self._get_filters_from_queryset()
        if queryset_filters:
            filters.append(queryset_filters)
        return filters

    def get_query(self):
        if False:
            while True:
                i = 10
        inner_query = self.get_inner_query()
        filters = self.get_filters()
        if len(filters) == 1:
            return {'bool': {'must': inner_query, 'filter': filters[0]}}
        elif len(filters) > 1:
            return {'bool': {'must': inner_query, 'filter': filters}}
        else:
            return inner_query

    def get_searchable_fields(self):
        if False:
            return 10
        return self.queryset.model.get_searchable_search_fields()

    def get_sort(self):
        if False:
            while True:
                i = 10
        if self.order_by_relevance:
            return
        if self.queryset.ordered:
            sort = []
            for (reverse, field) in self._get_order_by():
                column_name = self.mapping.get_field_column_name(field)
                sort.append({column_name: 'desc' if reverse else 'asc'})
            return sort
        else:
            return ['pk']

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return json.dumps(self.get_query())

class Elasticsearch7SearchResults(BaseSearchResults):
    fields_param_name = 'stored_fields'
    supports_facet = True

    def facet(self, field_name):
        if False:
            for i in range(10):
                print('nop')
        field = self.query_compiler._get_filterable_field(field_name)
        if field is None:
            raise FilterFieldError('Cannot facet search results with field "' + field_name + '". Please add index.FilterField(\'' + field_name + "') to " + self.query_compiler.queryset.model.__name__ + '.search_fields.', field_name=field_name)
        body = self._get_es_body()
        column_name = self.query_compiler.mapping.get_field_column_name(field)
        body['aggregations'] = {field_name: {'terms': {'field': column_name, 'missing': 0}}}
        response = self._backend_do_search(body, index=self.backend.get_index_for_model(self.query_compiler.queryset.model).name, size=0)
        return OrderedDict([(bucket['key'] if bucket['key'] != 0 else None, bucket['doc_count']) for bucket in response['aggregations'][field_name]['buckets']])

    def _get_es_body(self, for_count=False):
        if False:
            for i in range(10):
                print('nop')
        body = {'query': self.query_compiler.get_query()}
        if not for_count:
            sort = self.query_compiler.get_sort()
            if sort is not None:
                body['sort'] = sort
        return body

    def _get_results_from_hits(self, hits):
        if False:
            while True:
                i = 10
        '\n        Yields Django model instances from a page of hits returned by Elasticsearch\n        '
        pks = [hit['fields']['pk'][0] for hit in hits]
        scores = {str(hit['fields']['pk'][0]): hit['_score'] for hit in hits}
        results = {str(pk): None for pk in pks}
        for obj in self.query_compiler.queryset.filter(pk__in=pks):
            results[str(obj.pk)] = obj
            if self._score_field:
                setattr(obj, self._score_field, scores.get(str(obj.pk)))
        for pk in pks:
            result = results[str(pk)]
            if result:
                yield result
    if use_new_elasticsearch_api:

        def _backend_do_search(self, body, **kwargs):
            if False:
                print('Hello World!')
            return self.backend.es.search(**body, **kwargs)
    else:

        def _backend_do_search(self, body, **kwargs):
            if False:
                while True:
                    i = 10
            return self.backend.es.search(body=body, **kwargs)

    def _do_search(self):
        if False:
            i = 10
            return i + 15
        PAGE_SIZE = 100
        if self.stop is not None:
            limit = self.stop - self.start
        else:
            limit = None
        use_scroll = limit is None or limit > PAGE_SIZE
        body = self._get_es_body()
        params = {'index': self.backend.get_index_for_model(self.query_compiler.queryset.model).name, '_source': False, self.fields_param_name: 'pk'}
        if use_scroll:
            params.update({'scroll': '2m', 'size': PAGE_SIZE})
            skip = self.start
            page = self._backend_do_search(body, **params)
            while True:
                hits = page['hits']['hits']
                if len(hits) == 0:
                    break
                if skip < len(hits):
                    for result in self._get_results_from_hits(hits):
                        if limit is not None and limit == 0:
                            break
                        if skip == 0:
                            yield result
                            if limit is not None:
                                limit -= 1
                        else:
                            skip -= 1
                    if limit is not None and limit == 0:
                        break
                else:
                    skip -= len(hits)
                if '_scroll_id' not in page:
                    break
                page = self.backend.es.scroll(scroll_id=page['_scroll_id'], scroll='2m')
            if '_scroll_id' in page:
                self.backend.es.clear_scroll(scroll_id=page['_scroll_id'])
        else:
            params.update({'from_': self.start, 'size': limit or PAGE_SIZE})
            hits = self._backend_do_search(body, **params)['hits']['hits']
            for result in self._get_results_from_hits(hits):
                yield result

    def _do_count(self):
        if False:
            while True:
                i = 10
        hit_count = self.backend.es.count(index=self.backend.get_index_for_model(self.query_compiler.queryset.model).name, body=self._get_es_body(for_count=True))['count']
        hit_count -= self.start
        if self.stop is not None:
            hit_count = min(hit_count, self.stop - self.start)
        return max(hit_count, 0)

class ElasticsearchAutocompleteQueryCompilerImpl:

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        if self.fields:
            fields = []
            autocomplete_fields = {f.field_name: f for f in self.queryset.model.get_autocomplete_search_fields()}
            for field_name in self.fields:
                if field_name in autocomplete_fields:
                    field_name = self.mapping.get_field_column_name(autocomplete_fields[field_name])
                fields.append(field_name)
            self.remapped_fields = fields
        else:
            self.remapped_fields = None

    def get_inner_query(self):
        if False:
            i = 10
            return i + 15
        fields = self.remapped_fields or [self.mapping.edgengrams_field_name]
        fields = [Field(field) for field in fields]
        if len(fields) == 0:
            return {'bool': {'mustNot': {'match_all': {}}}}
        return self._compile_plaintext_query(self.query, fields)

class Elasticsearch7AutocompleteQueryCompiler(ElasticsearchAutocompleteQueryCompilerImpl, Elasticsearch7SearchQueryCompiler):
    pass

class ElasticsearchIndexRebuilder:

    def __init__(self, index):
        if False:
            return 10
        self.index = index

    def reset_index(self):
        if False:
            i = 10
            return i + 15
        self.index.reset()

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        self.reset_index()
        return self.index

    def finish(self):
        if False:
            for i in range(10):
                print('nop')
        self.index.refresh()

class ElasticsearchAtomicIndexRebuilder(ElasticsearchIndexRebuilder):

    def __init__(self, index):
        if False:
            print('Hello World!')
        self.alias = index
        self.index = index.backend.index_class(index.backend, self.alias.name + '_' + get_random_string(7).lower())

    def reset_index(self):
        if False:
            print('Hello World!')
        self.alias.delete()
        self.index.put()
        self.index.put_alias(self.alias.name)

    def start(self):
        if False:
            return 10
        self.index.put()
        return self.index

    def finish(self):
        if False:
            print('Hello World!')
        self.index.refresh()
        if self.alias.is_alias():
            old_index = self.alias.aliased_indices()
            self.index.put_alias(self.alias.name)
            for index in old_index:
                if index.name != self.index.name:
                    index.delete()
        else:
            self.alias.delete()
            self.index.put_alias(self.alias.name)

class Elasticsearch7SearchBackend(BaseSearchBackend):
    mapping_class = Elasticsearch7Mapping
    index_class = Elasticsearch7Index
    query_compiler_class = Elasticsearch7SearchQueryCompiler
    autocomplete_query_compiler_class = Elasticsearch7AutocompleteQueryCompiler
    results_class = Elasticsearch7SearchResults
    basic_rebuilder_class = ElasticsearchIndexRebuilder
    atomic_rebuilder_class = ElasticsearchAtomicIndexRebuilder
    catch_indexing_errors = True
    timeout_kwarg_name = 'timeout'
    settings = {'settings': {'analysis': {'analyzer': {'ngram_analyzer': {'type': 'custom', 'tokenizer': 'lowercase', 'filter': ['asciifolding', 'ngram']}, 'edgengram_analyzer': {'type': 'custom', 'tokenizer': 'lowercase', 'filter': ['asciifolding', 'edgengram']}}, 'tokenizer': {'ngram_tokenizer': {'type': 'ngram', 'min_gram': 3, 'max_gram': 15}, 'edgengram_tokenizer': {'type': 'edge_ngram', 'min_gram': 2, 'max_gram': 15, 'side': 'front'}}, 'filter': {'ngram': {'type': 'ngram', 'min_gram': 3, 'max_gram': 15}, 'edgengram': {'type': 'edge_ngram', 'min_gram': 1, 'max_gram': 15}}}, 'index': {'max_ngram_diff': 12}}}

    def _get_host_config_from_url(self, url):
        if False:
            while True:
                i = 10
        'Given a parsed URL, return the host configuration to be added to self.hosts'
        use_ssl = url.scheme == 'https'
        port = url.port or (443 if use_ssl else 80)
        http_auth = None
        if url.username is not None and url.password is not None:
            http_auth = (url.username, url.password)
        return {'host': url.hostname, 'port': port, 'url_prefix': url.path, 'use_ssl': use_ssl, 'verify_certs': use_ssl, 'http_auth': http_auth}

    def _get_options_from_host_urls(self, urls):
        if False:
            while True:
                i = 10
        "Given a list of parsed URLs, return a dict of additional options to be passed into the\n        Elasticsearch constructor; necessary for options that aren't valid as part of the 'hosts' config\n        "
        return {}

    def __init__(self, params):
        if False:
            while True:
                i = 10
        super().__init__(params)
        self.hosts = params.pop('HOSTS', None)
        self.index_name = params.pop('INDEX', 'wagtail')
        self.timeout = params.pop('TIMEOUT', 10)
        if params.pop('ATOMIC_REBUILD', False):
            self.rebuilder_class = self.atomic_rebuilder_class
        else:
            self.rebuilder_class = self.basic_rebuilder_class
        self.settings = deepcopy(self.settings)
        self.settings = deep_update(self.settings, params.pop('INDEX_SETTINGS', {}))
        options = params.pop('OPTIONS', {})
        if self.hosts is None:
            es_urls = params.pop('URLS', ['http://localhost:9200'])
            if isinstance(es_urls, str):
                es_urls = [es_urls]
            parsed_urls = [urlparse(url) for url in es_urls]
            self.hosts = [self._get_host_config_from_url(url) for url in parsed_urls]
            options.update(self._get_options_from_host_urls(parsed_urls))
        options[self.timeout_kwarg_name] = self.timeout
        self.es = Elasticsearch(hosts=self.hosts, **options)

    def get_index_for_model(self, model):
        if False:
            i = 10
            return i + 15
        root_model = get_model_root(model)
        index_suffix = '__' + root_model._meta.app_label.lower() + '_' + root_model.__name__.lower()
        return self.index_class(self, self.index_name + index_suffix)

    def get_index(self):
        if False:
            for i in range(10):
                print('nop')
        return self.index_class(self, self.index_name)

    def get_rebuilder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rebuilder_class(self.get_index())

    def reset_index(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_rebuilder().reset_index()
SearchBackend = Elasticsearch7SearchBackend