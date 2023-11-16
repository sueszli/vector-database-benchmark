"""
Query subclasses which provide extra functionality beyond simple data retrieval.
"""
from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
__all__ = ['DeleteQuery', 'UpdateQuery', 'InsertQuery', 'AggregateQuery']

class DeleteQuery(Query):
    """A DELETE SQL query."""
    compiler = 'SQLDeleteCompiler'

    def do_query(self, table, where, using):
        if False:
            i = 10
            return i + 15
        self.alias_map = {table: self.alias_map[table]}
        self.where = where
        cursor = self.get_compiler(using).execute_sql(CURSOR)
        if cursor:
            with cursor:
                return cursor.rowcount
        return 0

    def delete_batch(self, pk_list, using):
        if False:
            i = 10
            return i + 15
        '\n        Set up and execute delete queries for all the objects in pk_list.\n\n        More than one physical query may be executed if there are a\n        lot of values in pk_list.\n        '
        num_deleted = 0
        field = self.get_meta().pk
        for offset in range(0, len(pk_list), GET_ITERATOR_CHUNK_SIZE):
            self.clear_where()
            self.add_filter(f'{field.attname}__in', pk_list[offset:offset + GET_ITERATOR_CHUNK_SIZE])
            num_deleted += self.do_query(self.get_meta().db_table, self.where, using=using)
        return num_deleted

class UpdateQuery(Query):
    """An UPDATE SQL query."""
    compiler = 'SQLUpdateCompiler'

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._setup_query()

    def _setup_query(self):
        if False:
            print('Hello World!')
        '\n        Run on initialization and at the end of chaining. Any attributes that\n        would normally be set in __init__() should go here instead.\n        '
        self.values = []
        self.related_ids = None
        self.related_updates = {}

    def clone(self):
        if False:
            for i in range(10):
                print('nop')
        obj = super().clone()
        obj.related_updates = self.related_updates.copy()
        return obj

    def update_batch(self, pk_list, values, using):
        if False:
            i = 10
            return i + 15
        self.add_update_values(values)
        for offset in range(0, len(pk_list), GET_ITERATOR_CHUNK_SIZE):
            self.clear_where()
            self.add_filter('pk__in', pk_list[offset:offset + GET_ITERATOR_CHUNK_SIZE])
            self.get_compiler(using).execute_sql(NO_RESULTS)

    def add_update_values(self, values):
        if False:
            return 10
        '\n        Convert a dictionary of field name to value mappings into an update\n        query. This is the entry point for the public update() method on\n        querysets.\n        '
        values_seq = []
        for (name, val) in values.items():
            field = self.get_meta().get_field(name)
            direct = not (field.auto_created and (not field.concrete)) or not field.concrete
            model = field.model._meta.concrete_model
            if not direct or (field.is_relation and field.many_to_many):
                raise FieldError('Cannot update model field %r (only non-relations and foreign keys permitted).' % field)
            if model is not self.get_meta().concrete_model:
                self.add_related_update(model, field, val)
                continue
            values_seq.append((field, model, val))
        return self.add_update_fields(values_seq)

    def add_update_fields(self, values_seq):
        if False:
            for i in range(10):
                print('nop')
        '\n        Append a sequence of (field, model, value) triples to the internal list\n        that will be used to generate the UPDATE query. Might be more usefully\n        called add_update_targets() to hint at the extra information here.\n        '
        for (field, model, val) in values_seq:
            if field.generated:
                continue
            if hasattr(val, 'resolve_expression'):
                val = val.resolve_expression(self, allow_joins=False, for_save=True)
            self.values.append((field, model, val))

    def add_related_update(self, model, field, value):
        if False:
            while True:
                i = 10
        '\n        Add (name, value) to an update query for an ancestor model.\n\n        Update are coalesced so that only one update query per ancestor is run.\n        '
        self.related_updates.setdefault(model, []).append((field, None, value))

    def get_related_updates(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of query objects: one for each update required to an\n        ancestor model. Each query will have the same filtering conditions as\n        the current query but will only update a single table.\n        '
        if not self.related_updates:
            return []
        result = []
        for (model, values) in self.related_updates.items():
            query = UpdateQuery(model)
            query.values = values
            if self.related_ids is not None:
                query.add_filter('pk__in', self.related_ids[model])
            result.append(query)
        return result

class InsertQuery(Query):
    compiler = 'SQLInsertCompiler'

    def __init__(self, *args, on_conflict=None, update_fields=None, unique_fields=None, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.fields = []
        self.objs = []
        self.on_conflict = on_conflict
        self.update_fields = update_fields or []
        self.unique_fields = unique_fields or []

    def insert_values(self, fields, objs, raw=False):
        if False:
            return 10
        self.fields = fields
        self.objs = objs
        self.raw = raw

class AggregateQuery(Query):
    """
    Take another query as a parameter to the FROM clause and only select the
    elements in the provided list.
    """
    compiler = 'SQLAggregateCompiler'

    def __init__(self, model, inner_query):
        if False:
            while True:
                i = 10
        self.inner_query = inner_query
        super().__init__(model)