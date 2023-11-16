from __future__ import annotations
import itertools
from datetime import timedelta
from typing import Any, Generator
from uuid import uuid4
from django.db import connections, router
from django.utils import timezone

class BulkDeleteQuery:

    def __init__(self, model, project_id=None, dtfield=None, days=None, order_by=None):
        if False:
            print('Hello World!')
        self.model = model
        self.project_id = int(project_id) if project_id else None
        self.dtfield = dtfield
        self.days = int(days) if days is not None else None
        self.order_by = order_by
        self.using = router.db_for_write(model)

    def execute(self, chunk_size=10000):
        if False:
            i = 10
            return i + 15
        quote_name = connections[self.using].ops.quote_name
        where = []
        if self.dtfield and self.days is not None:
            where.append("{} < '{}'::timestamptz".format(quote_name(self.dtfield), (timezone.now() - timedelta(days=self.days)).isoformat()))
        if self.project_id:
            where.append(f'project_id = {self.project_id}')
        if where:
            where_clause = 'where {}'.format(' and '.join(where))
        else:
            where_clause = ''
        if self.order_by:
            if self.order_by[0] == '-':
                direction = 'desc'
                order_field = self.order_by[1:]
            else:
                direction = 'asc'
                order_field = self.order_by
            order_clause = f'order by {quote_name(order_field)} {direction}'
        else:
            order_clause = ''
        query = '\n            delete from {table}\n            where id = any(array(\n                select id\n                from {table}\n                {where}\n                {order}\n                limit {chunk_size}\n            ));\n        '.format(table=self.model._meta.db_table, chunk_size=chunk_size, where=where_clause, order=order_clause)
        return self._continuous_query(query)

    def _continuous_query(self, query):
        if False:
            i = 10
            return i + 15
        results = True
        cursor = connections[self.using].cursor()
        while results:
            cursor.execute(query)
            results = cursor.rowcount > 0

    def iterator(self, chunk_size=100, batch_size=100000) -> Generator[tuple[int, ...], None, None]:
        if False:
            return 10
        assert self.days is not None
        assert self.dtfield is not None and self.dtfield == self.order_by
        dbc = connections[self.using]
        quote_name = dbc.ops.quote_name
        position: object | None = None
        cutoff = timezone.now() - timedelta(days=self.days)
        with dbc.get_new_connection(dbc.get_connection_params()) as conn:
            conn.autocommit = False
            chunk = []
            completed = False
            while not completed:
                with conn.cursor(uuid4().hex) as cursor:
                    where: list[tuple[str, list[Any]]] = [(f'{quote_name(self.dtfield)} < %s', [cutoff])]
                    if self.project_id:
                        where.append(('project_id = %s', [self.project_id]))
                    if self.order_by[0] == '-':
                        direction = 'desc'
                        order_field = self.order_by[1:]
                        if position is not None:
                            where.append((f'{quote_name(order_field)} <= %s', [position]))
                    else:
                        direction = 'asc'
                        order_field = self.order_by
                        if position is not None:
                            where.append((f'{quote_name(order_field)} >= %s', [position]))
                    (conditions, parameters) = zip(*where)
                    parameters = list(itertools.chain.from_iterable(parameters))
                    query = '\n                        select id, {order_field}\n                        from {table}\n                        where {conditions}\n                        order by {order_field} {direction}\n                        limit {batch_size}\n                    '.format(table=self.model._meta.db_table, conditions=' and '.join(conditions), order_field=quote_name(order_field), direction=direction, batch_size=batch_size)
                    cursor.execute(query, parameters)
                    i = 0
                    for (i, row) in enumerate(cursor, 1):
                        (key, position) = row
                        chunk.append(key)
                        if len(chunk) == chunk_size:
                            yield tuple(chunk)
                            chunk = []
                    if i < batch_size:
                        completed = True
                conn.commit()
            if chunk:
                yield tuple(chunk)