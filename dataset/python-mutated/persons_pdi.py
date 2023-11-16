from typing import Dict, List
from posthog.hogql.ast import SelectQuery
from posthog.hogql.context import HogQLContext
from posthog.hogql.database.argmax import argmax_select
from posthog.hogql.database.models import IntegerDatabaseField, StringDatabaseField, LazyTable, FieldOrTable
from posthog.hogql.errors import HogQLException
from posthog.schema import HogQLQueryModifiers

def persons_pdi_select(requested_fields: Dict[str, List[str]]):
    if False:
        for i in range(10):
            print('nop')
    if 'person_id' not in requested_fields:
        requested_fields = {**requested_fields, 'person_id': ['person_id']}
    return argmax_select(table_name='raw_person_distinct_ids', select_fields=requested_fields, group_fields=['distinct_id'], argmax_field='version', deleted_field='is_deleted')

def persons_pdi_join(from_table: str, to_table: str, requested_fields: Dict[str, List[str]], context: HogQLContext, node: SelectQuery):
    if False:
        while True:
            i = 10
    from posthog.hogql import ast
    if not requested_fields:
        raise HogQLException('No fields requested from person_distinct_ids')
    join_expr = ast.JoinExpr(table=persons_pdi_select(requested_fields))
    join_expr.join_type = 'INNER JOIN'
    join_expr.alias = to_table
    join_expr.constraint = ast.JoinConstraint(expr=ast.CompareOperation(op=ast.CompareOperationOp.Eq, left=ast.Field(chain=[from_table, 'id']), right=ast.Field(chain=[to_table, 'person_id'])))
    return join_expr

class PersonsPDITable(LazyTable):
    fields: Dict[str, FieldOrTable] = {'team_id': IntegerDatabaseField(name='team_id'), 'distinct_id': StringDatabaseField(name='distinct_id'), 'person_id': StringDatabaseField(name='person_id')}

    def lazy_select(self, requested_fields: Dict[str, List[str]], modifiers: HogQLQueryModifiers):
        if False:
            return 10
        return persons_pdi_select(requested_fields)

    def to_printed_clickhouse(self, context):
        if False:
            i = 10
            return i + 15
        return 'person_distinct_id2'

    def to_printed_hogql(self):
        if False:
            print('Hello World!')
        return 'person_distinct_ids'