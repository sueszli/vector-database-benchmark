from datetime import datetime
from enum import IntEnum
from pydantic import Field
from ninja import NinjaAPI, Query, Schema
from ninja.testing import TestClient

class Range(IntEnum):
    TWENTY = 20
    FIFTY = 50
    TWO_HUNDRED = 200

class Filter(Schema):
    to_datetime: datetime = Field(alias='to')
    from_datetime: datetime = Field(alias='from')
    range: Range = Range.TWENTY

class Data(Schema):
    an_int: int = Field(alias='int', default=0)
    a_float: float = Field(alias='float', default=1.5)
api = NinjaAPI()

@api.get('/test')
def query_params_schema(request, filters: Filter=Query(...)):
    if False:
        return 10
    return filters.dict()

@api.get('/test-mixed')
def query_params_mixed_schema(request, query1: int, query2: int=5, filters: Filter=Query(...), data: Data=Query(...)):
    if False:
        print('Hello World!')
    return dict(query1=query1, query2=query2, filters=filters.dict(), data=data.dict())

def test_schema():
    if False:
        return 10
    schema = api.get_openapi_schema()
    params = schema['paths']['/api/test']['get']['parameters']
    print(params)
    assert params == [{'in': 'query', 'name': 'to', 'schema': {'format': 'date-time', 'title': 'To', 'type': 'string'}, 'required': True}, {'in': 'query', 'name': 'from', 'schema': {'format': 'date-time', 'title': 'From', 'type': 'string'}, 'required': True}, {'in': 'query', 'name': 'range', 'schema': {'allOf': [{'enum': [20, 50, 200], 'title': 'Range', 'type': 'integer'}], 'default': 20}, 'required': False}]

def test_schema_all_of_no_ref():
    if False:
        for i in range(10):
            print('nop')
    details = {'default': 1, 'allOf': [{'$ref': '#/components/schemas/Type'}, {'no-ref-here': 'xyzzy'}]}
    definitions = {'Type': {'title': 'Best Type Ever!'}}
    from ninja.openapi.schema import resolve_allOf
    resolve_allOf(details, definitions)
    assert details == {'default': 1, 'allOf': [{'title': 'Best Type Ever!'}, {'no-ref-here': 'xyzzy'}]}