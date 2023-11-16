from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Mapping, MutableMapping, Optional, Type, Union
from airbyte_cdk.sources.declarative.interpolation import InterpolatedString
from airbyte_cdk.sources.declarative.requesters.http_requester import HttpRequester
from airbyte_cdk.sources.declarative.schema.json_file_schema_loader import JsonFileSchemaLoader
from airbyte_cdk.sources.declarative.types import StreamSlice, StreamState

@dataclass
class MondayGraphqlRequester(HttpRequester):
    NEXT_PAGE_TOKEN_FIELD_NAME = 'next_page_token'
    limit: Union[InterpolatedString, str, int] = None
    nested_limit: Union[InterpolatedString, str, int] = None

    def __post_init__(self, parameters: Mapping[str, Any]):
        if False:
            for i in range(10):
                print('nop')
        super(MondayGraphqlRequester, self).__post_init__(parameters)
        self.limit = InterpolatedString.create(self.limit, parameters=parameters)
        self.nested_limit = InterpolatedString.create(self.nested_limit, parameters=parameters)
        self.name = parameters.get('name', '').lower()

    def _ensure_type(self, t: Type, o: Any):
        if False:
            return 10
        '\n        Ensure given object `o` is of type `t`\n        '
        if not isinstance(o, t):
            raise TypeError(f'{type(o)} {o} is not of type {t}')

    def _get_schema_root_properties(self):
        if False:
            while True:
                i = 10
        schema_loader = JsonFileSchemaLoader(config=self.config, parameters={'name': self.name})
        schema = schema_loader.get_json_schema()['properties']
        delete_fields = ['updated_at_int', 'created_at_int', 'pulse_id']
        if self.name == 'activity_logs':
            delete_fields.append('board_id')
        for field in delete_fields:
            if field in schema:
                schema.pop(field)
        return schema

    def _get_object_arguments(self, **object_arguments) -> str:
        if False:
            print('Hello World!')
        return ','.join([f'{argument}:{value}' if argument != 'fromt' else f'from:"{value}"' for (argument, value) in object_arguments.items() if value is not None])

    def _build_query(self, object_name: str, field_schema: dict, **object_arguments) -> str:
        if False:
            return 10
        '\n        Recursive function that builds a GraphQL query string by traversing given stream schema properties.\n        Attributes\n            object_name (str): the name of root object\n            field_schema (dict): configured catalog schema for current stream\n            object_arguments (dict): arguments such as limit, page, ids, ... etc to be passed for given object\n        '
        fields = []
        for (field, nested_schema) in field_schema.items():
            nested_fields = nested_schema.get('properties', nested_schema.get('items', {}).get('properties'))
            if nested_fields:
                fields.append(self._build_query(field, nested_fields))
            else:
                fields.append(field)
        arguments = self._get_object_arguments(**object_arguments)
        arguments = f'({arguments})' if arguments else ''
        fields = ','.join(fields)
        return f'{object_name}{arguments}{{{fields}}}'

    def _build_items_query(self, object_name: str, field_schema: dict, sub_page: Optional[int], **object_arguments) -> str:
        if False:
            while True:
                i = 10
        '\n        Special optimization needed for items stream. Starting October 3rd, 2022 items can only be reached through boards.\n        See https://developer.monday.com/api-reference/docs/items-queries#items-queries\n        '
        nested_limit = self.nested_limit.eval(self.config)
        query = self._build_query('items', field_schema, limit=nested_limit, page=sub_page)
        arguments = self._get_object_arguments(**object_arguments)
        return f'boards({arguments}){{{query}}}'

    def _build_items_incremental_query(self, object_name: str, field_schema: dict, stream_slice: dict, **object_arguments) -> str:
        if False:
            return 10
        '\n        Special optimization needed for items stream. Starting October 3rd, 2022 items can only be reached through boards.\n        See https://developer.monday.com/api-reference/docs/items-queries#items-queries\n        '
        nested_limit = self.nested_limit.eval(self.config)
        object_arguments['limit'] = nested_limit
        object_arguments['ids'] = stream_slice['ids']
        return self._build_query('items', field_schema, **object_arguments)

    def _build_teams_query(self, object_name: str, field_schema: dict, **object_arguments) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Special optimization needed for tests to pass successfully because of rate limits.\n        It makes a query cost less points, but it is never used in production\n        '
        teams_limit = self.config.get('teams_limit')
        if teams_limit:
            self._ensure_type(int, teams_limit)
            arguments = self._get_object_arguments(**object_arguments)
            query = f'{{id,name,picture_url,users(limit:{teams_limit}){{id}}}}'
            return f'{object_name}({arguments}){query}'
        return self._build_query(object_name=object_name, field_schema=field_schema, **object_arguments)

    def _build_activity_query(self, object_name: str, field_schema: dict, sub_page: Optional[int], **object_arguments) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Special optimization needed for items stream. Starting October 3rd, 2022 items can only be reached through boards.\n        See https://developer.monday.com/api-reference/docs/items-queries#items-queries\n        '
        nested_limit = self.nested_limit.eval(self.config)
        created_at = (object_arguments.get('stream_state', dict()) or dict()).get('created_at_int')
        object_arguments.pop('stream_state')
        if created_at:
            created_at = datetime.fromtimestamp(created_at).strftime('%Y-%m-%dT%H:%M:%SZ')
        query = self._build_query(object_name, field_schema, limit=nested_limit, page=sub_page, fromt=created_at)
        arguments = self._get_object_arguments(**object_arguments)
        return f'boards({arguments}){{{query}}}'

    def get_request_params(self, *, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> MutableMapping[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Combines queries to a single GraphQL query.\n        '
        limit = self.limit.eval(self.config)
        page = next_page_token and next_page_token[self.NEXT_PAGE_TOKEN_FIELD_NAME]
        if self.name == 'boards' and stream_slice:
            query_builder = partial(self._build_query, **stream_slice)
        elif self.name == 'items':
            (page, sub_page) = page if page else (None, None)
            if not stream_slice:
                query_builder = partial(self._build_items_query, sub_page=sub_page)
            else:
                query_builder = partial(self._build_items_incremental_query, stream_slice=stream_slice)
        elif self.name == 'teams':
            query_builder = self._build_teams_query
        elif self.name == 'activity_logs':
            (page, sub_page) = page if page else (None, None)
            query_builder = partial(self._build_activity_query, sub_page=sub_page, stream_state=stream_state)
        else:
            query_builder = self._build_query
        query = query_builder(object_name=self.name, field_schema=self._get_schema_root_properties(), limit=limit or None, page=page)
        return {'query': f'query{{{query}}}'}

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(tuple(self.__dict__))