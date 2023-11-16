"""
This module provides a raw, untyped representation for daemon query
responses which handles unwrapping the outer layer of responses sent
by the classic daemon. Client code must decode the `object`-typed
`response_json` field, which requires knowing the expected shape
of the response for any given query.
"""
from __future__ import annotations
import dataclasses
import json

class InvalidQueryResponse(Exception):
    pass

@dataclasses.dataclass(frozen=True)
class Response:
    payload: object

    @staticmethod
    def from_json(response_json: object) -> Response:
        if False:
            while True:
                i = 10
        if isinstance(response_json, list) and len(response_json) > 1 and (response_json[0] == 'Query'):
            return Response(response_json[1])
        else:
            raise InvalidQueryResponse(f'Unexpected JSON response from server: {response_json}')

    @staticmethod
    def parse(response_text: str) -> Response:
        if False:
            for i in range(10):
                print('nop')
        try:
            response_json = json.loads(response_text)
            return Response.from_json(response_json)
        except json.JSONDecodeError as decode_error:
            message = f'Cannot parse response as JSON: {decode_error}'
            raise InvalidQueryResponse(message) from decode_error