import gzip
import json
from io import BytesIO
from typing import Any, Dict, Optional, cast
from urllib.parse import parse_qs
import lzstring
import yaml
from django.http.multipartparser import MultiPartParser
from django.test.client import FakePayload
from jsonschema import validate

def validate_response(openapi_spec: Dict[str, Any], response: Any, path_override: Optional[str]=None):
    if False:
        while True:
            i = 10
    paths = openapi_spec['paths']
    path = path_override or response.request['PATH_INFO']
    path_spec = paths.get(path)
    assert path_spec, f'\n        Response {path} not defined in OpenAPI spec:\n        {yaml.dump(paths, indent=2)}\n    '
    response_method_spec = path_spec.get(response.request['REQUEST_METHOD'].lower())
    assert response_method_spec, f"\n        Response {response.request['REQUEST_METHOD'].lower()} not defined in OpenAPI spec:\n        {yaml.dump(path_spec, indent=2)}\n    "
    responses = response_method_spec['responses']
    response_status_spec = responses.get(str(response.status_code))
    assert response_status_spec, f'\n        Response {response.status_code} not defined in OpenAPI spec:\n        {yaml.dump(responses, indent=2)}\n    '
    response_spec = response_status_spec['content'].get('application/json')
    assert response_spec, f'\n        Response for application/json not defined in OpenAPI spec:\n        {yaml.dump(response_status_spec, indent=2)}\n    '
    validate(response.json(), response_spec)
    if response.status_code < 400 or response.status_code >= 500:
        request_fake_payload: FakePayload = response.request.get('wsgi.input', FakePayload(b''))
        request_body_content_type = response.request.get('CONTENT_TYPE', '*/*').split(';')[0]
        request_body_content_encoding = response.request.get('HTTP_CONTENT_ENCODING', None)
        request_body_value = cast(bytes, request_fake_payload._FakePayload__content.getvalue())
        if request_body_content_encoding == 'gzip':
            request_body = gzip.decompress(request_body_value)
        elif request_body_content_encoding == 'lz64':
            request_body_string = lzstring.LZString().decompressFromBase64(request_body_value.decode())
            assert request_body_string
            request_body = request_body_string
        else:
            request_body = request_body_value
        if response.request['REQUEST_METHOD'] in ['POST', 'PUT', 'PATCH', 'DELETE']:
            request_body_spec = response_method_spec['requestBody']['content'].get(request_body_content_type)
            assert request_body_spec, f"\n                Request body for {request_body_content_type} not defined in OpenAPI spec:\n                {yaml.dump(response_method_spec['requestBody']['content'], indent=2)}\n            "
            if request_body_content_type == 'multipart/form-data':
                request_body_schema = request_body_spec['schema']
                request_body_parser = MultiPartParser(response.request, BytesIO(request_body), [])
                (query_dict, _) = request_body_parser.parse()
                validate(query_dict, request_body_schema)
            elif request_body_content_type == 'application/json':
                request_body_schema = request_body_spec['schema']
                validate(json.loads(request_body), request_body_schema)
            elif request_body_content_type == '*/*':
                pass
            elif request_body_content_type == 'text/plain':
                pass
            else:
                raise Exception(f'Unknown content type: {request_body_content_type}')
        if response.request['REQUEST_METHOD'].lower() != 'options':
            query_parameter_specs = {parameter['name']: parameter for parameter in response_method_spec.get('parameters', []) if parameter['in'] == 'query'}
            sent_query_parameters = parse_qs(response.request['QUERY_STRING'])
            for (name, values) in sent_query_parameters.items():
                spec = query_parameter_specs[name]
                schema = spec['schema']
                for value in values:
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        parsed_value = value
                    validate(parsed_value, schema)
            required_parameters = {key for (key, spec) in query_parameter_specs.items() if spec.get('required')}
            for required_parameter in required_parameters:
                assert required_parameter in sent_query_parameters.keys(), f'Required parameter {required_parameter} was not sent in query string'