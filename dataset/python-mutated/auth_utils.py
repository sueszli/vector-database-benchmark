"""
Utilities for checking authorization of certain resource types
"""
import logging
from typing import List, Tuple
from samcli.commands.local.lib.swagger.reader import SwaggerReader
from samcli.lib.providers.provider import Stack
from samcli.lib.providers.sam_function_provider import SamFunctionProvider
LOG = logging.getLogger(__name__)

def auth_per_resource(stacks: List[Stack]):
    if False:
        print('Hello World!')
    '\n    Check if authentication has been set for the function resources defined in the template that have `Api` Event type\n    or the function property FunctionUrlConfig.\n\n    Parameters\n    ----------\n    stacks: List[Stack]\n        The list of stacks where resources are looked for\n\n    Returns\n    -------\n\n    List of tuples per function resource that have the `Api` or `HttpApi` event types, that describes the resource\n    (function logical_id - event type or function resource logical_id and description - FURL\n    and if authorization is required per resource.\n\n    '
    _auth_per_resource: List[Tuple[str, bool]] = []
    sam_function_provider = SamFunctionProvider(stacks, ignore_code_extraction_warnings=True)
    for sam_function in sam_function_provider.get_all():
        if sam_function.events:
            _auth_resource_event(sam_function_provider, sam_function, _auth_per_resource)
        if sam_function.function_url_config:
            authorization_type = sam_function.function_url_config.get('AuthType')
            function_resource_name = f'{sam_function.name} Function Url'
            _auth_per_resource.append((function_resource_name, bool(authorization_type != 'NONE')))
    return _auth_per_resource

def _auth_resource_event(sam_function_provider: SamFunctionProvider, sam_function, auth_resource_list):
    if False:
        while True:
            i = 10
    '\n\n    Parameters\n    ----------\n    sam_function_provider: SamFunctionProvider\n    sam_function: Current function which has all intrinsics resolved.\n    auth_resource_list: List of tuples with function name and auth. eg: [("Name", True)]\n\n    Returns\n    -------\n\n    '
    for event in sam_function.events.values():
        for (event_type, identifier) in [('Api', 'RestApiId'), ('HttpApi', 'ApiId')]:
            if event.get('Type') == event_type:
                if event.get('Properties', {}).get('Auth', False):
                    auth_resource_list.append((sam_function.name, True))
                elif _auth_id(sam_function_provider.get_resources_by_stack_path(sam_function.stack_path), event.get('Properties', {}), identifier):
                    auth_resource_list.append((sam_function.name, True))
                else:
                    auth_resource_list.append((sam_function.name, False))

def _auth_id(resources_dict, event_properties, identifier):
    if False:
        print('Hello World!')
    '\n\n    Parameters\n    ----------\n    resources_dict: dict\n        Resolved resources defined in the SAM Template\n    event_properties: dict\n        Properties of given event supplied to a function resource\n    identifier: str\n        Id: `ApiId` or `RestApiId`\n\n    Returns\n    -------\n    bool\n        Returns if the given identifier under the event properties maps to a resource and has authorization enabled.\n\n    '
    resource_name = event_properties.get(identifier, '')
    api_resource = resources_dict.get(resource_name, {})
    return any([api_resource.get('Properties', {}).get('Auth', False), _auth_definition_body_and_uri(definition_body=api_resource.get('Properties', {}).get('DefinitionBody', {}), definition_uri=api_resource.get('Properties', {}).get('DefinitionUri', None))])

def _auth_definition_body_and_uri(definition_body, definition_uri):
    if False:
        return 10
    '\n\n    Parameters\n    ----------\n    definition_body: dict\n        inline definition body defined in the template\n    definition_uri: string\n        Either an s3 url or a local path to a definition uri\n\n    Returns\n    -------\n    bool\n        Is security defined on the swagger or not?\n\n\n    '
    reader = SwaggerReader(definition_body=definition_body, definition_uri=definition_uri)
    swagger = reader.read()
    _auths = []
    if not swagger:
        swagger = {}
    for (_, verb) in swagger.get('paths', {}).items():
        for _property in verb.values():
            if isinstance(_property, dict):
                _auths.append(bool(_property.get('security', False)))
    _auths.append(bool(swagger.get('security', False)))
    if swagger:
        LOG.debug('Auth checks done on swagger are not exhaustive!')
    return any(_auths)