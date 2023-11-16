from drf_yasg.inspectors.view import SwaggerAutoSchema
from drf_yasg.openapi import resolve_ref, Schema
from .utils import resolve_lazy_ref
import copy

class ComposableSchema:
    """A composable schema defines a transformation on drf_yasg Operation. These
    schema can then be composed with another composable schema using the composeWith method
    yielding a new composable schema whose transformation is defined as the function composition
    of the transformation of the two source schema.
    """

    def transform_operation(self, operation, resolver):
        if False:
            return 10
        'Defines an operation transformation\n\n        Args:\n            operation (Operation): the operation to transform\n            resolver (Resolver): the schema refs resolver\n        '

    def composeWith(self, schema):
        if False:
            i = 10
            return i + 15
        "Allow two schema to be composed into a new schema.\n        Given the caller schema 'self' and another schema 'schema',\n        this operation yields a new composable schema whose transform_operation\n        if defined as\n            transform_operation(op, res) = schema.transform_operation(self.transform_operation(op, res), res)\n\n        Args:\n            schema (ComposableSchema): The schema to compose with\n\n        Returns:\n            ComposableSchema: the newly composed schema\n        "
        op = self.transform_operation

        class _Wrapper(ComposableSchema):

            def transform_operation(self, operation, resolver):
                if False:
                    i = 10
                    return i + 15
                return schema.transform_operation(op(operation, resolver), resolver)
        return _Wrapper()

    def to_schema(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert the composable schema into a SwaggerAutoSchema that\n        can be used with the drf_yasg library code\n\n        Returns:\n            SwaggerAutoSchema: the swagger auto schema derived from the composable schema\n        '
        op = self.transform_operation

        class _Schema(SwaggerAutoSchema):

            def __init__(self, *args, **kwargs):
                if False:
                    return 10
                super().__init__(*args, **kwargs)

            def get_operation(self, operation_keys):
                if False:
                    print('Hello World!')
                operation = super().get_operation(operation_keys)
                return op(operation, self.components)
        return _Schema

class IdentitySchema(ComposableSchema):

    def transform_operation(self, operation, resolver):
        if False:
            print('Hello World!')
        return operation

class ExtraParameters(ComposableSchema):
    """Define a schema that can add parameters to the operation"""

    def __init__(self, operation_name, extra_parameters, *args, **kwargs):
        if False:
            print('Hello World!')
        'Initialize the schema\n\n        Args:\n            operation_name (string): the name of the operation to transform\n            extra_parameters (list[Parameter]): list of openapi parameters to add\n        '
        super().__init__(*args, **kwargs)
        self._extra_parameters = extra_parameters
        self._operation_name = operation_name

    def transform_operation(self, operation, resolver):
        if False:
            print('Hello World!')
        operation_id = operation['operationId']
        if not operation_id.endswith(self._operation_name):
            return operation
        for param in self._extra_parameters:
            operation['parameters'].append(resolve_lazy_ref(param, resolver))
        return operation

class ExtraResponseField(ComposableSchema):
    """Define a schema that can add fields to the responses of the operation"""

    def __init__(self, operation_name, extra_fields, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Initialize the schema\n\n        Args:\n            operation_name (string): the name of the operation to transform\n            extra_fields (dict()): description of the fields to add to the responses. The format is\n            {\n                parameters: list[openapi.Parameter](params1, params2, ...),\n                responses: {\n                    code1: {\n                        field1: openapi.Schema,\n                        field2: openapi.Schema,\n                        ...\n                    },\n                    code2: ...\n                }\n            }\n        '
        super().__init__(*args, **kwargs)
        self._extra_fields = extra_fields
        self._operation_name = operation_name

    def transform_operation(self, operation, resolver):
        if False:
            print('Hello World!')
        operation_id = operation['operationId']
        if not operation_id.endswith(self._operation_name):
            return operation
        responses = operation['responses']
        for (code, params) in self._extra_fields.items():
            if code in responses:
                original_schema = responses[code]['schema']
                schema = original_schema if isinstance(original_schema, Schema) else resolve_ref(original_schema, resolver)
                schema = copy.deepcopy(schema)
                for (name, param) in params.items():
                    schema['properties'][name] = resolve_lazy_ref(param, resolver)
                responses[code]['schema'] = schema
        return operation