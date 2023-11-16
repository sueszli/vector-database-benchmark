import json
import os
import sys
from airbyte_cdk.models import AirbyteMessage, Type
from genson import SchemaBuilder
from genson.schema.strategies.object import Object
import genson.schema.strategies as strategies

class NoRequiredObj(Object):
    """
    This class has Object behaviour, but it does not generate "required[]" fields
    every time it parses object. So we dont add unnecessary extra field.
    """

    def to_schema(self):
        if False:
            i = 10
            return i + 15
        schema = super(Object, self).to_schema()
        schema['type'] = 'object'
        if self._properties:
            schema['properties'] = self._properties_to_schema(self._properties)
        if self._pattern_properties:
            schema['patternProperties'] = self._properties_to_schema(self._pattern_properties)
        schema['additionalProperties'] = True
        return schema

class NoRequiredSchemaBuilder(SchemaBuilder):
    EXTRA_STRATEGIES = (NoRequiredObj,)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(schema_uri='http://json-schema.org/draft-07/schema#')

def infer_schemas():
    if False:
        for i in range(10):
            print('nop')
    default_folder = os.path.join(os.getcwd(), 'schemas')
    if not os.path.exists(default_folder):
        os.mkdir(default_folder)
    builders = {}
    for line in sys.stdin:
        print(line)
        message = AirbyteMessage.parse_raw(line)
        if message.type == Type.RECORD:
            stream_name = message.record.stream
            if stream_name not in builders:
                builder = NoRequiredSchemaBuilder()
                builders[stream_name] = builder
            else:
                builder = builders[stream_name]
            builder.add_object(message.record.data)
    for (stream_name, builder) in builders.items():
        schema = builder.to_schema()
        output_file_name = os.path.join(default_folder, stream_name + '.json')
        with open(output_file_name, 'w') as outfile:
            json.dump(schema, outfile, indent=2, sort_keys=True)