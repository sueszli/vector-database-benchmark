"""
A script that generates st2 metadata (pack, action, rule, ...) schemas.
This is used by `st2-generate-schemas` to update contrib/schemas/*.json.
"""
from __future__ import absolute_import
import json
import os
import sys
from st2common.models.api import action as action_models
from st2common.models.api import pack as pack_models
from st2common.models.api import policy as policy_models
from st2common.models.api import rule as rule_models
from st2common.models.api import sensor as sensor_models
__all__ = ['generate_schemas', 'write_schemas']
content_models = {'pack': pack_models.PackAPI, 'action': action_models.ActionAPI, 'alias': action_models.ActionAliasAPI, 'policy': policy_models.PolicyAPI, 'rule': rule_models.RuleAPI, 'sensor': sensor_models.SensorTypeAPI}
default_schemas_dir = '.'

def generate_schemas():
    if False:
        for i in range(10):
            print('nop')
    for (name, model) in content_models.items():
        schema_text = json.dumps(model.schema, indent=4)
        yield (name, schema_text)

def write_schemas(schemas_dir):
    if False:
        for i in range(10):
            print('nop')
    for (name, schema_text) in generate_schemas():
        print('Generated schema for the "%s" model.' % name)
        schema_file = os.path.join(schemas_dir, name + '.json')
        print('Schema will be written to "%s".' % schema_file)
        with open(schema_file, 'w') as f:
            f.write(schema_text)
            f.write('\n')

def main():
    if False:
        i = 10
        return i + 15
    argv = sys.argv[1:]
    schemas_dir = argv[0] if argv else default_schemas_dir
    write_schemas(schemas_dir)
    return 0