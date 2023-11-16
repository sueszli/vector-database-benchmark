from flatc_test import *
import json

class SchemaTests:

    def EnumValAttributes(self):
        if False:
            return 10
        flatc(['--schema', '--binary', '--bfbs-builtins', 'enum_val_attributes.fbs'])
        assert_file_exists('enum_val_attributes.bfbs')
        flatc(['--json', '--strict-json', str(reflection_fbs_path()), '--', 'enum_val_attributes.bfbs'])
        schema_json = json.loads(get_file_contents('enum_val_attributes.json'))
        assert schema_json['enums'][0]['name'] == 'ValAttributes'
        assert schema_json['enums'][0]['values'][0]['name'] == 'Val1'
        assert schema_json['enums'][0]['values'][0]['attributes'][0]['key'] == 'display_name'
        assert schema_json['enums'][0]['values'][0]['attributes'][0]['value'] == 'Value 1'
        assert schema_json['enums'][0]['values'][1]['name'] == 'Val2'
        assert schema_json['enums'][0]['values'][1]['attributes'][0]['key'] == 'display_name'
        assert schema_json['enums'][0]['values'][1]['attributes'][0]['value'] == 'Value 2'
        assert schema_json['enums'][0]['values'][2]['name'] == 'Val3'
        assert schema_json['enums'][0]['values'][2]['attributes'][0]['key'] == 'deprecated'
        assert schema_json['enums'][0]['values'][2]['attributes'][1]['key'] == 'display_name'
        assert schema_json['enums'][0]['values'][2]['attributes'][1]['value'] == 'Value 3 (deprecated)'