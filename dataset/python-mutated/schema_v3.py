"""Schema v3 Validation Module."""
import json
import logging
import os
from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import ValidationError
from molecule import api
from molecule.data import __file__ as data_module
LOG = logging.getLogger(__name__)

def validate(c):
    if False:
        for i in range(10):
            print('nop')
    'Perform schema validation.'
    result = []
    schemas = []
    schema_files = [os.path.dirname(data_module) + '/molecule.json']
    driver_name = c['driver']['name']
    driver_schema_file = None
    if driver_name in api.drivers():
        driver_schema_file = api.drivers()[driver_name].schema_file()
    if driver_schema_file is None:
        msg = f'Driver {driver_name} does not provide a schema.'
        LOG.warning(msg)
    elif not os.path.exists(driver_schema_file):
        msg = f'Schema {driver_schema_file} for driver {driver_name} not found.'
        LOG.warning(msg)
    else:
        schema_files.append(driver_schema_file)
    for schema_file in schema_files:
        with open(schema_file, encoding='utf-8') as f:
            schema = json.load(f)
        schemas.append(schema)
    try:
        for schema in schemas:
            jsonschema_validate(c, schema)
    except ValidationError as exc:
        if exc.json_path == '$.driver.name' and exc.message.endswith(("is not of type 'string'", 'is not valid under any of the given schemas')):
            wrong_driver_name = str(exc.message.split()[0])
            driver_name_err_msg = exc.schema['messages']['anyOf']
            result.append(f'{wrong_driver_name} {driver_name_err_msg}')
        else:
            result.append(exc.message)
    return result