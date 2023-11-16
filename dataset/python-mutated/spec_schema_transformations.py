import json
import re
from jsonschema import RefResolver

def resolve_refs(schema: dict) -> dict:
    if False:
        return 10
    '\n    For spec schemas generated using Pydantic models, the resulting JSON schema can contain refs between object\n    relationships.\n    '
    json_schema_ref_resolver = RefResolver.from_schema(schema)
    str_schema = json.dumps(schema)
    for ref_block in re.findall('{"\\$ref": "#\\/definitions\\/.+?(?="})"}', str_schema):
        ref = json.loads(ref_block)['$ref']
        str_schema = str_schema.replace(ref_block, json.dumps(json_schema_ref_resolver.resolve(ref)[1]))
    pyschema: dict = json.loads(str_schema)
    del pyschema['definitions']
    return pyschema