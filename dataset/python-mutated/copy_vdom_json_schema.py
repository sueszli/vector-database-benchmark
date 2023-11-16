import json
from pathlib import Path
from sphinx.application import Sphinx
from reactpy.core.vdom import VDOM_JSON_SCHEMA

def setup(app: Sphinx) -> None:
    if False:
        for i in range(10):
            print('nop')
    schema_file = Path(__file__).parent.parent / 'vdom-json-schema.json'
    current_schema = json.dumps(VDOM_JSON_SCHEMA, indent=2, sort_keys=True)
    if not schema_file.exists() or schema_file.read_text() != current_schema:
        schema_file.write_text(current_schema)