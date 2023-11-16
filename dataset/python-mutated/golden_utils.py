import sys
from pathlib import Path
script_path = Path(__file__).parent.resolve()
root_path = script_path.parent.absolute()
schema_path = Path(script_path, 'schema')
sys.path.append(str(root_path.absolute()))
from scripts.util import flatc

def flatc_golden(options, schema, prefix):
    if False:
        i = 10
        return i + 15
    flatc(options=options, prefix=prefix, schema=str(Path(schema_path, schema)), cwd=script_path)