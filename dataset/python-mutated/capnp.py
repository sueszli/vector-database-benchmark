import os
from pathlib import Path
import capnp

def get_capnp_schema(schema_file: str) -> type:
    if False:
        i = 10
        return i + 15
    here = os.path.dirname(__file__)
    root_dir = Path(here) / '..' / 'capnp'
    capnp_path = os.path.abspath(root_dir / schema_file)
    return capnp.load(str(capnp_path))