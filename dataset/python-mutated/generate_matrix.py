import json
from pathlib import Path
from typing import Any
THIS_DIR = Path(__file__).parent.resolve()

def generate_matrix() -> dict[str, Any]:
    if False:
        return 10
    dockerfiles = sorted((file.name for file in THIS_DIR.glob('*.dockerfile')))
    return {'dockerfile': dockerfiles, 'runsOn': ['ubuntu-latest', 'ARM64'], 'exclude': [{'dockerfile': 'oracledb.dockerfile', 'runsOn': 'ARM64'}]}
if __name__ == '__main__':
    print('matrix=' + json.dumps(generate_matrix()))