from __future__ import annotations
import json
from pathlib import Path
import yaml
from jsonschema import validate
CHART_DIR = Path(__file__).parents[2] / 'chart'

class TestChartQuality:
    """Tests chart quality."""

    def test_values_validate_schema(self):
        if False:
            return 10
        values = yaml.safe_load((CHART_DIR / 'values.yaml').read_text())
        schema = json.loads((CHART_DIR / 'values.schema.json').read_text())
        schema['additionalProperties'] = False
        schema['minProperties'] = len(schema['properties'].keys())
        validate(instance=values, schema=schema)