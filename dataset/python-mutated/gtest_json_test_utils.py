"""Unit test utilities for gtest_json_output."""
import re

def normalize(obj):
    if False:
        print('Hello World!')
    "Normalize output object.\n\n  Args:\n     obj: Google Test's JSON output object to normalize.\n\n  Returns:\n     Normalized output without any references to transient information that may\n     change from run to run.\n  "

    def _normalize(key, value):
        if False:
            while True:
                i = 10
        if key == 'time':
            return re.sub('^\\d+(\\.\\d+)?s$', '*', value)
        elif key == 'timestamp':
            return re.sub('^\\d{4}-\\d\\d-\\d\\dT\\d\\d:\\d\\d:\\d\\dZ$', '*', value)
        elif key == 'failure':
            value = re.sub('^.*[/\\\\](.*:)\\d+\\n', '\\1*\n', value)
            return re.sub('Stack trace:\\n(.|\\n)*', 'Stack trace:\n*', value)
        else:
            return normalize(value)
    if isinstance(obj, dict):
        return {k: _normalize(k, v) for (k, v) in obj.items()}
    if isinstance(obj, list):
        return [normalize(x) for x in obj]
    else:
        return obj