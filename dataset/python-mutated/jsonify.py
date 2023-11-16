from __future__ import annotations
import json

def jsonify(result, format=False):
    if False:
        print('Hello World!')
    ' format JSON output (uncompressed or uncompressed) '
    if result is None:
        return '{}'
    indent = None
    if format:
        indent = 4
    try:
        return json.dumps(result, sort_keys=True, indent=indent, ensure_ascii=False)
    except UnicodeDecodeError:
        return json.dumps(result, sort_keys=True, indent=indent)