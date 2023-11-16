from __future__ import annotations
import json
from typing import Iterator
import requests
K8S_DEFINITIONS = 'https://raw.githubusercontent.com/yannh/kubernetes-json-schema/master/v1.22.0-standalone-strict/_definitions.json'
VALUES_SCHEMA_FILE = 'chart/values.schema.json'
with open(VALUES_SCHEMA_FILE) as f:
    schema = json.load(f)

def find_refs(props: dict) -> Iterator[str]:
    if False:
        while True:
            i = 10
    for value in props.values():
        if '$ref' in value:
            yield value['$ref']
        if 'items' in value:
            if '$ref' in value['items']:
                yield value['items']['$ref']
        if 'properties' in value:
            yield from find_refs(value['properties'])

def get_remote_schema(url: str) -> dict:
    if False:
        for i in range(10):
            print('nop')
    req = requests.get(url)
    req.raise_for_status()
    return req.json()
schema['definitions'] = {k: v for (k, v) in schema.get('definitions', {}).items() if not k.startswith('io.k8s')}
defs = get_remote_schema(K8S_DEFINITIONS)
refs = set(find_refs(schema['properties']))
for step in range(15):
    starting_refs = refs
    for ref in refs:
        ref_id = ref.split('/')[-1]
        remote_def = defs['definitions'].get(ref_id)
        if remote_def:
            schema['definitions'][ref_id] = remote_def
    refs = set(find_refs(schema['definitions']))
    if refs == starting_refs:
        break
else:
    raise SystemExit("Wasn't able to find all nested references in 15 cycles")
schema['definitions'] = dict(sorted(schema['definitions'].items()))
with open(VALUES_SCHEMA_FILE, 'w') as f:
    json.dump(schema, f, indent=4)
    f.write('\n')