from __future__ import annotations
import pytest
pytest
from math import nan
from bokeh.core.json_encoder import serialize_json
from bokeh.core.serialization import Serializer

def test_json_encoder():
    if False:
        for i in range(10):
            print('nop')
    val0 = [None, True, False, -128, -1, 0, 1, 128, nan, {'key_0': b'uvw'}]
    rep0 = Serializer().serialize(val0)
    assert rep0.buffers is not None and len(rep0.buffers) == 1
    assert serialize_json(rep0.content) == '[null,true,false,-128,-1,0,1,128,{"type":"number","value":"nan"},{"type":"map","entries":[["key_0",{"type":"bytes","data":"dXZ3"}]]}]'
    assert serialize_json(rep0) == '[null,true,false,-128,-1,0,1,128,{"type":"number","value":"nan"},{"type":"map","entries":[["key_0",{"type":"bytes","data":{"id":"%s"}}]]}]' % rep0.buffers[0].id
    assert serialize_json(rep0.content, pretty=True) == '[\n  null,\n  true,\n  false,\n  -128,\n  -1,\n  0,\n  1,\n  128,\n  {\n    "type": "number",\n    "value": "nan"\n  },\n  {\n    "type": "map",\n    "entries": [\n      [\n        "key_0",\n        {\n          "type": "bytes",\n          "data": "dXZ3"\n        }\n      ]\n    ]\n  }\n]'
    assert serialize_json(rep0, pretty=True) == '[\n  null,\n  true,\n  false,\n  -128,\n  -1,\n  0,\n  1,\n  128,\n  {\n    "type": "number",\n    "value": "nan"\n  },\n  {\n    "type": "map",\n    "entries": [\n      [\n        "key_0",\n        {\n          "type": "bytes",\n          "data": {\n            "id": "%s"\n          }\n        }\n      ]\n    ]\n  }\n]' % rep0.buffers[0].id

def test_json_encoder_dict_no_sort():
    if False:
        print('Hello World!')
    val0 = {nan: 0, 'key_1': 1, 'abc': 2, 'key_0': 3}
    rep0 = Serializer().serialize(val0)
    assert serialize_json(rep0) == '{"type":"map","entries":[[{"type":"number","value":"nan"},0],["key_1",1],["abc",2],["key_0",3]]}'