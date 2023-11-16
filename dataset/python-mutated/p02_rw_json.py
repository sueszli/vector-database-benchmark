"""
Topic: JSON读写
Desc :
"""
import json
from collections import OrderedDict

class JSONObject:

    def __init__(self, d):
        if False:
            return 10
        self.__dict__ = d

class Point:

    def __init__(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.y = y

def serialize_instance(obj):
    if False:
        return 10
    d = {'__classname__': type(obj).__name__}
    d.update(vars(obj))
    return d

def unserialize_object(d):
    if False:
        print('Hello World!')
    clsname = d.pop('__classname__', None)
    if clsname:
        cls = classes[clsname]
        obj = cls.__new__(cls)
        for (key, value) in d.items():
            setattr(obj, key, value)
            return obj
    else:
        return d
classes = {'Point': Point}

def rw_json():
    if False:
        while True:
            i = 10
    data = {'name': 'ACME', 'shares': 100, 'price': 542.23}
    json_str = json.dumps(data)
    data = json.loads(json_str)
    with open('data.json', 'w') as f:
        json.dump(data, f)
    with open('data.json', 'r') as f:
        data = json.load(f)
    s = '{"name": "ACME", "shares": 50, "price": 490.1}'
    data = json.loads(s, object_pairs_hook=OrderedDict)
    print(data)
    print(json.dumps(data))
    print(json.dumps(data, indent=4))
    p = Point(2, 3)
    s = json.dumps(p, default=serialize_instance)
    print(s)
    a = json.loads(s, object_hook=unserialize_object)
    print(a)
if __name__ == '__main__':
    rw_json()