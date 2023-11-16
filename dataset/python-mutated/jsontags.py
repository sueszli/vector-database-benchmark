"""
Register JSON tags, so the nltk data loader knows what module and class to look for.

NLTK uses simple '!' tags to mark the types of objects, but the fully-qualified
"tag:nltk.org,2011:" prefix is also accepted in case anyone ends up
using it.
"""
import json
json_tags = {}
TAG_PREFIX = '!'

def register_tag(cls):
    if False:
        for i in range(10):
            print('nop')
    "\n    Decorates a class to register it's json tag.\n    "
    json_tags[TAG_PREFIX + getattr(cls, 'json_tag')] = cls
    return cls

class JSONTaggedEncoder(json.JSONEncoder):

    def default(self, obj):
        if False:
            return 10
        obj_tag = getattr(obj, 'json_tag', None)
        if obj_tag is None:
            return super().default(obj)
        obj_tag = TAG_PREFIX + obj_tag
        obj = obj.encode_json_obj()
        return {obj_tag: obj}

class JSONTaggedDecoder(json.JSONDecoder):

    def decode(self, s):
        if False:
            while True:
                i = 10
        return self.decode_obj(super().decode(s))

    @classmethod
    def decode_obj(cls, obj):
        if False:
            print('Hello World!')
        if isinstance(obj, dict):
            obj = {key: cls.decode_obj(val) for (key, val) in obj.items()}
        elif isinstance(obj, list):
            obj = list((cls.decode_obj(val) for val in obj))
        if not isinstance(obj, dict) or len(obj) != 1:
            return obj
        obj_tag = next(iter(obj.keys()))
        if not obj_tag.startswith('!'):
            return obj
        if obj_tag not in json_tags:
            raise ValueError('Unknown tag', obj_tag)
        obj_cls = json_tags[obj_tag]
        return obj_cls.decode_json_obj(obj[obj_tag])
__all__ = ['register_tag', 'json_tags', 'JSONTaggedEncoder', 'JSONTaggedDecoder']