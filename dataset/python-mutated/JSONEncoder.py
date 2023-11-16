import collections
import json
import re
from datetime import datetime
from coala_utils.decorators import get_public_members
from coalib.settings.FunctionMetadata import FunctionMetadata

def create_json_encoder(**kwargs):
    if False:
        for i in range(10):
            print('nop')

    class JSONEncoder(json.JSONEncoder):

        @classmethod
        def _filter_params(cls, op, nop):
            if False:
                i = 10
                return i + 15
            params = set(op) | set(nop)
            return {key: kwargs[key] for key in set(kwargs) & params}

        def default(self, obj):
            if False:
                return 10
            if hasattr(obj, '__json__'):
                fdata = FunctionMetadata.from_function(obj.__json__)
                params = self._filter_params(fdata.optional_params, fdata.non_optional_params)
                return obj.__json__(**params)
            elif isinstance(obj, collections.Iterable):
                return list(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__getitem__') and hasattr(obj, 'keys'):
                return dict(obj)
            elif hasattr(obj, '__dict__'):
                return {member: getattr(obj, member) for member in get_public_members(obj)}
            elif isinstance(obj, re._pattern_type):
                return obj.pattern
            return json.JSONEncoder.default(self, obj)
    return JSONEncoder