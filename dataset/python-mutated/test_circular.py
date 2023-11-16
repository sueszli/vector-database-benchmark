import pytest
import orjson

class TestCircular:

    def test_circular_dict(self):
        if False:
            print('Hello World!')
        '\n        dumps() circular reference dict\n        '
        obj = {}
        obj['obj'] = obj
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(obj)

    def test_circular_dict_sort_keys(self):
        if False:
            return 10
        '\n        dumps() circular reference dict OPT_SORT_KEYS\n        '
        obj = {}
        obj['obj'] = obj
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)

    def test_circular_dict_non_str_keys(self):
        if False:
            while True:
                i = 10
        '\n        dumps() circular reference dict OPT_NON_STR_KEYS\n        '
        obj = {}
        obj['obj'] = obj
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)

    def test_circular_list(self):
        if False:
            return 10
        '\n        dumps() circular reference list\n        '
        obj = []
        obj.append(obj)
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(obj)

    def test_circular_nested(self):
        if False:
            while True:
                i = 10
        '\n        dumps() circular reference nested dict, list\n        '
        obj = {}
        obj['list'] = [{'obj': obj}]
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(obj)

    def test_circular_nested_sort_keys(self):
        if False:
            return 10
        '\n        dumps() circular reference nested dict, list OPT_SORT_KEYS\n        '
        obj = {}
        obj['list'] = [{'obj': obj}]
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)

    def test_circular_nested_non_str_keys(self):
        if False:
            return 10
        '\n        dumps() circular reference nested dict, list OPT_NON_STR_KEYS\n        '
        obj = {}
        obj['list'] = [{'obj': obj}]
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)