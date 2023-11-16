from collections import namedtuple
from marshmallow import missing
from marshmallow.error_store import merge_errors

def test_missing_is_falsy():
    if False:
        while True:
            i = 10
    assert bool(missing) is False
CustomError = namedtuple('CustomError', ['code', 'message'])

class TestMergeErrors:

    def test_merging_none_and_string(self):
        if False:
            i = 10
            return i + 15
        assert 'error1' == merge_errors(None, 'error1')

    def test_merging_none_and_custom_error(self):
        if False:
            i = 10
            return i + 15
        assert CustomError(123, 'error1') == merge_errors(None, CustomError(123, 'error1'))

    def test_merging_none_and_list(self):
        if False:
            for i in range(10):
                print('nop')
        assert ['error1', 'error2'] == merge_errors(None, ['error1', 'error2'])

    def test_merging_none_and_dict(self):
        if False:
            print('Hello World!')
        assert {'field1': 'error1'} == merge_errors(None, {'field1': 'error1'})

    def test_merging_string_and_none(self):
        if False:
            return 10
        assert 'error1' == merge_errors('error1', None)

    def test_merging_custom_error_and_none(self):
        if False:
            for i in range(10):
                print('nop')
        assert CustomError(123, 'error1') == merge_errors(CustomError(123, 'error1'), None)

    def test_merging_list_and_none(self):
        if False:
            for i in range(10):
                print('nop')
        assert ['error1', 'error2'] == merge_errors(['error1', 'error2'], None)

    def test_merging_dict_and_none(self):
        if False:
            for i in range(10):
                print('nop')
        assert {'field1': 'error1'} == merge_errors({'field1': 'error1'}, None)

    def test_merging_string_and_string(self):
        if False:
            while True:
                i = 10
        assert ['error1', 'error2'] == merge_errors('error1', 'error2')

    def test_merging_custom_error_and_string(self):
        if False:
            while True:
                i = 10
        assert [CustomError(123, 'error1'), 'error2'] == merge_errors(CustomError(123, 'error1'), 'error2')

    def test_merging_string_and_custom_error(self):
        if False:
            while True:
                i = 10
        assert ['error1', CustomError(123, 'error2')] == merge_errors('error1', CustomError(123, 'error2'))

    def test_merging_custom_error_and_custom_error(self):
        if False:
            i = 10
            return i + 15
        assert [CustomError(123, 'error1'), CustomError(456, 'error2')] == merge_errors(CustomError(123, 'error1'), CustomError(456, 'error2'))

    def test_merging_string_and_list(self):
        if False:
            i = 10
            return i + 15
        assert ['error1', 'error2'] == merge_errors('error1', ['error2'])

    def test_merging_string_and_dict(self):
        if False:
            print('Hello World!')
        assert {'_schema': 'error1', 'field1': 'error2'} == merge_errors('error1', {'field1': 'error2'})

    def test_merging_string_and_dict_with_schema_error(self):
        if False:
            for i in range(10):
                print('nop')
        assert {'_schema': ['error1', 'error2'], 'field1': 'error3'} == merge_errors('error1', {'_schema': 'error2', 'field1': 'error3'})

    def test_merging_custom_error_and_list(self):
        if False:
            for i in range(10):
                print('nop')
        assert [CustomError(123, 'error1'), 'error2'] == merge_errors(CustomError(123, 'error1'), ['error2'])

    def test_merging_custom_error_and_dict(self):
        if False:
            print('Hello World!')
        assert {'_schema': CustomError(123, 'error1'), 'field1': 'error2'} == merge_errors(CustomError(123, 'error1'), {'field1': 'error2'})

    def test_merging_custom_error_and_dict_with_schema_error(self):
        if False:
            for i in range(10):
                print('nop')
        assert {'_schema': [CustomError(123, 'error1'), 'error2'], 'field1': 'error3'} == merge_errors(CustomError(123, 'error1'), {'_schema': 'error2', 'field1': 'error3'})

    def test_merging_list_and_string(self):
        if False:
            i = 10
            return i + 15
        assert ['error1', 'error2'] == merge_errors(['error1'], 'error2')

    def test_merging_list_and_custom_error(self):
        if False:
            while True:
                i = 10
        assert ['error1', CustomError(123, 'error2')] == merge_errors(['error1'], CustomError(123, 'error2'))

    def test_merging_list_and_list(self):
        if False:
            for i in range(10):
                print('nop')
        assert ['error1', 'error2'] == merge_errors(['error1'], ['error2'])

    def test_merging_list_and_dict(self):
        if False:
            return 10
        assert {'_schema': ['error1'], 'field1': 'error2'} == merge_errors(['error1'], {'field1': 'error2'})

    def test_merging_list_and_dict_with_schema_error(self):
        if False:
            for i in range(10):
                print('nop')
        assert {'_schema': ['error1', 'error2'], 'field1': 'error3'} == merge_errors(['error1'], {'_schema': 'error2', 'field1': 'error3'})

    def test_merging_dict_and_string(self):
        if False:
            i = 10
            return i + 15
        assert {'_schema': 'error2', 'field1': 'error1'} == merge_errors({'field1': 'error1'}, 'error2')

    def test_merging_dict_and_custom_error(self):
        if False:
            print('Hello World!')
        assert {'_schema': CustomError(123, 'error2'), 'field1': 'error1'} == merge_errors({'field1': 'error1'}, CustomError(123, 'error2'))

    def test_merging_dict_and_list(self):
        if False:
            print('Hello World!')
        assert {'_schema': ['error2'], 'field1': 'error1'} == merge_errors({'field1': 'error1'}, ['error2'])

    def test_merging_dict_and_dict(self):
        if False:
            for i in range(10):
                print('nop')
        assert {'field1': 'error1', 'field2': ['error2', 'error3'], 'field3': 'error4'} == merge_errors({'field1': 'error1', 'field2': 'error2'}, {'field2': 'error3', 'field3': 'error4'})

    def test_deep_merging_dicts(self):
        if False:
            for i in range(10):
                print('nop')
        assert {'field1': {'field2': ['error1', 'error2']}} == merge_errors({'field1': {'field2': 'error1'}}, {'field1': {'field2': 'error2'}})