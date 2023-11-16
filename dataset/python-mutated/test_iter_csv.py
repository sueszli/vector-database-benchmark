from __future__ import annotations
import io
from river import stream

def test_iter_csv_custom_converter():
    if False:
        i = 10
        return i + 15
    example = io.StringIO('col1,col2,col3\n,1,2\n5,,4\n3,1,')

    def int_or_none(s):
        if False:
            print('Hello World!')
        try:
            return int(s)
        except ValueError:
            return None
    params = {'converters': {'col1': int_or_none, 'col2': int_or_none, 'col3': int_or_none}}
    dataset = stream.iter_csv(example, **params)
    assert list(dataset) == [({'col1': None, 'col2': 1, 'col3': 2}, None), ({'col1': 5, 'col2': None, 'col3': 4}, None), ({'col1': 3, 'col2': 1, 'col3': None}, None)]

def test_iter_csv_drop_nones():
    if False:
        print('Hello World!')
    example = io.StringIO('col1,col2,col3\n,1,2\n5,,4\n3,1,')

    def int_or_none(s):
        if False:
            i = 10
            return i + 15
        try:
            return int(s)
        except ValueError:
            return None
    params = {'converters': {'col1': int_or_none, 'col2': int_or_none, 'col3': int_or_none}, 'drop_nones': True}
    dataset = stream.iter_csv(example, **params)
    assert list(dataset) == [({'col2': 1, 'col3': 2}, None), ({'col1': 5, 'col3': 4}, None), ({'col1': 3, 'col2': 1}, None)]