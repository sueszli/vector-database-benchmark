from io import StringIO
import numpy as np
import pytest
from pandas import DataFrame, concat, read_csv
import pandas._testing as tm

class TestInvalidConcat:

    def test_concat_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        df1 = tm.makeCustomDataframe(10, 2)
        for obj in [1, {}, [1, 2], (1, 2)]:
            msg = f"cannot concatenate object of type '{type(obj)}'; only Series and DataFrame objs are valid"
            with pytest.raises(TypeError, match=msg):
                concat([df1, obj])

    def test_concat_invalid_first_argument(self):
        if False:
            while True:
                i = 10
        df1 = tm.makeCustomDataframe(10, 2)
        msg = 'first argument must be an iterable of pandas objects, you passed an object of type "DataFrame"'
        with pytest.raises(TypeError, match=msg):
            concat(df1)

    def test_concat_generator_obj(self):
        if False:
            i = 10
            return i + 15
        concat((DataFrame(np.random.default_rng(2).random((5, 5))) for _ in range(3)))

    def test_concat_textreader_obj(self):
        if False:
            for i in range(10):
                print('nop')
        data = 'index,A,B,C,D\n                  foo,2,3,4,5\n                  bar,7,8,9,10\n                  baz,12,13,14,15\n                  qux,12,13,14,15\n                  foo2,12,13,14,15\n                  bar2,12,13,14,15\n               '
        with read_csv(StringIO(data), chunksize=1) as reader:
            result = concat(reader, ignore_index=True)
        expected = read_csv(StringIO(data))
        tm.assert_frame_equal(result, expected)