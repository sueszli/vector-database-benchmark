from datetime import date, datetime
import perspective
from pytest import mark

class TestTableInfer(object):

    def test_table_limit_wraparound_does_not_respect_partial(self):
        if False:
            for i in range(10):
                print('nop')
        t = perspective.Table({'a': float, 'b': float}, limit=3)
        t.update([{'a': 10}, {'b': 1}, {'a': 20}, {'a': None, 'b': 2}])
        df = t.view().to_df()
        t2 = perspective.Table({'a': float, 'b': float}, limit=3)
        t2.update([{'a': 10}, {'b': 1}, {'a': 20}, {'b': 2}])
        df2 = t2.view().to_df()
        assert df.to_dict() == df2.to_dict()