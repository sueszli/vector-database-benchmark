from datetime import datetime
import pytest
from numpy import nan
from numpy.testing import assert_array_equal
from pandas import DataFrame
from superset.exceptions import InvalidPostProcessingError
from superset.utils.core import DTTM_ALIAS, PostProcessingContributionOrientation
from superset.utils.pandas_postprocessing import contribution

def test_contribution():
    if False:
        return 10
    df = DataFrame({DTTM_ALIAS: [datetime(2020, 7, 16, 14, 49), datetime(2020, 7, 16, 14, 50), datetime(2020, 7, 16, 14, 51)], 'a': [1, 3, nan], 'b': [1, 9, nan], 'c': [nan, nan, nan]})
    with pytest.raises(InvalidPostProcessingError, match='not numeric'):
        contribution(df, columns=[DTTM_ALIAS])
    with pytest.raises(InvalidPostProcessingError, match='same length'):
        contribution(df, columns=['a'], rename_columns=['aa', 'bb'])
    processed_df = contribution(df, orientation=PostProcessingContributionOrientation.ROW)
    assert processed_df.columns.tolist() == [DTTM_ALIAS, 'a', 'b', 'c']
    assert_array_equal(processed_df['a'].tolist(), [0.5, 0.25, nan])
    assert_array_equal(processed_df['b'].tolist(), [0.5, 0.75, nan])
    assert_array_equal(processed_df['c'].tolist(), [0, 0, nan])
    df.pop(DTTM_ALIAS)
    processed_df = contribution(df, orientation=PostProcessingContributionOrientation.COLUMN)
    assert processed_df.columns.tolist() == ['a', 'b', 'c']
    assert_array_equal(processed_df['a'].tolist(), [0.25, 0.75, 0])
    assert_array_equal(processed_df['b'].tolist(), [0.1, 0.9, 0])
    assert_array_equal(processed_df['c'].tolist(), [nan, nan, nan])
    processed_df = contribution(df, orientation=PostProcessingContributionOrientation.COLUMN, columns=['a'], rename_columns=['pct_a'])
    assert processed_df.columns.tolist() == ['a', 'b', 'c', 'pct_a']
    assert_array_equal(processed_df['a'].tolist(), [1, 3, nan])
    assert_array_equal(processed_df['b'].tolist(), [1, 9, nan])
    assert_array_equal(processed_df['c'].tolist(), [nan, nan, nan])
    assert processed_df['pct_a'].tolist() == [0.25, 0.75, 0]